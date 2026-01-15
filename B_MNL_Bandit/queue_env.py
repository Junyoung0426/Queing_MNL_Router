from typing import List, Tuple, Optional
from itertools import combinations
import math

import numpy as np
import torch

from mnl_router import MNLRouter
from queue_config import QueueConfig


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


class QueueEnv:
    def __init__(
        self,
        X_ctx: np.ndarray,
        acc_mat: np.ndarray,
        util_mat: np.ndarray,
        config: QueueConfig,
        router: MNLRouter,
        model_names: Optional[List[str]] = None,
        row_ids: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
    ):
        self.X_ctx = X_ctx
        self.acc_mat = acc_mat
        self.util_mat = util_mat
        self.config = config

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.device = torch.device(config.device)

        N, d_ctx = X_ctx.shape
        _, n_models = acc_mat.shape

        self.N = int(N)
        self.d_ctx = int(d_ctx)
        self.n_models = int(n_models)

        self.config.d_ctx = self.d_ctx
        self.config.n_models = self.n_models

        self.lambda_0 = float(self.config.lambda_0)

        # router: float32 end-to-end 가정
        self.router = router.to(self.device)
        self.router.eval()

        self.model_names = model_names
        self.idx2model = list(self.model_names) if self.model_names is not None else None

        self.row_ids = row_ids
        self.sample_ids = sample_ids

        if self.row_ids is not None:
            assert len(self.row_ids) == self.N, f"row_ids length {len(self.row_ids)} != N {self.N}"
        if self.sample_ids is not None:
            assert len(self.sample_ids) == self.N, f"sample_ids length {len(self.sample_ids)} != N {self.N}"

        self.explore_enabled = bool(getattr(self.config, "explore_enabled", True))

        self.K = int(config.assort_K)
        self.all_combis = list(combinations(range(self.n_models), self.K))
        self.C_size = len(self.all_combis)

        self.rng_arrival = np.random.RandomState(config.seed)
        self.rng_explore = np.random.RandomState(config.seed + 1)
        self.rng_router = np.random.RandomState(config.seed + 2)
        self.rng_oracle = np.random.RandomState(config.seed + 3)
        self.rng_feedback = np.random.RandomState(config.seed + 12345)

        self.job_pool = np.arange(self.N, dtype=np.int64)

        self.next_uid: int = 0
        self.queue_router: List[Tuple[int, int]] = []
        self.queue_oracle: List[Tuple[int, int]] = []
        self.last_arrival_uid: Optional[int] = None

        self.A_prev = 0
        self.E_prev = 0
        self.steps = 0
        self.comb_idx = 0

        self.cum_regret = 0.0

        self.regret_history: List[float] = []
        self.Q_regret_history: List[float] = []
        self.Q_router_history: List[int] = []
        self.Q_oracle_history: List[int] = []

        self.kappa = float(self.config.kappa)
        self.c1 = float(self.config.c1)
        self.d = int(getattr(self.router, "d", self.config.d_proj))

        denom_M = math.log(1.0 - 1.0 / (4.0 * math.sqrt(math.e * math.pi)))
        self.M_sample = max(1, int(math.ceil(1.0 - math.log(self.K) / denom_M)))

        self.exploit_job_batch = 128

        # tensor caches (float32)
        self.X_ctx_t = torch.from_numpy(self.X_ctx).to(self.device, dtype=torch.float32)  # (N,d_ctx)
        self.S_tensor = torch.tensor(self.all_combis, device=self.device, dtype=torch.long)  # (C,K)

        # -----------------------------
        # ground-truth odds: util -> global minmax -> r -> odds
        # -----------------------------
        self.r_eps = float(self.config.r_eps)
        self.r_lo = float(self.config.r_lo)
        self.r_hi = float(self.config.r_hi)

        u = self.util_mat.astype(np.float32, copy=False)

        u_min = np.nanmin(u)
        u_max = np.nanmax(u)
        den = (u_max - u_min)
        den = 1.0 if den < 1e-12 else den

        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.u_den = float(den)

        u01 = (u - self.u_min) / self.u_den
        r = self.r_lo + (self.r_hi - self.r_lo) * u01
        r = np.clip(r, self.r_lo, self.r_hi)

        self.r_mat = r
        self.odds_mat = r / np.maximum(1.0 - r, self.r_eps)

        # best S per ctx by true odds
        self.max_departure_rates = np.zeros(self.N, dtype=np.float32)
        self.best_S_idx = np.zeros(self.N, dtype=np.int64)

        combi_idx_np = np.asarray(self.all_combis, dtype=np.int64)
        for i in range(self.N):
            odds_row = self.odds_mat[i]
            odds_S = odds_row[combi_idx_np]      # (C,K)
            sum_odds = odds_S.sum(axis=1)        # (C,)
            rates = sum_odds / (1.0 + sum_odds)  # (C,)
            best_c = int(np.argmax(rates))
            self.best_S_idx[i] = best_c
            self.max_departure_rates[i] = float(rates[best_c])

        self.debug_verbose = bool(self.config.debug_verbose)
        self.debug_topk = int(self.config.debug_topk)
        self.debug_print_every = int(self.config.log_every)

        self._last_S_router: Optional[List[int]] = None
        self._last_ctx_router: Optional[int] = None
        self._last_uid_router: Optional[int] = None

        self._last_S_oracle: Optional[List[int]] = None
        self._last_ctx_oracle: Optional[int] = None
        self._last_uid_oracle: Optional[int] = None

        self._last_S_star_ctx: Optional[List[int]] = None
        self._last_choice_router: Optional[Tuple[bool, Optional[int]]] = None
        self._last_choice_oracle: Optional[Tuple[bool, Optional[int]]] = None

        self._last_router_metrics: Optional[dict] = None
        self._last_oracle_metrics: Optional[dict] = None

        self.cnt_decision = 0
        self.cnt_explore = 0

        # -----------------------------
        # Departure logs (decision-step only)
        #   - queue 비었을 때는 기록하지 않는다
        # -----------------------------
        self.dep_prob_router_hist: List[float] = []   # dep_alg_true (Router)
        self.dep_event_router_hist: List[float] = []  # 1 if departed else 0 (Router)
        self.dep_router_step_idx: List[int] = []      # 어떤 step에서 기록했는지

        self.dep_prob_oracle_hist: List[float] = []   # oracle(queue progression) 기대 dep
        self.dep_event_oracle_hist: List[float] = []  # oracle 실제 departed(0/1)
        self.dep_oracle_step_idx: List[int] = []

        self.dep_prob_star_hist: List[float] = []     # oracle@queue-max 기대 dep*
        self.dep_star_step_idx: List[int] = []

        print(
            f"[Queue-Env] N={self.N}, n_models={self.n_models}, |C|={self.C_size}, "
            f"ArrRate={self.config.arrival_rate}, MaxSteps={self.config.max_steps}, "
            f"d={self.d}, M={self.M_sample}, B_type={self.config.b_type}, "
            f"ExploreEnabled={self.explore_enabled}"
        )
        print(
            f"[Queue-Env] global-minmax(util): u_min={self.u_min:.6f} u_max={self.u_max:.6f} den={self.u_den:.6f} "
            f"r_lo={self.r_lo} r_hi={self.r_hi} eps={self.r_eps}"
        )

    def _name(self, i: int) -> str:
        if self.idx2model is None:
            return str(i)
        return self.idx2model[int(i)]

    def _ctx_info(self, ctx_idx: int) -> str:
        s = f"ctx_idx={int(ctx_idx)}"
        if self.row_ids is not None:
            s += f" df_row={int(self.row_ids[int(ctx_idx)])}"
        if self.sample_ids is not None:
            s += f" sample_id={self.sample_ids[int(ctx_idx)]}"
        return s

    # Ground-truth(진짜 값) 기반 함수들 Router가 고른 S기반
    def _dep_rate_from_odds_row(self, odds_row: np.ndarray, S: List[int]) -> float:
        odds = odds_row[np.array(S, dtype=np.int64)]
        s = float(odds.sum())
        return float(s / (1.0 + s))

    # Ground-truth(진짜 값) 기반 함수들 Router가 고른 X로 위에 식 사용
    def true_departure_rate(self, ctx_idx: int, S: List[int]) -> float:
        return self._dep_rate_from_odds_row(self.odds_mat[ctx_idx], S)

    #(S안의 각 모델의 r)
    def _true_r_list(self, r_row: np.ndarray, S: List[int]) -> List[float]:
        return [float(r_row[int(i)]) for i in S]

    def _topk_models_by_true_r(self, r_row: np.ndarray, k: int = 3) -> List[int]:
        k = min(int(k), r_row.shape[0])
        return list(np.argsort(-r_row)[:k].astype(int))

    # --- predict utilities (float32) ---
    def _pred_u_list_from_theta(self, x_ctx: np.ndarray, S: List[int]) -> List[float]:
        x_t = torch.from_numpy(x_ctx).to(self.device, dtype=torch.float32)
        S_t = torch.tensor(S, device=self.device, dtype=torch.long)
        with torch.no_grad():
            z_S = self.router.z_for_S_from_ctx(x_t, S_t)  # (K,d) float32
            u_hat = self.router.scores_for_S_from_z(z_S)  # (K,) float32
        return [float(v) for v in u_hat.detach().cpu().tolist()]

    def _pred_u_all_models(self, x_ctx: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x_ctx).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            u = self.router.logits_all_models(x_t).squeeze(0)  # (n_models,) float32
        return u.detach().cpu().numpy().astype(np.float32)

    def _pred_r_all_models(self, x_ctx: np.ndarray) -> np.ndarray:
        return _sigmoid_np(self._pred_u_all_models(x_ctx))

    def _topk_models_by_pred_r(self, x_ctx: np.ndarray, k: int = 3) -> List[int]:
        r_hat_all = self._pred_r_all_models(x_ctx)
        k = min(int(k), r_hat_all.shape[0])
        return list(np.argsort(-r_hat_all)[:k].astype(int))

    def sample_mnl_choice_u(self, ctx_idx: int, S: List[int], u: float):
        odds_row = self.odds_mat[ctx_idx]
        odds_S = odds_row[np.array(S, dtype=np.int64)]
        sum_odds = float(odds_S.sum())
        denom = 1.0 + sum_odds

        p_out = 1.0 / denom
        p_in = odds_S / denom

        probs = np.empty(len(S) + 1, dtype=np.float32)
        probs[0] = p_out
        probs[1:] = p_in
        probs /= probs.sum()

        cdf = np.cumsum(probs)
        j = int(np.searchsorted(cdf, float(u), side="right"))

        if j == 0:
            return False, None, None
        j_local = j - 1
        return True, S[j_local], j_local

    def _select_best_by_exhaustive_batch(self, noise_vectors: torch.Tensor) -> Tuple[int, int, List[int]]:
        q = self.queue_router
        q_ctx = torch.tensor([ctx for (_, ctx) in q], device=self.device, dtype=torch.long)
        q_pos = torch.arange(len(q), device=self.device, dtype=torch.long)

        C = self.C_size
        K = self.K
        d = self.d

        best_val = -1.0
        best_qpos = 0
        best_ctx = int(q_ctx[0].item()) if q_ctx.numel() > 0 else -1
        best_c = 0

        with torch.no_grad():
            a_S_all = self.router.a_table[self.S_tensor]  # (C,K,d) float32

        bs = self.exploit_job_batch
        for st in range(0, q_pos.numel(), bs):
            ed = min(st + bs, q_pos.numel())
            pos_chunk = q_pos[st:ed]
            ctx_chunk = q_ctx[st:ed]

            Xb = self.X_ctx_t[ctx_chunk]  # (B,d_ctx) float32
            Bsz = int(Xb.shape[0])

            with torch.no_grad():
                z_ctx = self.router.B_projection(Xb)  # (B,d) float32
                if getattr(self.router, "combine_mode", "mul") == "mul":
                    z = z_ctx[:, None, None, :] * a_S_all[None, :, :, :]
                else:
                    z = z_ctx[:, None, None, :] + a_S_all[None, :, :, :]
                z_flat = z.reshape(Bsz * C, K, d)  # (B*C,K,d) float32
                vals = self.router.sample_optimistic_reward_from_zS(z_flat, noise_vectors).reshape(Bsz, C)

            best_vals_job, best_c_job = vals.max(dim=1)
            chunk_best_val, local_best = best_vals_job.max(dim=0)

            v = float(chunk_best_val.item())
            if v > best_val:
                best_val = v
                best_qpos = int(pos_chunk[local_best].item())
                best_ctx = int(ctx_chunk[local_best].item())
                best_c = int(best_c_job[local_best].item())

        uid_r, ctx_idx_r = self.queue_router[best_qpos]
        return best_qpos, best_ctx, list(self.all_combis[best_c])

    def step(self):
        self.steps += 1
        t = self.steps

        # Synchronize u_dep so router and oracle sample with the same random value
        u_dep = float(self.rng_feedback.rand())

        # Take a snapshot of the router queue before decision (Reference for Queue-max oracle/regret)
        X_t_snapshot_ctx = [ctx for (_, ctx) in self.queue_router]

        R_alg_t = 0.0
        R_star_t = 0.0
        loss_mnl_curr = 0.0

        # Reset debug cache
        self._last_S_router = None
        self._last_ctx_router = None
        self._last_uid_router = None
        self._last_choice_router = None
        self._last_router_metrics = None

        self._last_S_oracle = None
        self._last_ctx_oracle = None
        self._last_uid_oracle = None
        self._last_choice_oracle = None
        self._last_oracle_metrics = None  # Stores queue-max oracle (best X, S) metrics

        # -----------------------------
        # Router decision (TS or explore)
        # -----------------------------
        if len(self.queue_router) > 0:
            self.cnt_decision += 1

            do_explore = self.explore_enabled and (self.A_prev == 1) and (self.E_prev == 1)

            uid_r = None
            ctx_idx_r = None
            x_ctx = None
            S_t = None
            explore_used = False
            alpha_t = None

            # (1) Forced Exploration: Find the last arrival job and cycle through combinations
            if do_explore and (self.last_arrival_uid is not None):
                pos = next((i for i, (u, _) in enumerate(self.queue_router) if u == self.last_arrival_uid), None)
                if pos is not None:
                    uid_r, ctx_idx_r = self.queue_router.pop(pos)
                    x_ctx = self.X_ctx[ctx_idx_r]
                    S_t = list(self.all_combis[self.comb_idx])
                    self.comb_idx = (self.comb_idx + 1) % self.C_size
                    explore_used = True
                    self.cnt_explore += 1

            # (2) Exploitation: Evaluate all (job, combination) pairs using TS noise and select best
            if x_ctx is None:
                t_eff = self.config.max_steps
                term1 = self.d * math.log(1.0 + (t_eff * self.K) / (self.d * self.lambda_0))
                term2 = 4.0 * math.log(t_eff)
                term3 = self.kappa * math.sqrt(self.lambda_0)
                alpha_t = (0.5 * self.kappa * math.sqrt(term1 + term2) + term3) * self.config.alpha_coef

                noise_vectors = self.router.sample_theta_noise(alpha_t=alpha_t, M=self.M_sample)
                best_qpos, best_ctx, best_S = self._select_best_by_exhaustive_batch(noise_vectors)

                uid_r, ctx_idx_r = self.queue_router.pop(best_qpos)
                x_ctx = self.X_ctx[ctx_idx_r]
                S_t = best_S

            # Save router selection for debug
            self._last_S_router = list(S_t)
            self._last_ctx_router = int(ctx_idx_r)
            self._last_uid_router = int(uid_r)

            # Retrieve ground-truth data
            odds_row_r = self.odds_mat[ctx_idx_r]
            r_row_r = self.r_mat[ctx_idx_r]

            # True departure rate for the chosen (X, S)
            dep_alg_true = self._dep_rate_from_odds_row(odds_row_r, S_t)
            R_alg_t = float(dep_alg_true)

            # Sample ground-truth MNL choice
            departed, chosen_model, j_local = self.sample_mnl_choice_u(ctx_idx_r, S_t, u=u_dep)
            self._last_choice_router = (bool(departed), int(chosen_model) if chosen_model is not None else None)

            # -----------------------------
            # Departure logs (Router): queue 비면 기록 안 함 -> 여기서만 append
            # -----------------------------
            self.dep_prob_router_hist.append(float(dep_alg_true))
            self.dep_event_router_hist.append(1.0 if departed else 0.0)
            self.dep_router_step_idx.append(int(t))

            # Update router with label (one-hot, outside=0)
            K_count = len(S_t)
            y_vec = torch.zeros(K_count + 1, device=self.device, dtype=torch.float32)
            if j_local is None:
                y_vec[0] = 1.0
            else:
                y_vec[int(j_local) + 1] = 1.0

            loss_mnl_curr, _ = self.router.update_from_ctx(x_ctx, S_t, y_vec)

            # Re-insert into queue if the job did not depart (outside choice)
            if not departed:
                self.queue_router.append((uid_r, ctx_idx_r))

            # -----------------------------
            # Router metrics: pred/true output for selected X, S
            # -----------------------------
            true_r_alg = self._true_r_list(r_row_r, S_t)
            pred_u_alg = self._pred_u_list_from_theta(x_ctx, S_t)
            pred_r_alg = _sigmoid_np(np.asarray(pred_u_alg, dtype=np.float32)).tolist()

            topk_pred = self._topk_models_by_pred_r(x_ctx, k=self.debug_topk)
            r_hat_all = self._pred_r_all_models(x_ctx)
            topk_pred_names = [self._name(i) for i in topk_pred]
            topk_pred_vals = [float(r_hat_all[i]) for i in topk_pred]

            topk_true = self._topk_models_by_true_r(r_row_r, k=self.debug_topk)
            topk_true_names = [self._name(i) for i in topk_true]
            topk_true_vals = [float(r_row_r[i]) for i in topk_true]

            self._last_router_metrics = {
                "explore_used": explore_used,
                "alpha_t": float(alpha_t) if alpha_t is not None else None,
                "dep_alg_true": float(dep_alg_true),
                "true_r_alg": [float(v) for v in true_r_alg],
                "pred_r_alg": [float(v) for v in pred_r_alg],
                "pred_u_alg": [float(v) for v in pred_u_alg],
                "topk_pred": topk_pred,
                "topk_pred_names": topk_pred_names,
                "topk_pred_vals": topk_pred_vals,
                "topk_true": topk_true,
                "topk_true_names": topk_true_names,
                "topk_true_vals": topk_true_vals,
            }

        # -----------------------------
        # Oracle system queue progression (for Q-gap calculation)
        #   - queue 비면 기록/샘플링 자체가 없다고 본다
        # -----------------------------
        if len(self.queue_oracle) > 0:
            q_ctx_arr = np.asarray([ctx for (_, ctx) in self.queue_oracle], dtype=np.int64)
            best_pos = int(np.argmax(self.max_departure_rates[q_ctx_arr]))

            uid_o, ctx_idx_o = self.queue_oracle.pop(best_pos)
            S_o = list(self.all_combis[int(self.best_S_idx[ctx_idx_o])])

            # 기대 dep(oracle queue-progression)
            dep_oracle_true = float(self._dep_rate_from_odds_row(self.odds_mat[ctx_idx_o], S_o))

            departed_o, chosen_o, _ = self.sample_mnl_choice_u(ctx_idx_o, S_o, u=u_dep)
            self._last_choice_oracle = (bool(departed_o), int(chosen_o) if chosen_o is not None else None)

            # -----------------------------
            # Departure logs (Oracle progression): queue 비면 기록 안 함 -> 여기서만 append
            # -----------------------------
            self.dep_prob_oracle_hist.append(dep_oracle_true)
            self.dep_event_oracle_hist.append(1.0 if departed_o else 0.0)
            self.dep_oracle_step_idx.append(int(t))

            if not departed_o:
                self.queue_oracle.append((uid_o, ctx_idx_o))

        # -----------------------------
        # Regret update (queue-max oracle): Best X + Best S
        #   - snapshot이 비면(=router queue 비면) 기록 안 한다
        # -----------------------------
        ctx_star = None
        S_star = None
        dep_star = None
        true_r_star = None

        if len(X_t_snapshot_ctx) > 0:
            ctx_arr = np.asarray(X_t_snapshot_ctx, dtype=np.int64)
            ctx_star = int(ctx_arr[np.argmax(self.max_departure_rates[ctx_arr])])
            R_star_t = float(self.max_departure_rates[ctx_star])

            c_star = int(self.best_S_idx[ctx_star])
            S_star = list(self.all_combis[c_star])
            dep_star = float(self._dep_rate_from_odds_row(self.odds_mat[ctx_star], S_star))
            true_r_star = self._true_r_list(self.r_mat[ctx_star], S_star)

            # -----------------------------
            # Departure logs (Oracle@queue-max): snapshot 비면 기록 안 함 -> 여기서만 append
            # -----------------------------
            self.dep_prob_star_hist.append(float(dep_star))
            self.dep_star_step_idx.append(int(t))

            # Save oracle metrics for debug
            topk_star = self._topk_models_by_true_r(self.r_mat[ctx_star], k=self.debug_topk)
            self._last_oracle_metrics = {
                "ctx_star": int(ctx_star),
                "S_star": list(S_star),
                "dep_star": float(dep_star),
                "true_r_star": [float(v) for v in (true_r_star or [])],
                "topk_true": topk_star,
                "topk_true_names": [self._name(i) for i in topk_star],
                "topk_true_vals": [float(self.r_mat[ctx_star][i]) for i in topk_star],
            }

        self.cum_regret += (float(R_star_t) - float(R_alg_t))
        self.regret_history.append(self.cum_regret)

        # -----------------------------
        # Job Arrival
        # -----------------------------
        A_curr = 0
        if self.rng_arrival.rand() < self.config.arrival_rate:
            ctx_idx_new = int(self.rng_arrival.choice(self.job_pool))
            uid_new = self.next_uid
            self.next_uid += 1
            self.queue_router.append((uid_new, ctx_idx_new))
            self.queue_oracle.append((uid_new, ctx_idx_new))
            self.last_arrival_uid = uid_new
            A_curr = 1

        Q_r = len(self.queue_router)
        Q_o = len(self.queue_oracle)
        self.Q_regret_history.append((Q_r - Q_o))
        self.Q_router_history.append(Q_r)
        self.Q_oracle_history.append(Q_o)

        # -----------------------------
        # Explore coin (for next step)
        # -----------------------------
        if self.explore_enabled:
            eta_t = min(1.0, self.c1 * ((t + 1.0) ** (-0.5)))
            E_curr = 1 if (self.rng_explore.rand() < eta_t) else 0
        else:
            E_curr = 0

        self.A_prev = A_curr
        self.E_prev = E_curr

        # -----------------------------
        # Logging
        # -----------------------------
        if (self.steps % self.config.log_every == 0):
            avg_reg = self.cum_regret / self.steps
            q_diff = self.Q_regret_history[-1]
            T_hist = max(getattr(self.router, "_T", 1), 1)
            print("\n  ==================== DEBUG START ====================")
            msg = (
                f"[Queue-Env step={self.steps}] "
                f"regret(avg)={avg_reg:.6f}  "
                f"Q-gap={q_diff:.3f}  "
                f"Q_r={Q_r} Q_o={Q_o}  "
                f"L_mnl(avg)={(loss_mnl_curr / T_hist):.4f}"
            )
            print(msg)

        # -----------------------------
        # Debug Output
        # -----------------------------
        if self.debug_verbose and (self.steps % self.debug_print_every == 0):

            def _print_pred_topk(metrics: dict):
                print(f"  [Router] top{self.debug_topk} models by PRED r:")
                for mi, mn, mr in zip(
                    metrics.get("topk_pred", []),
                    metrics.get("topk_pred_names", []),
                    metrics.get("topk_pred_vals", []),
                ):
                    print(f"    - {int(mi):2d} {mn}  pred_r={float(mr):.6f}")

            def _print_true_topk(ctx_idx: int, topk_idx, topk_names, topk_vals, header: str):
                print(f"  [{header}] top{self.debug_topk} models by TRUE r:")
                for mi, mn, tr in zip(topk_idx, topk_names, topk_vals):
                    mi = int(mi)
                    acc = float(self.acc_mat[ctx_idx, mi])
                    util = float(self.util_mat[ctx_idx, mi])
                    u01 = (util - self.u_min) / self.u_den
                    r = float(tr)
                    odds = float(self.odds_mat[ctx_idx, mi])
                    cost_implied = (acc - util) / self.lambda_0 if self.lambda_0 != 0 else float("nan")
                    print(f"    - {mi:2d} {mn} acc={acc:.6f} util={util:.6f} u01={u01:.6f}")
                    print(f"      r={r:.6f}  odds={odds:.6f}  cost_imp={cost_implied:.8f}")

            # Router Block
            if self._last_S_router is not None and self._last_ctx_router is not None:
                ctx = int(self._last_ctx_router)
                uid = int(self._last_uid_router) if self._last_uid_router is not None else -1
                S_alg = list(self._last_S_router)
                inside_r, chosen_r = self._last_choice_router or (False, None)
                m = self._last_router_metrics or {}

                print(f"  [Router] uid={uid} {self._ctx_info(ctx)} explore={m.get('explore_used')} alpha={m.get('alpha_t')}")
                _print_pred_topk(m)
                _print_true_topk(ctx, m.get("topk_true", []), m.get("topk_true_names", []), m.get("topk_true_vals", []), "Router")

                print(f"  [Router] S_t idx={S_alg} pred_r={[f'{u:.4f}' for u in m.get('pred_r_alg', [])]}")
                print(f"          true_r={[f'{u:.4f}' for u in m.get('true_r_alg', [])]} dep_true={m.get('dep_alg_true', 0.0):.6f}")
                status = f"chosen={chosen_r} ({self._name(chosen_r)})" if inside_r else "outside"
                print(f"  [Sample Router] inside={inside_r} {status}")
            else:
                print("  [Router] N/A")

            # Oracle Block
            om = self._last_oracle_metrics or {}
            if om.get("ctx_star") is not None:
                ctx_star = int(om["ctx_star"])
                S_star = list(om["S_star"])
                print(f"  [Oracle@queue-max] {self._ctx_info(ctx_star)}")
                _print_true_topk(ctx_star, om.get("topk_true", []), om.get("topk_true_names", []), om.get("topk_true_vals", []), "Oracle@queue-max")
                print(f"  [Oracle@queue-max] S* idx={S_star} true_r={[f'{u:.4f}' for u in om.get('true_r_star', [])]} dep_true={om.get('dep_star', 0.0):.6f}")

                dep_alg = float((self._last_router_metrics or {}).get("dep_alg_true", 0.0))
                print(f"  [Gap] dep_true(S*) - dep_true(S_t) = {(float(om.get('dep_star', 0.0)) - dep_alg):.6f}")

            print(f"  [Regret terms] R_star_t={R_star_t:.6f} R_alg_t={R_alg_t:.6f}")
            print("  ===================== DEBUG END =====================")

        return False

    def run(self):
        while self.steps < self.config.max_steps:
            _ = self.step()

        Q_regret_T = self.Q_regret_history[-1] if self.Q_regret_history else 0.0
        avg_regret = self.cum_regret / max(1, self.steps)

        explore_rate = 0.0
        if self.cnt_decision > 0:
            explore_rate = self.cnt_explore / self.cnt_decision

        def _mean(x: List[float]) -> float:
            return float(np.mean(np.asarray(x, dtype=float))) if len(x) > 0 else float("nan")

        router_dep_mean = _mean(self.dep_prob_router_hist)
        router_evt_mean = _mean(self.dep_event_router_hist)

        oracle_dep_mean = _mean(self.dep_prob_oracle_hist)
        oracle_evt_mean = _mean(self.dep_event_oracle_hist)

        star_dep_mean = _mean(self.dep_prob_star_hist)

        print(
            f"--- Finished. AvgRegret={avg_regret:.6f}, Final Q_gap={Q_regret_T:.3f}, ExploreRate={explore_rate:.4f} ---"
        )
        print(
            "[Departure(decision-only)] "
            f"Router E[dep]={router_dep_mean:.6f}  dep_evt={router_evt_mean:.6f}  "
            f"Oracle(queue) E[dep]={oracle_dep_mean:.6f}  dep_evt={oracle_evt_mean:.6f}  "
            f"Oracle@queue-max E[dep*]={star_dep_mean:.6f}"
        )

        # 반환 시그니처 깨지지 않게 router 객체에 로그를 붙인다
        try:
            self.router._queue_env_logs = {
                "dep_prob_router_hist": self.dep_prob_router_hist,
                "dep_event_router_hist": self.dep_event_router_hist,
                "dep_router_step_idx": self.dep_router_step_idx,
                "dep_prob_oracle_hist": self.dep_prob_oracle_hist,
                "dep_event_oracle_hist": self.dep_event_oracle_hist,
                "dep_oracle_step_idx": self.dep_oracle_step_idx,
                "dep_prob_star_hist": self.dep_prob_star_hist,
                "dep_star_step_idx": self.dep_star_step_idx,
            }
        except Exception:
            pass

        return (
            self.router,
            avg_regret,
            Q_regret_T,
            self.regret_history,
            self.Q_regret_history,
            self.Q_router_history,
            self.Q_oracle_history,
            float(explore_rate),
        )


def queue_env(X_ctx, acc_mat, util_mat, config, router, model_names=None, row_ids=None, sample_ids=None):
    env = QueueEnv(
        X_ctx,
        acc_mat,
        util_mat,
        config,
        router,
        model_names=model_names,
        row_ids=row_ids,
        sample_ids=sample_ids,
    )
    return env.run()
