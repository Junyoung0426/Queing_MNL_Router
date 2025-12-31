# queue_env.py
from typing import List, Tuple, Optional
from itertools import combinations
import math

import numpy as np
import torch

from mnl_router import MNLRouter, build_X_S_from_context_idx as build_X_S_from_context
from queue_config import QueueConfig

class QueueEnv:

    def __init__(
        self,
        X_ctx: np.ndarray,
        acc_mat: np.ndarray,
        util_mat: np.ndarray,
        config: QueueConfig,
        router: MNLRouter,
        model_names: Optional[List[str]] = None,
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
        N2, n_models = acc_mat.shape
        N3, K2 = util_mat.shape
        assert N == N2 == N3
        assert n_models == K2
        if N == 0:
            pass

        self.N = N
        self.d_ctx = d_ctx
        self.n_models = n_models

        self.config.d_ctx = d_ctx
        self.config.n_models = n_models

        self.lambda_0 = float(self.config.lambda_0)

        self.router = router.to(self.device)
        self.router.train()

        self.model_names = model_names
        if self.model_names is not None:
            if len(self.model_names) != self.n_models:
                pass
            self.idx2model = list(self.model_names)
        else:
            self.idx2model = None

        assert 1 <= config.assort_K <= self.n_models
        self.K = int(config.assort_K)
        self.all_combis = list(combinations(range(self.n_models), self.K))
        self.C_size = len(self.all_combis)

        self.rng_arrival = np.random.RandomState(config.seed)
        self.rng_explore = np.random.RandomState(config.seed + 1)
        self.rng_router = np.random.RandomState(config.seed + 2)
        self.rng_oracle = np.random.RandomState(config.seed + 3)

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

        self.kappa = float(self.config.kappa)
        self.c1 = float(self.config.c1)
        self.d = int(getattr(self.router, "d", self.config.d_proj))

        denom_M = math.log(1.0 - 1.0 / (4.0 * math.sqrt(math.e * math.pi)))
        self.M_sample = max(1, int(math.ceil(1.0 - math.log(self.K) / denom_M)))

        self.exploit_job_batch = 128

        print(
            f"[Queue-Env] N={self.N}, n_models={self.n_models}, |C|={self.C_size}, "
            f"ArrRate={self.config.arrival_rate}, MaxSteps={self.config.max_steps}, "
            f"d={self.d}, M={self.M_sample}, B_type={self.config.b_type}"
        )

        self.X_ctx_t = torch.from_numpy(self.X_ctx).float().to(self.device)
        self.S_tensor = torch.tensor(self.all_combis, device=self.device, dtype=torch.long)

        print("[Queue-Env] Pre-computing Oracle Rates + best S...")
        self.max_departure_rates = np.zeros(self.N, dtype=np.float64)
        self.best_S_idx = np.zeros(self.N, dtype=np.int64)

        combi_idx_np = np.asarray(self.all_combis, dtype=np.int64)
        for i in range(self.N):
            util_row = self.util_mat[i].astype(np.float64, copy=False)
            util_S = util_row[combi_idx_np]

            max_u = util_S.max(axis=1)
            max_u0 = np.maximum(max_u, 0.0)

            exps = np.exp(util_S - max_u0[:, None])
            sumexp = exps.sum(axis=1)
            denom = np.exp(-max_u0) + sumexp
            rates = sumexp / np.maximum(denom, 1e-300)

            best_c = int(np.argmax(rates))
            self.best_S_idx[i] = best_c
            self.max_departure_rates[i] = float(rates[best_c])

        print("[Queue-Env] Ready.")

        self.debug_verbose: bool = True
        self.debug_topk: int = 3
        self.debug_print_every: int = self.config.log_every

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

    def _name(self, i: int) -> str:
        if self.idx2model is None:
            return str(i)
        return self.idx2model[int(i)]

    def _dep_rate_from_util_row(self, util_row: np.ndarray, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        u = util_row[np.array(S, dtype=np.int64)]
        max_u = float(u.max())
        max_u0 = max(0.0, max_u)
        exps = np.exp(u - max_u0)
        denom = math.exp(-max_u0) + float(exps.sum())
        denom = max(denom, 1e-300)
        return float(exps.sum() / denom)

    def true_departure_rate(self, ctx_idx: int, S: List[int]) -> float:
        util_row = self.util_mat[ctx_idx].astype(np.float64, copy=False)
        return self._dep_rate_from_util_row(util_row, S)

    def _true_u_list(self, util_row: np.ndarray, S: List[int]) -> List[float]:
        return [float(util_row[int(i)]) for i in S]

    def _topk_models_by_true_u(self, util_row: np.ndarray, k: int = 3) -> List[int]:
        k = min(int(k), util_row.shape[0])
        return list(np.argsort(-util_row)[:k].astype(int))

    def _pred_u_list_from_theta(self, x_ctx: np.ndarray, S: List[int]) -> List[float]:
        if len(S) == 0:
            return []
        X_S = build_X_S_from_context(x_ctx, S, self.device)
        with torch.no_grad():
            u_hat = self.router.get_scores(X_S)
        return [float(v) for v in u_hat.detach().cpu().tolist()]

    def _pred_u_all_models(self, x_ctx: np.ndarray) -> np.ndarray:
        x_ctx_tensor = torch.from_numpy(x_ctx).float().to(self.device)
        x_rep = x_ctx_tensor.unsqueeze(0).expand(self.n_models, -1)
        idx = torch.arange(self.n_models, device=self.device, dtype=torch.float32).unsqueeze(1)
        X_all = torch.cat([x_rep, idx], dim=1)
        with torch.no_grad():
            u_hat = self.router.get_scores(X_all)
        return u_hat.detach().cpu().numpy().astype(np.float64)

    def _topk_models_by_pred_u(self, x_ctx: np.ndarray, k: int = 3) -> List[int]:
        u_hat_all = self._pred_u_all_models(x_ctx)
        k = min(int(k), u_hat_all.shape[0])
        return list(np.argsort(-u_hat_all)[:k].astype(int))

    def sample_mnl_choice(self, ctx_idx: int, S: List[int], rng: np.random.RandomState):
        util_row = self.util_mat[ctx_idx].astype(np.float64, copy=False)
        util_S = np.array([util_row[k] for k in S], dtype=np.float64)
        if util_S.size == 0:
            return False, None, None

        max_u = float(util_S.max())
        max_u0 = max(0.0, max_u)

        exps = np.exp(util_S - max_u0)
        exp_out = math.exp(-max_u0)
        denom = exp_out + float(exps.sum())
        denom = max(denom, 1e-300)

        p_out = exp_out / denom
        p_in = exps / denom

        probs = np.empty(len(S) + 1, dtype=np.float64)
        probs[0] = p_out
        probs[1:] = p_in
        probs /= probs.sum()

        choice = rng.choice(len(probs), p=probs)
        if choice == 0:
            return False, None, None
        j_local = choice - 1
        return True, S[j_local], j_local

    def _select_best_by_exhaustive_batch(self, noise_vectors: torch.Tensor) -> Tuple[int, int, List[int]]:
        q = self.queue_router
        if len(q) == 0:
            pass

        q_ctx = torch.tensor([ctx for (_, ctx) in q], device=self.device, dtype=torch.long)
        q_pos = torch.arange(len(q), device=self.device, dtype=torch.long)

        C = self.C_size
        K = self.K
        d_in = self.d_ctx + 1

        best_val: Optional[float] = None
        best_qpos: Optional[int] = None
        best_ctx: Optional[int] = None
        best_c: Optional[int] = None

        bs = self.exploit_job_batch
        for st in range(0, q_pos.numel(), bs):
            ed = min(st + bs, q_pos.numel())
            pos_chunk = q_pos[st:ed]
            ctx_chunk = q_ctx[st:ed]

            Xb = self.X_ctx_t[ctx_chunk]
            B = Xb.shape[0]

            x_rep = Xb[:, None, None, :].expand(B, C, K, self.d_ctx)
            idx_rep = self.S_tensor[None, :, :].expand(B, C, K).to(torch.float32).unsqueeze(-1)
            X = torch.cat([x_rep, idx_rep], dim=-1).reshape(B * C, K, d_in)

            vals = self.router.sample_optimistic_reward_batch(X, noise_vectors).reshape(B, C)

            best_vals_job, best_c_job = vals.max(dim=1)
            chunk_best_val, local_best = best_vals_job.max(dim=0)

            chunk_best_val_f = float(chunk_best_val.item())
            if best_val is None or chunk_best_val_f > best_val:
                best_val = chunk_best_val_f
                best_qpos = int(pos_chunk[local_best].item())
                best_ctx = int(ctx_chunk[local_best].item())
                best_c = int(best_c_job[local_best].item())

        assert best_qpos is not None and best_ctx is not None and best_c is not None
        return best_qpos, best_ctx, list(self.all_combis[best_c])

    def step(self):
        self.steps += 1
        t = self.steps

        X_t_snapshot_ctx = [ctx for (_, ctx) in self.queue_router]

        R_alg_t = 0.0
        R_star_t = 0.0
        loss_mnl_curr = 0.0

        self._last_S_router = None
        self._last_ctx_router = None
        self._last_uid_router = None
        self._last_S_star_ctx = None
        self._last_choice_router = None
        self._last_router_metrics = None

        self._last_S_oracle = None
        self._last_ctx_oracle = None
        self._last_uid_oracle = None
        self._last_choice_oracle = None
        self._last_oracle_metrics = None

        if len(self.queue_router) > 0:
            self.cnt_decision += 1
            do_explore = (self.A_prev == 1) and (self.E_prev == 1)

            uid_r = None
            ctx_idx_r = None
            x_ctx = None
            S_t = None
            explore_used = False
            alpha_t = None

            if do_explore and (self.last_arrival_uid is not None):
                pos = next((i for i, (u, _) in enumerate(self.queue_router) if u == self.last_arrival_uid), None)
                if pos is not None:
                    uid_r, ctx_idx_r = self.queue_router.pop(pos)
                    x_ctx = self.X_ctx[ctx_idx_r]
                    S_t = list(self.all_combis[self.comb_idx])
                    self.comb_idx = (self.comb_idx + 1) % self.C_size
                    explore_used = True
                    self.cnt_explore += 1

            if x_ctx is None:
                t_eff =  max(t - 1, 1)
                term1 = self.d * math.log(1.0 + (t_eff * self.K) / (self.d * self.lambda_0))
                term2 = 4.0 * math.log(t_eff)
                term3 = self.kappa * math.sqrt(self.lambda_0)
                alpha_t = ((0.5 * self.kappa * math.sqrt(term1 + term2)+ term3))

                noise_vectors = self.router.sample_theta_noise(alpha_t=alpha_t, M=self.M_sample)
                best_qpos, best_ctx, best_S = self._select_best_by_exhaustive_batch(noise_vectors)

                uid_r, ctx_idx_r = self.queue_router.pop(best_qpos)
                x_ctx = self.X_ctx[ctx_idx_r]
                S_t = best_S

            assert uid_r is not None and ctx_idx_r is not None and x_ctx is not None and S_t is not None

            self._last_S_router = list(S_t)
            self._last_ctx_router = int(ctx_idx_r)
            self._last_uid_router = int(uid_r)

            util_row_r = self.util_mat[ctx_idx_r].astype(np.float64, copy=False)

            dep_alg_true = self._dep_rate_from_util_row(util_row_r, S_t)
            R_alg_t = dep_alg_true

            departed, chosen_model, j_local = self.sample_mnl_choice(ctx_idx_r, S_t, rng=self.rng_router)
            self._last_choice_router = (bool(departed), int(chosen_model) if chosen_model is not None else None)

            y_vec = torch.zeros(len(S_t), device=self.device)
            if j_local is not None:
                y_vec[j_local] = 1.0
            X_S = build_X_S_from_context(x_ctx, S_t, self.device)
            loss_mnl_curr, _ = self.router.update(X_S, y_vec)

            if not departed:
                self.queue_router.append((uid_r, ctx_idx_r))

            c_star_ctx = int(self.best_S_idx[ctx_idx_r])
            S_star_ctx = list(self.all_combis[c_star_ctx])
            self._last_S_star_ctx = list(S_star_ctx)
            dep_star_true = self._dep_rate_from_util_row(util_row_r, S_star_ctx)

            true_u_alg = self._true_u_list(util_row_r, S_t)
            true_u_star = self._true_u_list(util_row_r, S_star_ctx)

            pred_u_alg = self._pred_u_list_from_theta(x_ctx, S_t)
            pred_u_star = self._pred_u_list_from_theta(x_ctx, S_star_ctx)

            topk_pred = self._topk_models_by_pred_u(x_ctx, k=self.debug_topk)
            u_hat_all = self._pred_u_all_models(x_ctx)
            topk_pred_names = [self._name(i) for i in topk_pred]
            topk_pred_vals = [float(u_hat_all[i]) for i in topk_pred]

            topk_true = self._topk_models_by_true_u(util_row_r, k=self.debug_topk)
            topk_true_names = [self._name(i) for i in topk_true]
            topk_true_vals = [float(util_row_r[i]) for i in topk_true]

            self._last_router_metrics = {
                "explore_used": explore_used,
                "alpha_t": float(alpha_t) if alpha_t is not None else None,
                "dep_alg_true": float(dep_alg_true),
                "dep_star_true": float(dep_star_true),
                "true_u_alg": true_u_alg,
                "true_u_star": true_u_star,
                "pred_u_alg": pred_u_alg,
                "pred_u_star": pred_u_star,
                "topk_pred": topk_pred,
                "topk_pred_names": topk_pred_names,
                "topk_pred_vals": topk_pred_vals,
                "topk_true": topk_true,
                "topk_true_names": topk_true_names,
                "topk_true_vals": topk_true_vals,
            }

        if len(self.queue_oracle) > 0:
            q_ctx = np.asarray([ctx for (_, ctx) in self.queue_oracle], dtype=np.int64)
            best_pos = int(np.argmax(self.max_departure_rates[q_ctx]))

            uid_o, ctx_idx_o = self.queue_oracle.pop(best_pos)
            S_o = list(self.all_combis[int(self.best_S_idx[ctx_idx_o])])

            departed_o, chosen_o, _ = self.sample_mnl_choice(ctx_idx_o, S_o, rng=self.rng_oracle)
            self._last_choice_oracle = (bool(departed_o), int(chosen_o) if chosen_o is not None else None)

            if not departed_o:
                self.queue_oracle.append((uid_o, ctx_idx_o))

            self._last_S_oracle = list(S_o)
            self._last_ctx_oracle = int(ctx_idx_o)
            self._last_uid_oracle = int(uid_o)

            util_row_o = self.util_mat[ctx_idx_o].astype(np.float64, copy=False)
            dep_o_true = self._dep_rate_from_util_row(util_row_o, S_o)
            true_u_o = self._true_u_list(util_row_o, S_o)

            topk_o = self._topk_models_by_true_u(util_row_o, k=self.debug_topk)
            topk_o_names = [self._name(i) for i in topk_o]
            topk_o_vals = [float(util_row_o[i]) for i in topk_o]

            self._last_oracle_metrics = {
                "dep_o_true": float(dep_o_true),
                "true_u_o": true_u_o,
                "topk_o": topk_o,
                "topk_o_names": topk_o_names,
                "topk_o_vals": topk_o_vals,
            }

        if len(X_t_snapshot_ctx) > 0:
            R_star_t = max(self.max_departure_rates[ctx] for ctx in X_t_snapshot_ctx)

        self.cum_regret += (R_star_t - R_alg_t)
        self.regret_history.append(self.cum_regret)

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

        eta_t = min(1.0, self.c1 * (t + 1.0) ** (-0.5))
        E_curr = 1 if (self.rng_explore.rand() < eta_t) else 0
        self.A_prev = A_curr
        self.E_prev = E_curr

        if (self.steps % self.config.log_every == 0):
            avg_reg = self.cum_regret / self.steps
            q_diff = self.Q_regret_history[-1]
            T_hist = max(getattr(self.router, "_T", 1), 1)

            msg = (
                f"[Queue-Env step={self.steps}] "
                f"regret(avg)={avg_reg:.6f}  "
                f"Q-gap={q_diff:.3f}  "
                f"Q_r={Q_r} Q_o={Q_o}  "
                f"L_mnl(sum)={loss_mnl_curr:.2f}  "
                f"L_mnl(avg)={(loss_mnl_curr / T_hist):.4f}"
            )
            print(msg)

        if self.debug_verbose and (self.steps % self.debug_print_every == 0):
            print("")

            print("  ==================== DEBUG START ====================")

            if self._last_S_router is not None and self._last_ctx_router is not None:
                ctx = int(self._last_ctx_router)
                uid = int(self._last_uid_router) if self._last_uid_router is not None else -1
                S_alg = list(self._last_S_router)
                S_star = list(self._last_S_star_ctx) if self._last_S_star_ctx is not None else None
                inside_r, chosen_r = self._last_choice_router if self._last_choice_router is not None else (False, None)

                m = self._last_router_metrics or {}
                explore_used = m.get("explore_used", None)
                alpha_t = m.get("alpha_t", None)

                print(f"  [Router] uid={uid} ctx_idx={ctx}  explore_used={explore_used}  alpha_t={alpha_t}")

                print(f"  [Router] top{self.debug_topk} models by PRED util (z·theta):")
                for mi, mn, mu in zip(m.get("topk_pred", []), m.get("topk_pred_names", []), m.get("topk_pred_vals", [])):
                    print(f"    - {int(mi):2d} {mn}  pred_util={float(mu):.6f}")

                print(f"  [Router] top{self.debug_topk} models by TRUE util (util_mat):")
                for mi, mn, mu in zip(m.get("topk_true", []), m.get("topk_true_names", []), m.get("topk_true_vals", [])):
                    print(f"    - {int(mi):2d} {mn}  util={float(mu):.6f}")

                S_alg_names = [self._name(i) for i in S_alg]
                print(f"  [Router] S_t idx={S_alg} name={S_alg_names}")
                print(f"          pred_util(S_t)={[f'{u:.4f}' for u in m.get('pred_u_alg', [])]}")
                print(f"          true_util(S_t)={[f'{u:.4f}' for u in m.get('true_u_alg', [])]}  dep_true(S_t)={m.get('dep_alg_true', 0.0):.6f}")

                if S_star is not None:
                    S_star_names = [self._name(i) for i in S_star]
                    print(f"  [Oracle@same-ctx] S*_ctx idx={S_star} name={S_star_names}")
                    print(f"                pred_util(S*_ctx)={[f'{u:.4f}' for u in m.get('pred_u_star', [])]}")
                    print(f"                true_util(S*_ctx)={[f'{u:.4f}' for u in m.get('true_u_star', [])]}  dep_true(S*_ctx)={m.get('dep_star_true', 0.0):.6f}")

                    dep_alg = m.get("dep_alg_true", None)
                    dep_star = m.get("dep_star_true", None)
                    if dep_alg is not None and dep_star is not None:
                        print(f"  [Gap] dep_true(S*_ctx) - dep_true(S_t) = {(dep_star - dep_alg):.6f}")

                if inside_r:
                    print(f"  [Sample Router] inside=True chosen={chosen_r} ({self._name(chosen_r) if chosen_r is not None else 'None'})")
                else:
                    print("  [Sample Router] inside=False (outside)")

                print(f"  [Regret terms] R_star_t(queue max)={R_star_t:.6f}  R_alg_t={R_alg_t:.6f}")

            else:
                print("  [Router] N/A (queue_router empty at this step)")

            if self._last_S_oracle is not None and self._last_ctx_oracle is not None:
                ctx_o = int(self._last_ctx_oracle)
                uid_o = int(self._last_uid_oracle) if self._last_uid_oracle is not None else -1
                S_o = list(self._last_S_oracle)
                S_o_names = [self._name(i) for i in S_o]
                inside_o, chosen_o = self._last_choice_oracle if self._last_choice_oracle is not None else (False, None)

                mo = self._last_oracle_metrics or {}
                print(f"  [Oracle system] uid={uid_o} ctx_idx={ctx_o}")
                print(f"  [Oracle system] top{self.debug_topk} models by TRUE util (util_mat):")
                for mi, mn, mu in zip(mo.get("topk_o", []), mo.get("topk_o_names", []), mo.get("topk_o_vals", [])):
                    print(f"    - {int(mi):2d} {mn}  util={float(mu):.6f}")

                print(f"  [Oracle system] S_o idx={S_o} name={S_o_names}")
                print(f"                true_util(S_o)={[f'{u:.4f}' for u in mo.get('true_u_o', [])]}  dep_true(S_o)={mo.get('dep_o_true', 0.0):.6f}")

                if inside_o:
                    print(f"  [Sample Oracle] inside=True chosen={chosen_o} ({self._name(chosen_o) if chosen_o is not None else 'None'})")
                else:
                    print("  [Sample Oracle] inside=False (outside)")

            else:
                print("  [Oracle system] N/A (queue_oracle empty at this step)")

            print("  ===================== DEBUG END =====================")

        return False

    def run(self) -> Tuple[MNLRouter, float, float, List[float], List[float]]:
        while self.steps < self.config.max_steps:
            _ = self.step()

        Q_regret_T = self.Q_regret_history[-1] if self.Q_regret_history else 0.0
        avg_regret = self.cum_regret / max(1, self.steps)

        explore_rate = 0.0
        if self.cnt_decision > 0:
            explore_rate = self.cnt_explore / self.cnt_decision

        print(f"--- Finished. AvgRegret={avg_regret:.6f}, Final Q_gap={Q_regret_T:.3f}, ExploreRate={explore_rate:.4f} ---")
        return self.router, avg_regret, Q_regret_T, self.regret_history, self.Q_regret_history

def queue_env(X_ctx, acc_mat, util_mat, config, router, model_names=None):
    env = QueueEnv(X_ctx, acc_mat, util_mat, config, router, model_names=model_names)
    return env.run()
