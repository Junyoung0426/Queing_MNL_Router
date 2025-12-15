# queue_env.py
from typing import List, Tuple, Optional
from itertools import combinations
import math

import numpy as np
import torch

from mnl_router import MNLRouter, build_X_S_from_context_idx as build_X_S_from_context


from queue_config import QueueConfig


class QueueEnv:
    """
    Unknown-horizon MNL queueing bandit 환경 (B 버전).

    - arrival_mode 삭제
    - 항상 job_pool에서 랜덤 샘플링 + 중복 허용 (with replacement)
    - 종료는 max_steps로만 한다
    """

    def __init__(
        self,
        X_ctx: np.ndarray,
        acc_mat: np.ndarray,
        util_mat: np.ndarray,
        config: QueueConfig,
        router: MNLRouter,
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
            raise ValueError("X_ctx is empty")

        self.N = N
        self.d_ctx = d_ctx
        self.n_models = n_models

        # config 채우기
        self.config.d_ctx = d_ctx
        self.config.n_models = n_models

        # lambda_0 결정
        self.lambda_0 = self.config.lambda0 if self.config.lambda0 is not None else self.config.reg_lambda

        # router (offline pretrain + freeze_B + reset_for_online 끝난 걸 받는다고 가정)
        self.router = router.to(self.device)
        self.router.train()

        # 조합 집합
        assert 1 <= config.assort_K <= self.n_models
        self.K = int(config.assort_K)
        self.all_combis = list(combinations(range(self.n_models), self.K))
        self.C_size = len(self.all_combis)

        self.rng = np.random.RandomState(config.seed)

        # 항상 i.i.d. 도착: job_pool에서 중복 허용 샘플
        self.job_pool = np.arange(self.N, dtype=np.int64)

        # 큐 / 상태
        self.queue_router: List[int] = []
        self.queue_oracle: List[int] = []
        self.last_arrival_idx: Optional[int] = None

        self.A_prev = 0
        self.E_prev = 0
        self.steps = 0
        self.comb_idx = 0

        # regret / 로그
        self.cum_regret = 0.0
        self.regret_history: List[float] = []
        self.Q_regret_history: List[float] = []

        self.log_l_mnl = 0.0
        self.log_cnt_updates = 0

        self.kappa = float(self.config.kappa)
        self.c0 = float(self.config.c0)
        self.d = int(getattr(self.router, "d", self.config.d_proj))

        denom_M = math.log(1.0 - 1.0 / (4.0 * math.sqrt(math.e * math.pi)))
        self.M_sample = max(1, int(math.ceil(1.0 - math.log(self.K) / denom_M)))

        self.exploit_job_batch = 128

        print(
            f"[Queue-Env] N={self.N}, n_models={self.n_models}, |C|={self.C_size}, "
            f"ArrRate={self.config.arrival_rate}, MaxSteps={self.config.max_steps}, "
            f"d={self.d}, M={self.M_sample}, B_type={self.config.b_type}"
        )

        # torch 캐시
        self.X_ctx_t = torch.from_numpy(self.X_ctx).float().to(self.device)
        self.S_tensor = torch.tensor(self.all_combis, device=self.device, dtype=torch.long)  # (C,K)
        self.S_flat_f = self.S_tensor.reshape(-1).float()                                    # (C*K,)

        # oracle precompute
        print("[Queue-Env] Pre-computing Oracle Rates + best S...")
        self.max_departure_rates = np.zeros(self.N, dtype=np.float64)
        self.best_S_idx = np.zeros(self.N, dtype=np.int64)

        combi_idx_np = np.asarray(self.all_combis, dtype=np.int64)
        for i in range(self.N):
            util_row = self.util_mat[i]
            util_S = util_row[combi_idx_np]  # (C,K)
            max_u = util_S.max(axis=1)
            exps = np.exp(util_S - max_u[:, None])
            sumexp = exps.sum(axis=1)
            denom = np.exp(-max_u) + sumexp
            rates = sumexp / denom
            best_c = int(np.argmax(rates))
            self.best_S_idx[i] = best_c
            self.max_departure_rates[i] = float(rates[best_c])

        print("[Queue-Env] Ready.")

    def true_departure_rate(self, job_idx: int, S: List[int]) -> float:
        util_row = self.util_mat[job_idx]
        util_S = np.array([util_row[k] for k in S], dtype=np.float64)
        if util_S.size == 0:
            return 0.0
        max_u = float(util_S.max())
        exps = np.exp(util_S - max_u)
        denom = math.exp(-max_u) + float(exps.sum())
        return float(exps.sum() / denom)

    def sample_mnl_choice(self, job_idx: int, S: List[int]):
        util_row = self.util_mat[job_idx]
        util_S = np.array([util_row[k] for k in S], dtype=np.float64)
        if util_S.size == 0:
            return False, None, None

        max_u = float(util_S.max())
        exps = np.exp(util_S - max_u)
        denom = math.exp(-max_u) + float(exps.sum())

        p_out = math.exp(-max_u) / denom
        p_in = exps / denom

        probs = np.empty(len(S) + 1, dtype=np.float64)
        probs[0] = p_out
        probs[1:] = p_in
        probs /= probs.sum()

        choice = self.rng.choice(len(probs), p=probs)
        if choice == 0:
            return False, None, None
        return True, S[choice - 1], choice - 1

    def _select_best_by_exhaustive_batch(self, noise_vectors: torch.Tensor) -> Tuple[int, List[int]]:
        if len(self.queue_router) == 0:
            raise RuntimeError("queue_router is empty")

        q_idxs_t = torch.as_tensor(self.queue_router, device=self.device, dtype=torch.long)

        best_val = None
        best_job = None
        best_c = None

        C = self.C_size
        K = self.K
        d_in = self.d_ctx + 1
        CK = C * K
        S_flat = self.S_flat_f  # (C*K,)

        bs = self.exploit_job_batch
        for st in range(0, q_idxs_t.numel(), bs):
            ed = min(st + bs, q_idxs_t.numel())
            idxs = q_idxs_t[st:ed]
            Xb = self.X_ctx_t[idxs]
            B = Xb.shape[0]

            x_rep = Xb[:, None, :].expand(B, CK, self.d_ctx)
            idx_rep = S_flat[None, :].expand(B, CK).unsqueeze(-1)
            X_flat = torch.cat([x_rep, idx_rep], dim=-1)              # (B, C*K, d_ctx+1)
            X_all = X_flat.reshape(B * C, K, d_in)                    # reshape가 안전하다

            vals = self.router.sample_optimistic_reward_batch(X_all, noise_vectors).reshape(B, C)
            best_vals_job, best_c_job = vals.max(dim=1)
            chunk_best_val, pos = best_vals_job.max(dim=0)

            if best_val is None or chunk_best_val.item() > best_val:
                best_val = float(chunk_best_val.item())
                best_job = int(idxs[pos].item())
                best_c = int(best_c_job[pos].item())

        assert best_job is not None and best_c is not None
        return best_job, list(self.all_combis[best_c])

    def step(self):
        self.steps += 1
        t = self.steps

        X_t_snapshot = list(self.queue_router)
        R_alg_t = 0.0

        # 1) router service
        if len(self.queue_router) > 0:
            do_explore = (self.A_prev == 1) and (self.E_prev == 1)

            if do_explore and (self.last_arrival_idx is not None) and (self.last_arrival_idx in self.queue_router):
                idx_r = self.last_arrival_idx
                self.queue_router.remove(idx_r)
                x_ctx = self.X_ctx[idx_r]
                S_t = list(self.all_combis[self.comb_idx])
                self.comb_idx = (self.comb_idx + 1) % self.C_size
            else:
                t_eff = max(t - 1, 1)
                term1 = self.d * math.log(1.0 + (t_eff * self.K) / (self.d * self.lambda_0))
                term2 = 4.0 * math.log(t_eff)
                term3 = self.kappa * math.sqrt(self.lambda_0)
                alpha_t = 0.5 * self.kappa * math.sqrt(term1 + term2 + term3)

                noise_vectors = self.router.sample_theta_noise(alpha_t, self.M_sample)
                best_idx, best_S = self._select_best_by_exhaustive_batch(noise_vectors)

                self.queue_router.remove(best_idx)
                idx_r = best_idx
                x_ctx = self.X_ctx[idx_r]
                S_t = best_S

            R_alg_t = self.true_departure_rate(idx_r, S_t)
            departed, _, j_local = self.sample_mnl_choice(idx_r, S_t)

            y_vec = torch.zeros(len(S_t), device=self.device)
            if j_local is not None:
                y_vec[j_local] = 1.0

            X_S = build_X_S_from_context(x_ctx, S_t, self.device)
            
            loss_mnl, _ = self.router.update(X_S, y_vec)

            # 로그는 평균처럼 보이게 T로 나눠서 저장
            loss_mnl = loss_mnl / max(getattr(self.router, "_T", 1), 1)
            self.log_l_mnl += float(loss_mnl)
            self.log_cnt_updates += 1

            if not departed:
                self.queue_router.append(idx_r)

        # 2) oracle service (precomputed best S)
        if len(self.queue_oracle) > 0:
            q_idxs = np.asarray(self.queue_oracle, dtype=np.int64)
            best_pos = int(np.argmax(self.max_departure_rates[q_idxs]))
            idx_o = int(q_idxs[best_pos])
            S_o = list(self.all_combis[int(self.best_S_idx[idx_o])])

            self.queue_oracle.remove(idx_o)
            departed_o, _, _ = self.sample_mnl_choice(idx_o, S_o)
            if not departed_o:
                self.queue_oracle.append(idx_o)

        # 3) regret update
        R_star_t = 0.0
        if len(X_t_snapshot) > 0:
            R_star_t = max(self.max_departure_rates[idx] for idx in X_t_snapshot)

        self.cum_regret += (R_star_t - R_alg_t)
        self.regret_history.append(self.cum_regret)

        # 4) i.i.d arrival with replacement
        A_curr = 0
        if self.rng.rand() < self.config.arrival_rate:
            new_idx = int(self.rng.choice(self.job_pool))
            self.queue_router.append(new_idx)
            self.queue_oracle.append(new_idx)
            self.last_arrival_idx = new_idx
            A_curr = 1

        Q_r, Q_o = len(self.queue_router), len(self.queue_oracle)
        self.Q_regret_history.append(Q_r - Q_o)

        # 5) exploration update
        eta_t = min(1.0, self.c0 * (t + 1.0) ** (-0.5))
        E_curr = 1 if (self.rng.rand() < eta_t) else 0
        self.A_prev, self.E_prev = A_curr, E_curr

        # done: max_steps로만 종료
        done = False

        if (self.steps % self.config.log_every == 0):
            avg_regret = self.cum_regret / self.steps
            avg_mnl_loss = (self.log_l_mnl / self.log_cnt_updates) if self.log_cnt_updates > 0 else 0.0
            print(
                f"[Step {self.steps}] Regret(avg)={avg_regret:.6f} "
                f"Q_gap={self.Q_regret_history[-1]} (Qr={Q_r},Qo={Q_o}) "
                f"MNL_Loss(avg)={avg_mnl_loss:.4f}"
            )
            self.log_l_mnl = 0.0
            self.log_cnt_updates = 0

        return done

    def run(self) -> Tuple[MNLRouter, float, float, List[float], List[float]]:
        while self.steps < self.config.max_steps:
            _ = self.step()

        Q_regret_T = self.Q_regret_history[-1] if self.Q_regret_history else 0.0
        avg_regret = self.cum_regret / max(1, self.steps)
        print(f"--- Finished. AvgRegret={avg_regret:.6f}, Final Q_gap={Q_regret_T} ---")
        return self.router, avg_regret, Q_regret_T, self.regret_history, self.Q_regret_history


def queue_env(X_ctx, acc_mat, util_mat, config, router):
    env = QueueEnv(X_ctx, acc_mat, util_mat, config, router)
    return env.run()
