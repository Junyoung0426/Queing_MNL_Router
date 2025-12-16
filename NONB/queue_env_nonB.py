# queue_env_nonB.py

from typing import List, Tuple, Optional
from itertools import combinations
import math

import numpy as np
import torch
from mnl_router_nonB import MNLRouter, build_X_S_from_context_onehot as build_X_S_from_context

from queue_config import QueueConfig


class QueueEnv:
    """
    Unknown-horizon MNL queueing bandit 환경 클래스 (NonB 버전).

    - router: 학습 중인 MNLRouter (linear, no B)
    - queue_router: 에이전트 큐
    - queue_oracle: queue-length regret 계산용 optimal policy 큐

    변경 포인트:
      - exploitation에서 (queue × all_combis) 이중 for-loop를 유지하지 않고,
        all_combis 축을 배치로 묶어서 GPU에서 한 번에 평가한다.
      - oracle은 init에서 job별 best S를 완전탐색으로 precompute해서,
        step마다 조합 루프를 없앤다(의미는 동일).
    """

    def __init__(
        self,
        X_ctx: np.ndarray,     # (N_jobs, d_ctx)
        acc_mat: np.ndarray,   # (N_jobs, K_models)  -- 현재 알고리즘에서는 직접 사용하지 않음
        util_mat: np.ndarray,  # (N_jobs, K_models)
        config: QueueConfig,
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
        N2, N_models = acc_mat.shape
        N3, K2 = util_mat.shape
        assert N == N2 == N3, "X_ctx / acc_mat / util_mat 크기가 일치하지 않는다."
        assert N_models == K2, "acc_mat / util_mat의 모델 수가 다르다."
        if N == 0:
            raise ValueError("X_ctx is empty.")

        self.N = N
        self.d_ctx = d_ctx
        self.N_models = N_models

        # Router 초기화 (NonB: 입력차원 d_ctx + N_models)
        d_in = d_ctx + N_models
        self.router = MNLRouter(
            d_in=d_in,
            lambda_0=config.lambda_0,
            device=config.device,
        ).to(self.device)
        self.router.train()

        # 조합 집합 C (assortment size = assort_K)
        assert 1 <= config.assort_K <= N_models
        self.K = config.assort_K
        self.all_combis = list(combinations(range(N_models), self.K))
        self.C_size = len(self.all_combis)

        # 난수 생성기
        self.rng = np.random.RandomState(config.seed)

        # arrival 샘플링 job pool (중복 허용)
        self.job_pool = np.arange(self.N, dtype=np.int64)

        # 큐
        self.queue_router: List[int] = []
        self.queue_oracle: List[int] = []
        self.last_arrival_idx: Optional[int] = None

        # 직전 라운드의 arrival / exploration flag
        self.A_prev = 0
        self.E_prev = 0

        # 루프 상태
        self.steps = 0
        self.comb_idx = 0

        # regret / 로그
        self.cum_regret = 0.0
        self.regret_history: List[float] = []
        self.Q_regret_history: List[float] = []
        self.log_l_mnl = 0.0
        self.log_l_sc = 0.0

        # Unknown horizon 하이퍼파라미터
        self.lambda_0 = config.lambda_0
        self.kappa = config.kappa
        self.c0 = config.c0

        # feature dimension d
        self.d = self.router.d

        # Thompson sampling에서 필요한 고정 sample size M (논문 식)
        denom_M = math.log(1.0 - 1.0 / (4.0 * math.sqrt(math.e * math.pi)))
        self.M_sample = max(1, int(math.ceil(1.0 - math.log(self.K) / denom_M)))

        # exploitation에서 queue batch 크기 (메모리 상황에 맞춰 조절 가능)
        self.exploit_job_batch = 128

        print(
            f"[Queue-Env] N_jobs={self.N}, models={self.N_models}, |C|={self.C_size}, "
            f"arrival_rate={self.config.arrival_rate}, max_steps={self.config.max_steps}, "
            f"lambda_0={self.lambda_0}, kappa={self.kappa}, c0={self.c0}, M={self.M_sample}, "
            f"job_pool={len(self.job_pool)}"
        )

        # -------- torch 캐시 (exploitation 배치화용) --------
        self.X_ctx_t = torch.from_numpy(self.X_ctx).float().to(self.device)     # (N, d_ctx)
        self.onehots = torch.eye(self.N_models, device=self.device)             # (N_models, N_models)
        self.S_tensor = torch.tensor(self.all_combis, device=self.device, dtype=torch.long)  # (C, K)
        self.oh_combis = self.onehots[self.S_tensor]                            # (C, K, N_models)
        self.oh_combis_flat = self.oh_combis.reshape(self.C_size * self.K, self.N_models)   # (C*K, N_models)

        # ---------- precompute: 각 job i별 max_S R(i,S,θ*) + argmax S ----------
        # (조합 완전탐색을 init에서 1번만 하고, oracle step에서는 이 값을 그대로 쓴다)
        print("[Queue-Env] Pre-computing per-job max departure rates + best S...")
        self.max_departure_rates = np.zeros(self.N, dtype=np.float64)
        self.best_S_idx = np.zeros(self.N, dtype=np.int64)

        combi_idx_np = np.asarray(self.all_combis, dtype=np.int64)  # (C, K)
        for i in range(self.N):
            util_row = self.util_mat[i]                # (N_models,)
            util_S = util_row[combi_idx_np]            # (C, K)

            max_u = util_S.max(axis=1)                 # (C,)
            exps = np.exp(util_S - max_u[:, None])     # (C, K)
            sumexp = exps.sum(axis=1)                  # (C,)
            denom = np.exp(-max_u) + sumexp            # (C,)
            rates = sumexp / denom                     # (C,)

            best_c = int(np.argmax(rates))
            self.best_S_idx[i] = best_c
            self.max_departure_rates[i] = float(rates[best_c])

        print("[Queue-Env] Done pre-computing.")

    # ---------- util_mat 기반 departure rate ----------
    def true_departure_rate(self, job_idx: int, S: List[int]) -> float:
        """
        util_mat 기반 MNL
        R(x,S,θ*) = sum_{k∈S} exp(u_k) / (1 + sum_{k∈S} exp(u_k))
        """
        util_row = self.util_mat[job_idx]
        util_S = np.array([util_row[k] for k in S], dtype=np.float64)
        if util_S.size == 0:
            return 0.0
        max_u = float(util_S.max())
        exps = np.exp(util_S - max_u)
        denom = math.exp(-max_u) + float(exps.sum())
        return float(exps.sum() / denom)

    # ---------- util_mat 기반 MNL choice 샘플 ----------
    def sample_mnl_choice(self, job_idx: int, S: List[int]):
        """
        util 기반 MNL에서 (outside 포함) 한 번 샘플한다.

        Returns
        -------
        departed : bool
        chosen_k : Optional[int]
        j_local  : Optional[int]
        """
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
        else:
            j_local = choice - 1
            return True, S[j_local], j_local

    # ---------- exploitation: 조합 완전탐색을 GPU 배치로 ----------
    def _select_best_by_exhaustive_batch(
        self,
        noise_vectors: torch.Tensor,   # (d, M)
    ) -> Tuple[int, List[int]]:
        """
        queue_router 안에서
          argmax_{x in queue} argmax_{S in all_combis} R^e(x,S)
        를 구한다. 
        """
        q = self.queue_router
        if len(q) == 0:
            raise RuntimeError("queue_router is empty")

        q_idxs_t = torch.tensor(q, device=self.device, dtype=torch.long)  # (B_total,)

        best_val = None
        best_job = None
        best_c = None

        C = self.C_size
        K = self.K
        d_in = self.d_ctx + self.N_models
        CK = C * K

        bs = self.exploit_job_batch
        for st in range(0, q_idxs_t.numel(), bs):
            ed = min(st + bs, q_idxs_t.numel())
            idxs = q_idxs_t[st:ed]                 # (B,)
            Xb = self.X_ctx_t[idxs]                # (B, d_ctx)
            B = Xb.shape[0]

            # (B, C*K, d_ctx) + (B, C*K, N_models) -> (B, C*K, d_in)
            x_rep = Xb[:, None, :].expand(B, CK, self.d_ctx)
            oh_rep = self.oh_combis_flat[None, :, :].expand(B, CK, self.N_models)
            X_flat = torch.cat([x_rep, oh_rep], dim=-1)                 # (B, C*K, d_in)

            # (B*C, K, d_in) 로 재구성 후 batch 평가
            X_all = X_flat.view(B * C, K, d_in)                         # (B*C, K, d_in)
            vals = self.router.sample_optimistic_reward_batch(X_all, noise_vectors)  # (B*C,)
            vals = vals.view(B, C)                                      # (B, C)

            best_vals_job, best_c_job = vals.max(dim=1)                 # (B,), (B,)
            chunk_best_val, pos = best_vals_job.max(dim=0)              # scalar, scalar

            if best_val is None or chunk_best_val.item() > best_val:
                best_val = float(chunk_best_val.item())
                best_job = int(idxs[pos].item())
                best_c = int(best_c_job[pos].item())

        assert best_job is not None and best_c is not None
        return best_job, list(self.all_combis[best_c])

    # ---------- 한 step 실행 ----------
    def step(self):
        self.steps += 1
        t = self.steps

        # standard regret는 서비스 전에 snapshot을 뜬다
        X_t_snapshot = list(self.queue_router)

        R_alg_t = 0.0
        R_star_t = 0.0

        # (1) router queue service (agent 정책)
        if len(self.queue_router) > 0:
            do_explore = (self.A_prev == 1) and (self.E_prev == 1)

            # η(t)-exploration: 직전 도착 job을 뽑고 round-robin S를 준다
            if do_explore and (self.last_arrival_idx is not None) and (
                self.last_arrival_idx in self.queue_router
            ):
                idx_r = self.last_arrival_idx
                self.queue_router.remove(idx_r)

                x_ctx = self.X_ctx[idx_r]
                S_t = list(self.all_combis[self.comb_idx])
                self.comb_idx = (self.comb_idx + 1) % self.C_size

            # exploitation with optimistic reward (조합 완전탐색 유지, GPU 배치화)
            else:
                t_eff = max(t - 1, 1)
                term1 = self.d * math.log(
                    1.0 + (t_eff * self.K) / (self.d * self.lambda_0)
                )
                term2 = 4.0 * math.log(t_eff)
                term3 = self.kappa * math.sqrt(self.lambda_0)
                beta_term = term1 + term2 + term3
                alpha_t = 0.5 * self.kappa * math.sqrt(beta_term)

                noise_vectors = self.router.sample_theta_noise(
                    alpha_t=alpha_t, M=self.M_sample
                )

                best_idx, best_S = self._select_best_by_exhaustive_batch(noise_vectors)

                self.queue_router.remove(best_idx)
                idx_r = best_idx
                x_ctx = self.X_ctx[idx_r]
                S_t = best_S

            # router policy의 departure rate (진짜 θ* 기준, util_mat 사용)
            R_alg_t = self.true_departure_rate(idx_r, S_t)

            # 실제 MNL choice 샘플 → 라벨 & 큐 업데이트
            departed, k_chosen, j_local = self.sample_mnl_choice(idx_r, S_t)

            y_vec = torch.zeros(len(S_t), device=self.device)
            if j_local is not None:
                y_vec[j_local] = 1.0

            # SupCon buffer: NonB에서는 no-op
            if k_chosen is not None:
                x_win = build_X_S_from_context(
                    x_ctx, [k_chosen], self.N_models, self.device
                )[0]
                self.router.add_to_buffer(x_win, k_chosen)

            # joint update: θ + V_inv
            X_S = build_X_S_from_context(x_ctx, S_t, self.N_models, self.device)
            loss_mnl, loss_sc = self.router.update(X_S, y_vec)
            loss_mnl = loss_mnl / max(self.router._T, 1)   # _T는 router 내부 누적 라운드 수
            self.log_l_mnl += loss_mnl

            self.log_l_sc += loss_sc

            # departure 실패면 job 다시 큐에 삽입
            if not departed:
                self.queue_router.append(idx_r)

        # (2) oracle queue service (queue-length regret용 optimal policy)
        if len(self.queue_oracle) > 0:
            q_idxs = np.asarray(self.queue_oracle, dtype=np.int64)
            best_pos = int(np.argmax(self.max_departure_rates[q_idxs]))
            idx_o = int(q_idxs[best_pos])

            S_o = list(self.all_combis[int(self.best_S_idx[idx_o])])

            self.queue_oracle.remove(idx_o)

            departed_o, _, _ = self.sample_mnl_choice(idx_o, S_o)
            if not departed_o:
                self.queue_oracle.append(idx_o)

        # (3) standard regret용 optimal R_star_t 계산
        if len(X_t_snapshot) > 0:
            R_star_t = max(self.max_departure_rates[idx] for idx in X_t_snapshot)
        else:
            R_star_t = 0.0

        # (4) standard regret 업데이트
        self.cum_regret += (R_star_t - R_alg_t)
        self.regret_history.append(self.cum_regret)

        # (5) 이번 round 후 새로운 arrival A(t): job_pool에서 중복 허용 랜덤 샘플링
        A_curr = 0
        if self.rng.rand() < self.config.arrival_rate:
            new_idx = int(self.rng.choice(self.job_pool))
            self.queue_router.append(new_idx)
            self.queue_oracle.append(new_idx)
            self.last_arrival_idx = new_idx
            A_curr = 1

        # (6) queue-length 차이 (Q(t+1) - Q*(t+1))
        Q_r = len(self.queue_router)
        Q_o = len(self.queue_oracle)
        self.Q_regret_history.append((Q_r - Q_o))

        # (7) E(t) ~ Bern(η(t)),  η(t) = min{1, c0 (t+1)^(-1/2)}
        eta_t = min(1.0, self.c0 * (t + 1.0) ** (-0.5))
        E_curr = 1 if (self.rng.rand() < eta_t) else 0

        # (8) 다음 라운드를 위한 상태 업데이트
        self.A_prev = A_curr
        self.E_prev = E_curr

        # (9) 종료 신호: max_steps로만 종료
        done = False

        # (10) 로깅
        if (self.steps % self.config.log_every == 0):
            avg_reg = self.cum_regret / self.steps
            q_diff = self.Q_regret_history[-1]
            print(
                f"[Queue-Env step={self.steps}] "
                f"regret(avg)={avg_reg:.6f}  "
                f"Q-reg(T_est)={q_diff:.3f}  "
                f"Q_r={Q_r} Q_o={Q_o}  "
                f"L_mnl(avg)={self.log_l_mnl / self.config.log_every:.4f} "
                f"L_supcon(avg)={self.log_l_sc / self.config.log_every:.4f}"
            )
            self.log_l_mnl, self.log_l_sc = 0.0, 0.0

        return done

    # ---------- 전체 시뮬레이션 실행 ----------
    def run(self) -> Tuple[MNLRouter, float, float, List[float], List[float]]:
        while self.steps < self.config.max_steps:
            _ = self.step()

        Q_regret_T = self.Q_regret_history[-1] if len(self.Q_regret_history) > 0 else 0.0
        avg_regret = self.cum_regret / max(self.steps, 1)

        print(
            f"--- [Queue-Env] training done. "
            f"avg_regret={avg_regret:.6f}, Q_regret_T={Q_regret_T:.3f} ---"
        )

        return (
            self.router,
            avg_regret,
            Q_regret_T,
            self.regret_history,
            self.Q_regret_history,
        )


def queue_env(
    X_ctx: np.ndarray,
    acc_mat: np.ndarray,
    util_mat: np.ndarray,
    config: QueueConfig,
) -> Tuple[MNLRouter, float, float, List[float], List[float]]:
    env = QueueEnv(X_ctx=X_ctx, acc_mat=acc_mat, util_mat=util_mat, config=config)
    return env.run()
