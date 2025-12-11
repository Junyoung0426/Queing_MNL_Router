from typing import List, Tuple, Optional
from itertools import combinations
import math

import numpy as np
import torch
from mnl_router import MNLRouter, build_X_S_from_context
# from mnl_router_nonB import MNLRouter, build_X_S_from_context
from queue_config import QueueConfig


class QueueEnv:
    """
    Unknown-horizon MNL queueing bandit 환경 클래스.

    - router: 학습 중인 MNLRouter
    - queue_router: 에이전트 큐
    - queue_oracle: queue-length regret 계산용 optimal policy 큐
    """

    def __init__(
        self,
        X_ctx: np.ndarray,     # (N_jobs, d_ctx)
        acc_mat: np.ndarray,   # (N_jobs, K)  -- 현재 알고리즘에서는 직접 사용하지 않음
        util_mat: np.ndarray,  # (N_jobs, K)  -- MNL 유틸리티 (perf - λ * cost 등)
        config: QueueConfig,
    ):
        self.X_ctx = X_ctx
        self.acc_mat = acc_mat
        self.util_mat = util_mat
        self.config = config

        # 공통 셋업
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

        # Router 초기화
        d_in = d_ctx + N_models
        self.router = MNLRouter(
            d_in=d_in,
            d_proj=config.d_proj,
            lam_ridge=config.reg_lambda,
            supcon_temp=config.supcon_temp,
            device=config.device,
        ).to(self.device)
        self.router.train()

        # 조합 집합 C (assortment size = assort_K)
        assert 1 <= config.assort_K <= N_models
        self.all_combis = list(combinations(range(N_models), config.assort_K))
        self.C_size = len(self.all_combis)

        # 난수 생성기
        self.rng = np.random.RandomState(config.seed)

        # 큐 및 인덱스
        self.queue_router: List[int] = []
        self.queue_oracle: List[int] = []
        self.next_job_idx = 0

        # 마지막으로 도착한 job index (A(t)=1일 때 들어온 job)
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
        self.lambda0 = config.lambda0 if config.lambda0 is not None else config.reg_lambda
        self.kappa = config.kappa
        self.c0 = config.c0

        self.K = N_models
        # nonB일때 
        # self.d = self.router.d
        # B로 하ㄹ때

        self.d = self.router.d_proj

        # Thompson sampling에서 필요한 고정 sample size M (논문 식)
        denom_M = math.log(1.0 - 1.0 / (4.0 * math.sqrt(math.e * math.pi)))
        self.M_sample = max(1, int(math.ceil(1.0 - math.log(self.K) / denom_M)))

        print(
            f"[Queue-Env] N_jobs={self.N}, K={self.N_models}, |C|={self.C_size}, "
            f"arrival_rate={self.config.arrival_rate}, max_steps={self.config.max_steps}, "
            f"lambda0={self.lambda0}, kappa={self.kappa}, c0={self.c0}, M={self.M_sample}"
        )

        # ---------- precompute: 각 job i별 max_S R(i,S,θ*) ----------
        print("[Queue-Env] Pre-computing per-job max departure rates...")
        self.max_departure_rates = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            util_row = self.util_mat[i]  # (K,)
            best_rate = 0.0
            for S in self.all_combis:
                idx_list = list(S)
                util_S = util_row[idx_list]          # (|S|,)
                if util_S.size == 0:
                    continue
                max_u = float(util_S.max())
                exps = np.exp(util_S - max_u)
                denom = math.exp(-max_u) + float(exps.sum())
                rate = float(exps.sum() / denom)
                if rate > best_rate:
                    best_rate = rate
            self.max_departure_rates[i] = best_rate
        print("[Queue-Env] Done pre-computing max departure rates.")

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
        util 기반 MNL에서 (outside 포함) 한 번 샘플.

        Returns
        -------
        departed : bool          # 서버 선택됐으면 True
        chosen_k : Optional[int] # 선택 서버 index (outside면 None)
        j_local  : Optional[int] # S 내에서의 local index
        """
        util_row = self.util_mat[job_idx]
        util_S = np.array([util_row[k] for k in S], dtype=np.float64)
        if util_S.size == 0:
            return False, None, None

        max_u = float(util_S.max())
        exps = np.exp(util_S - max_u)
        denom = math.exp(-max_u) + float(exps.sum())
        p_out = math.exp(-max_u) / denom  # 아무 서버 선택 X (outside)
        p_in = exps / denom               # 서버 선택 확률 (|S|,)

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

    # ---------- 한 step 실행 ----------
    def step(self):
        """
        한 time step(t)을 진행하고,
        standard regret 및 queue-length regret을 업데이트한다.
        """
        self.steps += 1
        t = self.steps

        # Eq.(60) 정의에 맞게, standard regret는
        # "현재 agent queue 상태 X_t" 기준으로 계산해야 하므로,
        # router가 서비스를 하기 전에 snapshot을 떠둔다.
        X_t_snapshot = list(self.queue_router)

        R_alg_t = 0.0   # agent가 실제 선택한 (x_t, S_t)에 대한 R(x_t, S_t, θ*)
        R_star_t = 0.0  # 같은 X_t_snapshot에서의 최적 (x_t^*, S_t^*) 의 R(x_t^*, S_t^*, θ*)

        # (1) router queue service (agent 정책)
        if len(self.queue_router) > 0:
            do_explore = (self.A_prev == 1) and (self.E_prev == 1)

            # ---------- η(t)-exploration ----------
            if do_explore and (self.last_arrival_idx is not None) and (
                self.last_arrival_idx in self.queue_router
            ):
                idx_r = self.last_arrival_idx
                self.queue_router.remove(idx_r)

                x_ctx = self.X_ctx[idx_r]
                S_t = list(self.all_combis[self.comb_idx])
                self.comb_idx = (self.comb_idx + 1) % self.C_size

            # ---------- exploitation with optimistic reward ----------
            else:
                # α_{t-1} 계산 (t_eff = max(t-1, 1))
                t_eff = max(t - 1, 1)
                term1 = self.d * math.log(
                    1.0 + (t_eff * self.K) / (self.d * self.lambda0)
                )
                term2 = 4.0 * math.log(t_eff)
                term3 = self.kappa * math.sqrt(self.lambda0)
                beta_term = term1 + term2 + term3
                alpha_t = 0.5 * self.kappa * math.sqrt(beta_term)

                # TS 기반 optimistic reward:
                noise_vectors = self.router.sample_theta_noise(
                    alpha_t=alpha_t, M=self.M_sample
                )

                best_val = -1e18
                best_S: Optional[List[int]] = None
                best_idx: Optional[int] = None

                for idx_cand in self.queue_router:   # x ∈ X_t
                    x_c = self.X_ctx[idx_cand]
                    for S in self.all_combis:        # S ∈ C
                        S_list = list(S)
                        X_S_candidate = build_X_S_from_context(
                            x_c, S_list, self.N_models, self.device
                        )
                        val = self.router.sample_optimistic_reward(
                            X_S_candidate,
                            alpha_t=alpha_t,
                            M=self.M_sample,
                            noise_vectors=noise_vectors,
                        )
                        if val > best_val:
                            best_val = val
                            best_S = S_list
                            best_idx = idx_cand

                assert best_idx is not None and best_S is not None
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

            # SupCon buffer: 실제 선택된 서버만 저장
            if k_chosen is not None:
                x_win = build_X_S_from_context(
                    x_ctx, [k_chosen], self.N_models, self.device
                )[0]
                self.router.add_to_buffer(x_win, k_chosen)

            # joint update: θ + B + V_inv
            X_S = build_X_S_from_context(x_ctx, S_t, self.N_models, self.device)
            loss_mnl, loss_sc = self.router.update(
                X_S,
                y_vec,
                batch_supcon_size=self.config.supcon_bs,
                lambda_sc=self.config.supcon_weight,
            )
            self.log_l_mnl += loss_mnl
            self.log_l_sc += loss_sc

            # departure 실패면 job 다시 큐에 삽입
            if not departed:
                self.queue_router.append(idx_r)

        # (2) oracle queue service (queue-length regret용 optimal policy π*)
        #     여기서는 queue_oracle 위에서 최적 action을 택하고,
        #     Q*(t)을 추적하기 위해 departure 를 시뮬레이션만 한다.
        if len(self.queue_oracle) > 0:
            best_idx_o: Optional[int] = None
            best_S_o: Optional[List[int]] = None
            best_R_o = -1e18

            for idx_cand in self.queue_oracle:
                for S in self.all_combis:
                    S_list = list(S)
                    R_val = self.true_departure_rate(idx_cand, S_list)
                    if R_val > best_R_o:
                        best_R_o = R_val
                        best_idx_o = idx_cand
                        best_S_o = S_list

            assert best_idx_o is not None and best_S_o is not None
            self.queue_oracle.remove(best_idx_o)

            idx_o = best_idx_o
            S_o = best_S_o

            departed_o, _, _ = self.sample_mnl_choice(idx_o, S_o)
            if not departed_o:
                self.queue_oracle.append(idx_o)

        # (3) standard regret용 optimal R_star_t 계산
        #     같은 queue 상태 X_t_snapshot에서 x_t^*, S_t^* = argmax_{x∈X_t,S∈C} R(x,S,θ*)
        #     미리 계산한 per-job max R(i) 를 사용해서 O(|X_t|)로 계산
        if len(X_t_snapshot) > 0:
            R_star_t = max(self.max_departure_rates[idx] for idx in X_t_snapshot)
        else:
            R_star_t = 0.0

        # (4) standard regret 업데이트: R*(t) - R_alg(t)
        self.cum_regret += (R_star_t - R_alg_t)
        self.regret_history.append(self.cum_regret)

        # (5) 이번 round 후 새로운 arrival A(t)
        A_curr = 0
        if (self.next_job_idx < self.N) and (
            self.rng.rand() < self.config.arrival_rate
        ):
            self.queue_router.append(self.next_job_idx)
            self.queue_oracle.append(self.next_job_idx)
            self.last_arrival_idx = self.next_job_idx
            self.next_job_idx += 1
            A_curr = 1

        # (6) queue-length 차이 (Q(t+1) - Q*(t+1))
        Q_r = len(self.queue_router)
        Q_o = len(self.queue_oracle)
        self.Q_regret_history.append(abs(Q_r - Q_o))

        # (7) E(t) ~ Bern(η(t)),  η(t) = min{1, c0 (t+1)^(-1/2)}
        eta_t = min(1.0, self.c0 * (t + 1.0) ** (-0.5))
        E_curr = 1 if (self.rng.rand() < eta_t) else 0

        # (8) 다음 라운드를 위한 상태 업데이트
        self.A_prev = A_curr
        self.E_prev = E_curr

        # (9) 모든 job 소진 & 두 큐 모두 비었으면 종료 신호
        done = (
            (self.next_job_idx >= self.N)
            and (Q_r == 0)
            and (Q_o == 0)
        )

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
        """
        max_steps까지 queue 시뮬레이션을 돌리고,
        router와 regret 통계를 반환한다.
        """
        while self.steps < self.config.max_steps:
            done = self.step()
            if done:
                print(f"... [Queue-Env] simulation ended early at step={self.steps}")
                break

        # 최종 queue-length regret: Q(T) - Q*(T)
        Q_regret_T = self.Q_regret_history[-1] if len(self.Q_regret_history) > 0 else 0.0

        # standard regret 평균 (per time-step)
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
    """
    기존 인터페이스를 유지하기 위한 래퍼 함수.
    내부적으로 QueueEnv 클래스를 생성하고 run()을 호출한다.
    """
    env = QueueEnv(X_ctx=X_ctx, acc_mat=acc_mat, util_mat=util_mat, config=config)
    return env.run()
