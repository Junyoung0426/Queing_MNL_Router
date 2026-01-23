# queue_config.py
from dataclasses import dataclass
from typing import Optional
import math
import torch


def _eta_mean(c1: float, T: int) -> float:
    if T <= 0:
        return 0.0
    s = 0.0
    for t in range(1, T + 1):
        v = float(c1) / math.sqrt(t + 1.0)
        if v > 1.0:
            v = 1.0
        s += v
    return s / float(T)


def _solve_c1(target_explore_rate: float, arrival_rate: float, max_steps: int) -> float:
    a = float(arrival_rate)
    T = int(max_steps)

    if T <= 0 or a <= 0.0:
        return 0.0

    p = float(target_explore_rate)
    if p <= 0.0:
        return 0.0

    if p >= a:
        return float(1.01 * math.sqrt(T + 1.0))

    target_eta = p / a

    lo = 0.0
    hi = 1.0
    for _ in range(60):
        if _eta_mean(hi, T) >= target_eta:
            break
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _eta_mean(mid, T) >= target_eta:
            hi = mid
        else:
            lo = mid

    return float(hi)


def _first_t_eta_lt_1(c1: float) -> int:
    c1 = float(c1)
    if c1 <= 0.0:
        return 1
    # eta(t)=min(1, c1/sqrt(t+1))
    # eta(t) < 1  <=>  c1/sqrt(t+1) < 1  <=>  t+1 > c1^2  <=>  t > c1^2 - 1
    # first integer t satisfying: t >= floor(c1^2)
    tau = int(math.floor(c1 * c1))
    return 1 if tau < 1 else tau


@dataclass
class QueueConfig:
    seed: int = 42
    log_every: int = 100
    debug_verbose: bool = True
    debug_topk: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_size: float = 0.2

    use_cost: bool = True
    lam_cost: float = 5.0

    d_ctx: Optional[int] = None
    n_models: Optional[int] = None

    explore_enabled: bool = True

    assort_K: int = 1
    arrival_rate: float = 0.3
    max_steps: int = 5000

    r_eps: float = 1e-6
    r_lo: float = 0.1
    r_hi: float = 0.99

    # TS 관련
    kappa: float = 4.0

    # 원하는 평균 forced-explore 비율(대략 decision-step 기준 근사)
    target_explore_rate: float = 0.20

    lambda_0: float = 1.0
    alpha_coef: float = 0.006

    theta_solver: str = "lbfgs"
    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 500
    lbfgs_line_search: str = "strong_wolfe"

    hist_init_capacity: int = 2048

    offline_total_ratio: float = 0.10
    offline_tie_eps: float = 1e-9
    offline_seed_min_per_model: int = 10

    d_proj: int = 64
    b_type: str = "none"
    b_hidden_mult: int = 2

    supcon_temp: float = 0.07
    supcon_bs: int = 512
    offline_epochs: int = 10_000
    offline_lr_B: float = 3e-4
    supcon_grad_clip: float = 1.0

    supcon_pos_strategy: str = "topr_mass"
    supcon_topk_max_k: int = 100
    supcon_topk_q: float = 1
    supcon_topk_beta: float = 5.0
    supcon_topk_delta: float = 1.0

    balance_min_classes: int = 8
    balance_per_class: int = 64

    @property
    def c1(self) -> float:
        if not self.explore_enabled:
            return 0.0
        v = _solve_c1(self.target_explore_rate, self.arrival_rate, self.max_steps)
        return round(float(v), 4)

    @property
    def mean_explore_rate(self) -> float:
        if not self.explore_enabled:
            return 0.0
        T = int(self.max_steps)
        a = float(self.arrival_rate)
        v = a * _eta_mean(float(self.c1), T)   # self.c1은 이미 반올림된 값
        return round(float(v), 4)

    @property
    def cqb_tau(self) -> int:
        if not self.explore_enabled:
            return 1
        T = int(self.max_steps)
        tau = _first_t_eta_lt_1(float(self.c1))
        if tau > T:
            tau = T
        return tau
