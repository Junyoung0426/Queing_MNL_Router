# queue_config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class QueueConfig:
    # Router 내부
    d_proj: int = 128
    reg_lambda: float = 1.0
    supcon_temp: float = 0.07

    supcon_bs: int = 64
    supcon_weight: float = 1.0

    # Algorithm 1 관련
    beta_ucb: float = 1.0          # (fallback) 직접 α_T를 주고 싶을 때 쓸 고정값
    tau_pe: int = 1000
    eta: Optional[float] = None    # None이면 T^{-1/2}

    assort_K: int = 2
    M_sample: Optional[int] = None # None이면 논문 식으로 M 계산

    # α_T 스케줄 관련 (논문 수식)
    kappa: float = 10             # κ 4 ~ 무한 
    lambda0: Optional[float] = None  # 1.0 λ0 (None이면 reg_lambda 사용)
    alpha_T: Optional[float] = None  # 직접 α_T를 넘기고 싶으면 사용, None이면 수식으로 계산

    # Algorithm 2 (unknown horizon) 관련
    c0: float = 1.0                # η(t) = min{1, c0 (t+1)^(-1/2)} 10, 10000, 10000000

    # Queueing 환경
    arrival_rate: float = 0.7
    max_steps: int = 30000

    # 로깅/시드
    log_every: int = 1000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"