# queue_config.py

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class QueueConfig:
    # Router 내부 차원 관련 (NonB에서는 d_proj는 쓰지 않지만 인터페이스 유지)
    d_proj: int = 128
    lambda_0: float = 1.0
    supcon_temp: float = 0.07

    # SupCon 관련 (NonB에서는 실질적으로 사용하지 않는다)
    supcon_bs: int = 64
    supcon_weight: float = 0.0

    # Embedder / 데이터 스플릿
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_size: float = 0.2

    # utility(acc - λ * cost) 관련
    use_cost: bool = True
    lam_cost: float = 50.0

    assort_K: int = 2
    M_sample: Optional[int] = None # None이면 논문 식으로 M 계산

    # α_T 스케줄 관련 (논문 수식)
    # κ는 최소 4~5 이상 → 10 정도로 두는 설정
    kappa: float = 4.0
    alpha_T: Optional[float] = None  # 직접 α_T를 넘기고 싶으면 사용, None이면 수식으로 계산

    # c0는 충분히 큰 값 → 1e5 수준
    c0: float = 17.0  # η(t) = min{1, c0 (t+1)^(-1/2)}

    # Queueing 환경
    arrival_rate: float = 0.7
    max_steps: int = 30000

    # 로깅/시드
    log_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
