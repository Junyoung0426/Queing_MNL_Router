# queue_config.py
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class QueueConfig:
    # -----------------------------
    # Model Dimensions
    # -----------------------------
    d_proj: int = 128
    d_model_emb: int = 128

    # B type ("linear" or "mlp")
    b_type: str = "mlp"
    b_hidden_mult: int = 2

    # data-dependent (로드 후 채움)
    d_ctx: Optional[int] = None
    n_models: Optional[int] = None

    # -----------------------------
    # Offline Representation Learning (SupCon)
    # -----------------------------
    reg_lambda: float = 1.0
    supcon_temp: float = 0.07
    supcon_bs: int = 512
    offline_epochs: int = 500
    offline_lr_B: float = 0.00005
    offline_ratio: float = 0.2

    # Online SupCon (보통 0으로 둔다)
    supcon_weight: float = 0.0

    # -----------------------------
    # Bandit & Queue
    # -----------------------------
    assort_K: int = 2
    arrival_rate: float = 0.7
    max_steps: int = 100000

    kappa: float = 4.0
    c0: float = 12.0

    # 필요하면 override (None이면 reg_lambda 사용)
    lambda0: Optional[float] = None

    # -----------------------------
    # Data & System
    # -----------------------------
    test_size: float = 0.3
    use_cost: bool = True
    lam_cost: float = 50.0

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    log_every: int = 1000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
