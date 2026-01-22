#B_MNL_Bandit/queue_config.py
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class QueueConfig:
    # -----------------------------
    # Logging / Seed
    # -----------------------------
    seed: int = 42
    log_every: int = 100
    debug_verbose: bool = True
    debug_topk: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Data
    # -----------------------------
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_size: float = 0.2

    use_cost: bool = True
    lam_cost: float = 5.0

    d_ctx: Optional[int] = None
    n_models: Optional[int] = None

    # combine_mode: str = "mul"   # "mul" or "add"
    explore_enabled: bool = True

    # -----------------------------
    # Queue / Bandit (Online)
    # -----------------------------
    assort_K: int = 1
    arrival_rate: float = 0.3
    max_steps: int = 20000

    # util -> r -> odds
    r_eps: float = 1e-6
    r_lo: float = 0.1
    r_hi: float = 0.99

    # unknown horizon schedule
    kappa: float =4.0
    c1: float = 11.0

    # ridge for theta + initial V scale (V = lambda_0 I + Σ z z^T)
    lambda_0: float = 1.0
    alpha_coef: float = 0.001

    # theta solver (full-history)
    theta_solver: str = "lbfgs"  #  "lbfgs"

    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 500
    lbfgs_line_search: str = "strong_wolfe"


    # history buffer capacity (z_S, y) full-history
    hist_init_capacity: int = 2048

    # -----------------------------
    # Offline split
    # -----------------------------
    offline_total_ratio: float = 0.10
    offline_tie_eps: float = 1e-9
    offline_seed_min_per_model: int = 10

    # -----------------------------
    # B Projection architecture (Offline)
    # -----------------------------
    d_proj: int = 64
    b_type: str = "none"          # "none" / "linear" / "mlp"
    b_hidden_mult: int = 2

    # -----------------------------
    # SupCon pretrain (B only, Offline)
    # -----------------------------
    supcon_temp: float = 0.07
    supcon_bs: int = 512
    offline_epochs: int = 10_000
    offline_lr_B: float = 3e-4
    supcon_grad_clip: float = 1.0

    # positives selection: "top1" / "topr_mass"
    supcon_pos_strategy: str = "topr_mass"
    supcon_topk_max_k: int = 100
    supcon_topk_q: float = 1
    supcon_topk_beta: float = 5.0
    supcon_topk_delta: float = 1.0

    # winner-balanced batch sampler (SupCon)
    balance_min_classes: int = 8
    balance_per_class: int = 64

