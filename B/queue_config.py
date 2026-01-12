# queue_config.py
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
    debug_print_every: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Data
    # -----------------------------
    seed: int = 42
    log_every: int = 100
    debug_verbose: bool = True
    debug_topk: int = 3
    debug_print_every: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Data
    # -----------------------------
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_size: float = 0.2

    use_cost: bool = True
    lam_cost: float = 50.0

    # (data-dependent)
    d_ctx: Optional[int] = None
    n_models: Optional[int] = None

    # -----------------------------
    # Queue / Bandit (Online)
    # -----------------------------
    assort_K: int = 2
    arrival_rate: float = 0.7
    max_steps: int = 200000

    # util -> r -> odds 분모 안정화
    r_eps: float = 1e-6
    r_lo: float = 0.000
    r_hi: float = 1
    r_floor: Optional[float] = None

    # unknown horizon schedule
    kappa: float = 4.0
    c1: float = 60.0

    # ridge for theta + initial V_inv scale
    lambda_0: float = 1.0
    alpha_coef: float = 0.01


    # Online LBFGS
    lbfgs_max_iter: int = 50
    lbfgs_history_size: int = 50
    lbfgs_line_search: str = "strong_wolfe"
    hist_init_capacity: int = 2048

    # -----------------------------
    # TB(offline pool) split inside Train
    # -----------------------------
    offline_total_ratio: float = 0.20
    tb_rand_frac: float = 0.70
    tb_win_frac: float = 0.30

    # -----------------------------
    # B Projection architecture (Offline)
    # -----------------------------
    d_proj: int = 64
    b_type: str = "mlp"          # "none" / "linear" / "mlp"
    b_hidden_mult: int = 2

    # -----------------------------
    # SupCon pretrain (B only, Offline)
    # -----------------------------
    supcon_temp: float = 0.07
    supcon_bs: int = 512
    offline_epochs: int = 10_000
    offline_lr_B: float = 3e-4
    supcon_grad_clip: float = 1.0

    # positives selection: "top1" / "topr_mass" / "topr_margin"

    # positives selection: "top1" / "topr_mass" / "topr_margin"
    supcon_pos_strategy: str = "top1"
    supcon_topk_max_k: int = 3
    supcon_topk_q: float = 0.8
    supcon_topk_beta: float = 5.0
    supcon_topk_delta: float = 1.0

    # winner-balanced batch sampler (SupCon)
    balance_min_classes: int = 8
    balance_per_class= 64

    # -----------------------------
    # a_table (LLM embedding table) build configs (Offline)
    # -----------------------------
    # Step 1: Anchor selection
    anchor_n_per_model: int = 10
    anchor_use_margin: bool = False
    anchor_margin_mode: str = "abs"   # "abs" / "gap"
    anchor_seed_offset: int = 777

    # Step 2: xi centroids
    anchor_xi_normalize: bool = False

    # Step 3: build a_table from xi & score matrix S
    embed_a_normalize: bool = False
    embed_weight_mode: str = "topk_softmax"  # "topk_softmax" / "softmax_all" / "self_centroid"
    embed_topK: int = 3
    embed_tau: float = 0.2

    ak_init: str = "zeros"
