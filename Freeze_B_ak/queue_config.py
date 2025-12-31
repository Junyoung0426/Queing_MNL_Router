# queue_config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class QueueConfig:

    seed: int = 42
    log_every: int = 100
    debug_verbose: bool = True
    debug_topk: int = 3
    debug_print_every: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    test_size: float = 0.2

    use_cost: bool = True
    lam_cost: float = 50.0

    d_ctx: Optional[int] = None
    n_models: Optional[int] = None

    assort_K: int = 2
    arrival_rate: float = 0.7
    max_steps: int = 100000

    kappa: float = 4.0
    c1: float = 30.0

    lambda_0: float = 1

    lbfgs_max_iter: int = 50
    lbfgs_history_size: int = 50
    lbfgs_line_search: str = "strong_wolfe"
    hist_init_capacity: int = 2048

    d_proj: int = 64
    b_type: str = "mlp"
    b_hidden_mult: int = 2

    supcon_temp: float = 0.1
    supcon_bs: int = 512
    offline_epochs: int = 1000
    offline_lr_B: float = 3e-4

    supcon_pos_strategy: str = "topr_mass"

    supcon_topk_max_k: int = 5
    supcon_topk_q: float = 0.75
    supcon_topk_beta: float = 20.0
    supcon_topk_delta: float = 1.0

    offline_total_ratio: float = 0.20
    tb_rand_frac: float = 0.70
    tb_win_frac: float = 0.30

    anchor_n_per_model: int = 10
    anchor_use_margin: bool = False
    anchor_margin_mode: str = "abs"
    anchor_seed_offset: int = 777

    anchor_xi_normalize: bool = False
    embed_a_normalize: bool = False
    embed_weight_mode: str = "topk_softmax"
    embed_topK: int = 3
    embed_tau: float = 0.2

    ak_init: str = "zeros"
