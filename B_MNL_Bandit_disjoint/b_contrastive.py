# b_contrastive.py
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from queue_config import QueueConfig


def supcon_loss_posmask(
    features: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    features: (B,d) float32 권장
    pos_mask: (B,B) float/bool (diagonal ignored)
    """
    B = int(features.shape[0])
    if B < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    feats = F.normalize(features, dim=-1)
    pm = pos_mask.to(device=features.device, dtype=features.dtype)

    eye = torch.eye(B, device=features.device, dtype=features.dtype)
    pm = pm * (1.0 - eye)

    sim = (feats @ feats.T) / float(temperature)   # (B,B)
    sim = sim - eye * 1e9                          # remove self in denom

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_sum = pm.sum(dim=1).clamp_min(1e-12)
    loss_i = -(log_prob * pm).sum(dim=1) / pos_sum
    return loss_i.mean()


@torch.no_grad()
def build_pos_mask_adaptive_topk(
    util_batch: torch.Tensor,
    n_models: int,
    max_k: int,
    mode: str,
    delta: float,
    beta: float,
    q: float,
) -> torch.Tensor:
    """
    util_batch: (B,K_data) -> (B,n_models)로 pad/trim
    return pos_weight: (B,B)  (util_batch dtype 유지)
    """
    ub = util_batch
    B, K = ub.shape

    if K != n_models:
        if K > n_models:
            ub = ub[:, :n_models]
        else:
            pad = torch.zeros((B, n_models - K), device=ub.device, dtype=ub.dtype)
            ub = torch.cat([ub, pad], dim=1)
        B, K = ub.shape

    max_k = int(min(int(max_k), K))
    top_vals, top_idx = torch.topk(ub, k=max_k, dim=1)
    top1 = top_vals[:, [0]]

    mode = str(mode).lower().strip()
    if mode not in ("mass", "margin"):
        raise ValueError(f"mode must be 'mass' or 'margin', got {mode}")

    if mode == "margin":
        keep = (top_vals >= (top1 - float(delta)))
        keep[:, 0] = True

        delta_eff = max(float(delta), 1e-6)
        w = ((top_vals - (top1 - delta_eff)) / delta_eff).clamp(0.0, 1.0)
        w[:, 0] = 1.0
        w = w * keep.to(dtype=ub.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

        membership = torch.zeros((B, K), device=ub.device, dtype=ub.dtype)
        membership.scatter_(1, top_idx, w)

        pos_weight = membership @ membership.T
        pos_weight.fill_diagonal_(0.0)
        return pos_weight

    # mode == "mass"
    v = top_vals - top_vals.max(dim=1, keepdim=True).values
    p = torch.softmax(float(beta) * v, dim=1)

    # 기존
    # c = torch.cumsum(p, dim=1)

    # 수정 (CUDA deterministic 우회)
    if p.is_cuda and torch.are_deterministic_algorithms_enabled():
        c = torch.cumsum(p.detach().cpu(), dim=1).to(p.device)
    else:
        c = torch.cumsum(p, dim=1)

    cond = (c >= float(q))
    any_true = cond.any(dim=1)
    first = cond.int().argmax(dim=1)
    first = torch.where(any_true, first, torch.full_like(first, max_k - 1))
    k_i = first + 1

    ar = torch.arange(max_k, device=ub.device).unsqueeze(0)
    keep = (ar < k_i.unsqueeze(1))

    w = p * keep.to(dtype=ub.dtype)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

    membership = torch.zeros((B, K), device=ub.device, dtype=ub.dtype)
    membership.scatter_(1, top_idx, w)

    pos_weight = membership @ membership.T
    pos_weight.fill_diagonal_(0.0)
    return pos_weight


def _tie_break_winners_from_util(
    util_np: np.ndarray,
    rng: np.random.RandomState,
    tie_eps: float,
) -> np.ndarray:
    """
    util_np: (N,K)
    max 동점이면 co-first 집합에서 랜덤하게 winner를 하나 뽑는다.
    """
    U = np.asarray(util_np, dtype=np.float32)
    N, K = U.shape
    mx = U.max(axis=1, keepdims=True)
    co = U >= (mx - float(tie_eps))  # (N,K)
    winners = np.empty((N,), dtype=np.int64)
    for i in range(N):
        cand = np.where(co[i])[0]
        if cand.size == 0:
            winners[i] = int(np.argmax(U[i]))
        else:
            winners[i] = int(rng.choice(cand, size=1)[0])
    return winners


def sample_winner_balanced_batch_indices(
    winners: np.ndarray,
    bs: int,
    rng: np.random.RandomState,
    min_classes: int,
    per_class: int,
) -> np.ndarray:
    N = int(winners.shape[0])
    bs = int(bs)

    present = np.unique(winners)
    present = present[present >= 0]
    if present.size == 0:
        return rng.choice(N, size=bs, replace=(N < bs)).astype(np.int64)

    C = min(int(min_classes), int(present.size))
    chosen_classes = rng.choice(present, size=C, replace=False)

    take = int(per_class)
    if take * C > bs:
        take = max(1, bs // C)

    idx_list = []
    for c in chosen_classes:
        pool = np.where(winners == c)[0]
        if pool.size == 0:
            continue
        sel = rng.choice(pool, size=take, replace=(pool.size < take))
        idx_list.append(sel)

    if len(idx_list) == 0:
        return rng.choice(N, size=bs, replace=(N < bs)).astype(np.int64)

    idx = np.concatenate(idx_list)
    if idx.size < bs:
        extra = rng.choice(N, size=(bs - idx.size), replace=True)
        idx = np.concatenate([idx, extra])
    if idx.size > bs:
        idx = idx[:bs]
    return idx.astype(np.int64)


def offline_pretrain_B_supcon(
    router,
    X_ctx_off: np.ndarray,
    util_off: np.ndarray,
    n_models: int,
    cfg: QueueConfig,
):
    if getattr(router, "b_type", "").lower().strip() == "none":
        print("[Offline] skipped (b_type=none)")
        return

    N = int(X_ctx_off.shape[0])
    if int(cfg.offline_epochs) <= 0 or N < 2:
        print("[Offline] skipped (offline_epochs<=0 or N_off<2)")
        return

    router.unfreeze_B(lr_b=float(cfg.offline_lr_B))
    rng = np.random.RandomState(int(cfg.seed) + 2024)

    # float32 end-to-end
    X_t = torch.from_numpy(np.asarray(X_ctx_off)).to(device=router.dev, dtype=torch.float32)
    U_t = torch.from_numpy(np.asarray(util_off)).to(device=router.dev, dtype=torch.float32)

    pos_strategy = str(cfg.supcon_pos_strategy).lower().strip()

    # balanced batch winner labels
    if pos_strategy == "top1":
        winners_np = np.argmax(util_off, axis=1).astype(np.int64)
    else:
        winners_np = _tie_break_winners_from_util(
            util_np=util_off,
            rng=rng,
            tie_eps=float(getattr(cfg, "offline_tie_eps", 1e-9)),
        )

    per_class = getattr(cfg, "balance_per_class", None)
    if per_class is None:
        per_class = max(1, int(cfg.supcon_bs // max(1, cfg.balance_min_classes)))

    print(
        f"[Offline] SupCon(B only): N_off={N}, bs={cfg.supcon_bs}, epochs={cfg.offline_epochs}, "
        f"pos_strategy={pos_strategy}, temp={cfg.supcon_temp}, "
        f"balanced_batch(C={cfg.balance_min_classes}, per_class={per_class})"
    )

    for ep in range(1, int(cfg.offline_epochs) + 1):
        b_idx_np = sample_winner_balanced_batch_indices(
            winners=winners_np,
            bs=int(cfg.supcon_bs),
            rng=rng,
            min_classes=int(cfg.balance_min_classes),
            per_class=int(per_class),
        )
        b_idx = torch.as_tensor(b_idx_np, device=router.dev, dtype=torch.long)

        xb = X_t[b_idx]  # (bs,d_ctx) float32
        ub = U_t[b_idx]  # (bs,n_models) float32

        router.train()
        if router.opt_b is None:
            raise RuntimeError("router.opt_b is None. unfreeze_B created optimizer?")

        router.opt_b.zero_grad(set_to_none=True)

        z = router.forward_ctx_supcon(xb)  # (bs,d_proj) float32

        if pos_strategy == "top1":
            y = ub.argmax(dim=1)
            pos_mask = (y.unsqueeze(1) == y.unsqueeze(0)).to(dtype=torch.float32)
            pos_mask.fill_diagonal_(0.0)

        elif pos_strategy == "topr_mass":
            pos_mask = build_pos_mask_adaptive_topk(
                util_batch=ub,
                n_models=n_models,
                max_k=int(cfg.supcon_topk_max_k),
                mode="mass",
                delta=float(cfg.supcon_topk_delta),
                beta=float(cfg.supcon_topk_beta),
                q=float(cfg.supcon_topk_q),
            )

        elif pos_strategy == "topr_margin":
            pos_mask = build_pos_mask_adaptive_topk(
                util_batch=ub,
                n_models=n_models,
                max_k=int(cfg.supcon_topk_max_k),
                mode="margin",
                delta=float(cfg.supcon_topk_delta),
                beta=float(cfg.supcon_topk_beta),
                q=float(cfg.supcon_topk_q),
            )
        else:
            raise ValueError(f"Unknown supcon_pos_strategy: {pos_strategy}")

        loss = supcon_loss_posmask(z, pos_mask, temperature=float(cfg.supcon_temp))
        loss.backward()

        if float(cfg.supcon_grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(router.B.parameters(), max_norm=float(cfg.supcon_grad_clip))

        router.opt_b.step()

        if (ep % 50 == 0) or (ep == 1) or (ep == int(cfg.offline_epochs)):
            with torch.no_grad():
                avg_pos = float((pos_mask > 0).to(dtype=torch.float32).sum(dim=1).mean().item())
            print(f"[Offline] epoch={ep:5d} | supcon_loss={float(loss.item()):.4f} | avg_pos={avg_pos:.1f}")

    router.freeze_B()
    router.eval()
