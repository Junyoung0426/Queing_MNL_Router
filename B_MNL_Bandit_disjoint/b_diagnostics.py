# b_diagnostics.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from queue_config import QueueConfig
from b_contrastive import (
    supcon_loss_posmask,
    sample_winner_balanced_batch_indices,
    build_pos_mask_adaptive_topk,
    build_soft_pos_mask_from_util,
    _row_norm,  # b_contrastive의 row_norm을 재사용해도 되고, 여기서 따로 정의해도 된다
)


@torch.no_grad()
def make_pos_mask(ub: torch.Tensor, n_models: int, cfg: QueueConfig) -> torch.Tensor:
    """
    b_contrastive.offline_pretrain_B_supcon과 동일 규칙
    """
    pos = str(cfg.supcon_pos_strategy).lower().strip()

    if pos == "top1":
        y = ub.argmax(dim=1)
        pm = (y.unsqueeze(1) == y.unsqueeze(0)).to(dtype=ub.dtype)
        pm.fill_diagonal_(0.0)
        pm = _row_norm(pm)

    elif pos == "topr_mass":
        pm = build_pos_mask_adaptive_topk(
            util_batch=ub,
            n_models=n_models,
            max_k=int(cfg.supcon_topk_max_k),
            mode="mass",
            delta=float(cfg.supcon_topk_delta),
            beta=float(cfg.supcon_topk_beta),
            q=float(cfg.supcon_topk_q),
        ).to(dtype=ub.dtype)
        pm = _row_norm(pm)

    elif pos == "topr_margin":
        pm = build_pos_mask_adaptive_topk(
            util_batch=ub,
            n_models=n_models,
            max_k=int(cfg.supcon_topk_max_k),
            mode="margin",
            delta=float(cfg.supcon_topk_delta),
            beta=float(cfg.supcon_topk_beta),
            q=float(cfg.supcon_topk_q),
        ).to(dtype=ub.dtype)
        pm = _row_norm(pm)

    else:
        raise ValueError(f"Unknown supcon_pos_strategy: {cfg.supcon_pos_strategy}")

    use_soft = bool(getattr(cfg, "supcon_soft_targets", False))
    soft_mix = float(getattr(cfg, "supcon_soft_mix", 0.0))
    if use_soft and soft_mix > 0.0:
        tau_u = float(getattr(cfg, "supcon_soft_tau_u", 0.05))
        topm = int(getattr(cfg, "supcon_soft_topm", 64))
        score_mode = str(getattr(cfg, "supcon_soft_score_mode", "top1"))

        soft_mask = build_soft_pos_mask_from_util(
            util_batch=ub,
            tau_u=tau_u,
            topm=topm,
            score_mode=score_mode,
        ).to(dtype=ub.dtype)

        pm = (1.0 - soft_mix) * pm + soft_mix * soft_mask
        pm.fill_diagonal_(0.0)
        pm = _row_norm(pm)

    return pm


@torch.no_grad()
def eval_supcon_loss(
    router,
    X_np: np.ndarray,
    U_np: np.ndarray,
    cfg: QueueConfig,
    n_models: int,
    n_batches: int = 5,
    seed: int = 0,
) -> float:
    router.eval()
    X = torch.from_numpy(np.asarray(X_np)).to(router.dev, dtype=torch.float32)
    U = torch.from_numpy(np.asarray(U_np)).to(router.dev, dtype=torch.float32)

    N = int(X.shape[0])
    bs = min(int(cfg.supcon_bs), N)

    rng = np.random.RandomState(seed)
    losses = []

    lambda_sem = float(getattr(cfg, "supcon_sem_reg", 0.0))

    for _ in range(int(n_batches)):
        idx = rng.choice(N, size=bs, replace=(N < bs))
        xb = X[idx]
        ub = U[idx]

        z = router.forward_ctx_supcon(xb)
        pm = make_pos_mask(ub, n_models=n_models, cfg=cfg)

        loss_main = supcon_loss_posmask(z, pm, temperature=float(cfg.supcon_temp))

        loss_sem = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        if lambda_sem > 0.0 and hasattr(router, "x_residual_proj"):
            x_ref = router.x_residual_proj(xb).detach()
            loss_sem = 1.0 - F.cosine_similarity(
                F.normalize(z, dim=-1),
                F.normalize(x_ref, dim=-1),
                dim=-1
            ).mean()

        losses.append(float((loss_main + lambda_sem * loss_sem).item()))

    return float(np.mean(losses)) if losses else 0.0


def _b_params_for_clip(router) -> list:
    params = []
    if hasattr(router, "B"):
        params += [p for p in router.B.parameters() if p.requires_grad]
    if getattr(router, "alpha_logit", None) is not None and router.alpha_logit.requires_grad:
        params.append(router.alpha_logit)
    # x_adapter는 기본 frozen이라 clip 대상에서 제외된다
    return params


def offline_pretrain_B_supcon_with_val(
    router,
    X_tr: np.ndarray, U_tr: np.ndarray,
    X_va: Optional[np.ndarray], U_va: Optional[np.ndarray],
    cfg: QueueConfig,
    n_models: int,
    log_every: int = 50,
):
    N = int(X_tr.shape[0])
    if int(cfg.offline_epochs) <= 0 or N < 2:
        print("[Offline] skipped")
        return

    if getattr(router, "b_type", "").lower().strip() == "none":
        print("[Offline] skipped (b_type=none)")
        return

    router.unfreeze_B(lr_b=float(cfg.offline_lr_B))
    if router.opt_b is None:
        raise RuntimeError("router.opt_b is None after unfreeze_B")

    rng = np.random.RandomState(int(cfg.seed) + 2024)

    X_t = torch.from_numpy(np.asarray(X_tr)).to(router.dev, dtype=torch.float32)
    U_t = torch.from_numpy(np.asarray(U_tr)).to(router.dev, dtype=torch.float32)

    winners_np = np.argmax(U_tr, axis=1).astype(np.int64)
    per_class = max(1, int(cfg.supcon_bs // max(1, cfg.balance_min_classes)))

    lambda_sem = float(getattr(cfg, "supcon_sem_reg", 0.0))
    use_soft = bool(getattr(cfg, "supcon_soft_targets", False))
    soft_mix = float(getattr(cfg, "supcon_soft_mix", 0.0))

    has_val = (X_va is not None) and (U_va is not None) and (len(X_va) >= 2)

    print(
        f"[Offline] Start B-Pretrain | epochs={int(cfg.offline_epochs)} "
        f"| sem_reg={lambda_sem} | soft={use_soft}(mix={soft_mix})"
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
        xb, ub = X_t[b_idx], U_t[b_idx]

        router.train()
        router.opt_b.zero_grad(set_to_none=True)

        z = router.forward_ctx_supcon(xb)
        pm = make_pos_mask(ub, n_models=n_models, cfg=cfg)

        loss_supcon = supcon_loss_posmask(z, pm, temperature=float(cfg.supcon_temp))

        loss_sem = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        if lambda_sem > 0.0 and hasattr(router, "x_residual_proj"):
            x_ref = router.x_residual_proj(xb).detach()
            loss_sem = 1.0 - F.cosine_similarity(
                F.normalize(z, dim=-1),
                F.normalize(x_ref, dim=-1),
                dim=-1
            ).mean()

        loss = loss_supcon + lambda_sem * loss_sem
        loss.backward()

        if float(cfg.supcon_grad_clip) > 0:
            params = _b_params_for_clip(router)
            if len(params) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.supcon_grad_clip))

        router.opt_b.step()

        if (ep % int(log_every) == 0) or (ep == 1) or (ep == int(cfg.offline_epochs)):
            with torch.no_grad():
                alpha_val = None
                if getattr(router, "alpha_logit", None) is not None:
                    alpha_val = float(torch.sigmoid(router.alpha_logit).item())
                dens = float(pm.mean().item())

                ltr = eval_supcon_loss(router, X_tr, U_tr, cfg, n_models, seed=ep)
                lva_str = ""
                if has_val:
                    lva = eval_supcon_loss(router, X_va, U_va, cfg, n_models, seed=ep + 999)
                    lva_str = f" | val_loss={lva:.4f}"

            print(
                f"[Ep {ep:4d}] L_tot={float(loss.item()):.4f} "
                f"(Sup={float(loss_supcon.item()):.4f}, Sem={float(loss_sem.item()):.4f}) "
                f"| alpha={alpha_val} | tr_loss={ltr:.4f}{lva_str} | dens={dens:.4f}"
            )

    router.freeze_B()
    router.eval()


# -----------------------------
# collapse report
# -----------------------------
@torch.no_grad()
def effective_rank(Zn: torch.Tensor) -> float:
    Zc = Zn - Zn.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / max(1, Zn.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eig / eig.sum()
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())


def sample_pairs_cos(Z: np.ndarray, max_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    N = Z.shape[0]
    if N < 2:
        return np.zeros((0,), dtype=np.float32)
    P = min(int(max_pairs), N * (N - 1) // 2)
    i = rng.randint(0, N, size=P)
    j = rng.randint(0, N, size=P)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.sum(Z[i] * Z[j], axis=1).astype(np.float32)


@torch.no_grad()
def get_Z(router, X_np: np.ndarray, n_vis: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = X_np.shape[0]
    idx = rng.choice(N, size=min(int(n_vis), N), replace=False)
    X = torch.from_numpy(np.asarray(X_np[idx])).to(router.dev, dtype=torch.float32)
    Z = router.forward_ctx_supcon(X)
    return Z.detach().cpu().numpy(), idx


@torch.no_grad()
def collapse_report(router, X_np: np.ndarray, U_np: Optional[np.ndarray], n_vis: int, max_pairs: int, seed: int):
    router.eval()
    Z, idx = get_Z(router, X_np, n_vis=n_vis, seed=seed)
    cos = sample_pairs_cos(Z, max_pairs=max_pairs, seed=seed)
    Zt = torch.from_numpy(Z).to(router.dev, dtype=torch.float32)

    rep = {
        "N_vis": int(Z.shape[0]),
        "collapse_std_mean": float(Z.std(axis=0).mean()),
        "cos_mean_offdiag": float(cos.mean()) if cos.size else float("nan"),
        "effective_rank": float(effective_rank(Zt)),
    }

    labels = None
    if U_np is not None:
        labels = np.argmax(U_np[idx], axis=1).astype(np.int64)
    return rep, Z, labels


# -----------------------------
# plots (PCA 제거)
# -----------------------------
def plot_tsne(Z: np.ndarray, labels: Optional[np.ndarray], title: str, tsne_n: int = 2000, perplexity: int = 30, seed: int = 0):
    rng = np.random.RandomState(seed)
    N = Z.shape[0]
    take = min(int(tsne_n), N)
    idx = rng.choice(N, size=take, replace=False)
    Zs = Z[idx]
    ys = labels[idx] if labels is not None else None

    Z2 = TSNE(
        n_components=2,
        perplexity=int(perplexity),
        init="random",
        random_state=int(seed),
        learning_rate="auto",
    ).fit_transform(Zs)

    plt.figure()
    if ys is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=8, alpha=0.6)
    else:
        sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=ys, s=8, alpha=0.7, cmap="tab10")
        plt.colorbar(sc, label="winner id")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
