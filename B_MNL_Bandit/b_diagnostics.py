#b_diagnostics.py
from __future__ import annotations

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from queue_config import QueueConfig

from b_contrastive import (
    build_pos_mask_adaptive_topk, 
    supcon_loss_posmask, 
    sample_winner_balanced_batch_indices
)


@torch.no_grad()
def make_pos_mask(ub: torch.Tensor, n_models: int, cfg: QueueConfig) -> torch.Tensor:
    pos = str(cfg.supcon_pos_strategy).lower().strip()

    if pos == "top1":
        y = ub.argmax(dim=1)
        pm = (y.unsqueeze(1) == y.unsqueeze(0)).float()
        pm.fill_diagonal_(0.0)
        return pm

    # Uses imported function from b_contrastive
    if pos == "topr_mass":
        return build_pos_mask_adaptive_topk(
            util_batch=ub, n_models=n_models,
            max_k=int(cfg.supcon_topk_max_k),
            mode="mass",
            delta=float(cfg.supcon_topk_delta),
            beta=float(cfg.supcon_topk_beta),
            q=float(cfg.supcon_topk_q),
        )

    # Uses imported function from b_contrastive
    if pos == "topr_margin":
        return build_pos_mask_adaptive_topk(
            util_batch=ub, n_models=n_models,
            max_k=int(cfg.supcon_topk_max_k),
            mode="margin",
            delta=float(cfg.supcon_topk_delta),
            beta=float(cfg.supcon_topk_beta),
            q=float(cfg.supcon_topk_q),
        )

    raise ValueError(f"Unknown supcon_pos_strategy: {cfg.supcon_pos_strategy}")


@torch.no_grad()
def eval_supcon_loss(router, X_np: np.ndarray, U_np: np.ndarray, cfg: QueueConfig, n_models: int, n_batches: int = 5, seed: int = 0) -> float:
    router.eval()
    X = torch.from_numpy(X_np).float().to(router.dev)
    U = torch.from_numpy(U_np).float().to(router.dev)

    N = X.shape[0]
    bs = min(int(cfg.supcon_bs), N)
    replace = (N < bs)

    rng = np.random.RandomState(seed)
    losses = []
    for _ in range(n_batches):
        idx = rng.choice(N, size=bs, replace=replace)
        xb = X[idx]
        ub = U[idx]
        z = router.forward_ctx_supcon(xb)
        pm = make_pos_mask(ub, n_models=n_models, cfg=cfg)
        
        # Uses imported function from b_contrastive
        loss = supcon_loss_posmask(z, pm, temperature=float(cfg.supcon_temp))
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def offline_pretrain_B_supcon_with_val(
    router,
    X_tr: np.ndarray,
    U_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    U_va: Optional[np.ndarray],
    cfg: QueueConfig,
    n_models: int,
    log_every: int = 50,
    pos_density_warn: float = 0.25,
):
    N = X_tr.shape[0]
    if int(cfg.offline_epochs) <= 0 or N < 2:
        print("[Offline] skipped")
        return

    router.unfreeze_B(lr_b=float(cfg.offline_lr_B))
    rng = np.random.RandomState(int(cfg.seed) + 2024)

    X_t = torch.from_numpy(X_tr).float().to(router.dev)
    U_t = torch.from_numpy(U_tr).float().to(router.dev)

    winners_np = np.argmax(U_tr, axis=1).astype(np.int64)

    has_val = (X_va is not None) and (U_va is not None) and (len(X_va) >= 2)

    per_class = cfg.balance_per_class
    if per_class is None:
        per_class = max(1, int(cfg.supcon_bs // max(1, cfg.balance_min_classes)))

    print(f"[Offline] SupCon(B) tr={len(X_tr)} val={len(X_va) if has_val else 0} bs={cfg.supcon_bs} epochs={cfg.offline_epochs} pos={cfg.supcon_pos_strategy}")

    for ep in range(1, int(cfg.offline_epochs) + 1):
        # Uses imported function from b_contrastive
        b_idx_np = sample_winner_balanced_batch_indices(
            winners=winners_np,
            bs=int(cfg.supcon_bs),
            rng=rng,
            min_classes=int(cfg.balance_min_classes),
            per_class=int(per_class),
        )
        b_idx = torch.as_tensor(b_idx_np, device=router.dev, dtype=torch.long)

        xb = X_t[b_idx]
        ub = U_t[b_idx]

        router.train()
        router.opt_b.zero_grad(set_to_none=True)

        z = router.forward_ctx_supcon(xb)
        pm = make_pos_mask(ub, n_models=n_models, cfg=cfg)
        
        # Uses imported function from b_contrastive
        loss_tr = supcon_loss_posmask(z, pm, temperature=float(cfg.supcon_temp))
        loss_tr.backward()

        if float(cfg.supcon_grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(router.B.parameters(), max_norm=float(cfg.supcon_grad_clip))

        router.opt_b.step()

        if (ep % log_every == 0) or (ep == 1) or (ep == int(cfg.offline_epochs)):
            with torch.no_grad():
                density = float(pm.float().mean().item())
                row_pos = pm.float().sum(dim=1)
                row_stat = (float(row_pos.mean().item()), float(row_pos.min().item()), float(row_pos.max().item()))
                uniq, cnt = np.unique(winners_np[b_idx_np], return_counts=True)
                top_classes = sorted(list(zip(uniq.tolist(), cnt.tolist())), key=lambda x: -x[1])[:5]

            ltr = eval_supcon_loss(router, X_tr, U_tr, cfg, n_models=n_models, n_batches=5, seed=ep)
            if has_val:
                lva = eval_supcon_loss(router, X_va, U_va, cfg, n_models=n_models, n_batches=5, seed=ep + 999)
                print(f"[ep={ep:5d}] batch={loss_tr.item():.4f} tr_eval={ltr:.4f} val_eval={lva:.4f} density={density:.3f} row_pos={row_stat} top_w={top_classes}")
            else:
                print(f"[ep={ep:5d}] batch={loss_tr.item():.4f} tr_eval={ltr:.4f} density={density:.3f} row_pos={row_stat} top_w={top_classes}")

            if density > float(pos_density_warn):
                print(f"[Warn] pos_density={density:.3f} > {pos_density_warn:.2f} 과밀")

    router.freeze_B()
    router.eval()


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
        return np.zeros((0,), dtype=np.float64)
    P = min(int(max_pairs), N * (N - 1) // 2)
    i = rng.randint(0, N, size=P)
    j = rng.randint(0, N, size=P)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.sum(Z[i] * Z[j], axis=1).astype(np.float64)


@torch.no_grad()
def get_Z(router, X_np: np.ndarray, n_vis: int, seed: int):
    rng = np.random.RandomState(seed)
    N = X_np.shape[0]
    idx = rng.choice(N, size=min(int(n_vis), N), replace=False)
    X = torch.from_numpy(X_np[idx]).float().to(router.dev)
    Z = F.normalize(router.B(X), dim=-1)
    return Z.detach().cpu().numpy(), idx


@torch.no_grad()
def collapse_report(router, X_np: np.ndarray, U_np: Optional[np.ndarray], n_vis: int, max_pairs: int, seed: int):
    router.eval()
    Z, idx = get_Z(router, X_np, n_vis=n_vis, seed=seed)
    cos = sample_pairs_cos(Z, max_pairs=max_pairs, seed=seed)

    Zt = torch.from_numpy(Z).float().to(router.dev)

    rep = {
        "N_vis": int(Z.shape[0]),
        "d_proj": int(Z.shape[1]),
        "collapse_std_mean": float(Z.std(axis=0).mean()),
        "cos_mean_offdiag": float(cos.mean()) if cos.size else float("nan"),
        "cos_p50": float(np.percentile(cos, 50)) if cos.size else float("nan"),
        "cos_p95": float(np.percentile(cos, 95)) if cos.size else float("nan"),
        "effective_rank": float(effective_rank(Zt)),
    }

    labels = None
    if U_np is not None:
        labels = np.argmax(U_np[idx], axis=1).astype(np.int64)

    return rep, Z, labels


def plot_pca(Z: np.ndarray, labels: Optional[np.ndarray], title: str):
    Z2 = PCA(n_components=2, random_state=0).fit_transform(Z)
    plt.figure()
    if labels is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=8, alpha=0.6)
    else:
        sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=8, alpha=0.7, cmap="tab10")
        plt.colorbar(sc, label="winner id")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tsne(Z: np.ndarray, labels: Optional[np.ndarray], title: str, tsne_n: int, perplexity: int):
    rng = np.random.RandomState(0)
    N = Z.shape[0]
    take = min(int(tsne_n), N)
    idx = rng.choice(N, size=take, replace=False)
    Zs = Z[idx]
    ys = labels[idx] if labels is not None else None

    Z2 = TSNE(
        n_components=2,
        perplexity=int(perplexity),
        learning_rate="auto",
        init="pca",
        random_state=0,
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


def plot_cos_hist(Z: np.ndarray, title: str, max_pairs: int):
    cos = sample_pairs_cos(Z, max_pairs=max_pairs, seed=0)
    plt.figure()
    plt.hist(cos, bins=60)
    plt.title(title)
    plt.xlabel("cosine(z_i, z_j)")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_singular_values(Z: np.ndarray, title: str):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Zc, full_matrices=False)
    plt.figure()
    plt.plot(s)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("singular value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def probe_separability(Z: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, float]:
    uniq = np.unique(y)
    if uniq.size < 2:
        return {"knn_acc": np.nan, "linear_acc": np.nan}

    Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=0.3, random_state=seed, stratify=y)

    try:
        knn = KNeighborsClassifier(n_neighbors=20, metric="cosine")
        knn.fit(Xtr, ytr)
        pred_knn = knn.predict(Xte)
        acc_knn = accuracy_score(yte, pred_knn)
    except Exception:
        knn = KNeighborsClassifier(n_neighbors=20, metric="minkowski", p=2)
        knn.fit(Xtr, ytr)
        pred_knn = knn.predict(Xte)
        acc_knn = accuracy_score(yte, pred_knn)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(Xtr, ytr)
    pred_lr = lr.predict(Xte)
    acc_lr = accuracy_score(yte, pred_lr)

    return {"knn_acc": float(acc_knn), "linear_acc": float(acc_lr)}


def plot_pca_3d(Z: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "PCA 3D", seed: int = 0):
    Z3 = PCA(n_components=3, random_state=seed).fit_transform(Z)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if labels is None:
        ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=8, alpha=0.6)
    else:
        sc = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=labels, s=8, alpha=0.7, cmap="tab10")
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="winner id")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()


def plot_tsne_3d(Z: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "t-SNE 3D",
                 seed: int = 0, tsne_n: int = 1500, perplexity: int = 30):
    rng = np.random.RandomState(seed)
    N = Z.shape[0]
    n = min(int(tsne_n), N)
    idx = rng.choice(N, size=n, replace=False)

    Zs = Z[idx]
    ys = labels[idx] if labels is not None else None

    Z3 = TSNE(
        n_components=3,
        perplexity=int(perplexity),
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(Zs)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if ys is None:
        ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=8, alpha=0.6)
    else:
        sc = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=ys, s=8, alpha=0.7, cmap="tab10")
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="winner id")

    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    plt.tight_layout()
    plt.show()