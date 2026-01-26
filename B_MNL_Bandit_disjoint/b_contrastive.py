from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue_config import QueueConfig


def _proj(router, x):
    if hasattr(router, "B_projection"):
        return router.B_projection(x)
    if hasattr(router, "forward_ctx_supcon"):
        return router.forward_ctx_supcon(x)
    return router.B(x)


def _get_cfg(cfg, name, default):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


class SemanticMultiPositiveSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, pos_topk: int = 8, pos_sim_thresh: float = 0.6):
        super().__init__()
        self.temperature = float(temperature)
        self.pos_topk = int(pos_topk)
        self.pos_sim_thresh = float(pos_sim_thresh)

    def forward(self, z: torch.Tensor, raw_x: torch.Tensor) -> torch.Tensor:
        device = z.device
        B = int(z.shape[0])
        if B < 2:
            return z.new_tensor(0.0)

        z = F.normalize(z, dim=1)
        x = F.normalize(raw_x, dim=1)

        sim_raw = x @ x.T
        eye = torch.eye(B, device=device, dtype=torch.bool)
        sim_raw = sim_raw.masked_fill(eye, -1e9)

        k = min(self.pos_topk, B - 1)
        if k <= 0:
            return z.new_tensor(0.0)

        topv, topi = sim_raw.topk(k=k, dim=1)
        mask = topv >= self.pos_sim_thresh
        if mask.numel() > 0:
            mask[:, 0] = True

        w_pos = torch.where(mask, topv.clamp_min(0.0), torch.zeros_like(topv))
        pos_w = torch.zeros((B, B), device=device, dtype=z.dtype)
        pos_w.scatter_(1, topi, w_pos.to(dtype=z.dtype))
        pos_w = pos_w.masked_fill(eye, 0.0)

        sim_z = (z @ z.T) / max(self.temperature, 1e-6)
        sim_z = sim_z.masked_fill(eye, -1e9)
        log_prob = sim_z - torch.logsumexp(sim_z, dim=1, keepdim=True)

        denom = pos_w.sum(dim=1)
        valid = denom > 0
        if not bool(valid.any().item()):
            return z.new_tensor(0.0)

        loss_i = -(log_prob * pos_w).sum(dim=1) / denom.clamp_min(1e-12)
        return loss_i[valid].mean()


def _simhash_labels(X: np.ndarray, seed: int, n_bits: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    n_bits = int(max(1, min(int(n_bits), 30)))
    rng = np.random.RandomState(int(seed))
    R = rng.normal(size=(D, n_bits)).astype(np.float32)
    bits = ((X @ R) > 0).astype(np.int64)
    shifts = (1 << np.arange(n_bits, dtype=np.int64))
    return (bits @ shifts).astype(np.int64)


def sample_balanced_batch_indices(labels: np.ndarray, bs: int, rng: np.random.RandomState, min_classes: int, per_class: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    N = int(labels.shape[0])
    bs = int(bs)

    present = np.unique(labels)
    if present.size == 0:
        return rng.choice(N, size=bs, replace=(N < bs)).astype(np.int64)

    C = min(int(min_classes), int(present.size))
    chosen = rng.choice(present, size=C, replace=False)

    take = int(per_class)
    if take * C > bs:
        take = max(1, bs // C)

    idx_list = []
    for c in chosen:
        pool = np.where(labels == c)[0]
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


def offline_pretrain_B_supcon(router, X_ctx_off, util_off=None, n_models=None, cfg: QueueConfig | None = None):
    if _get_cfg(cfg, "b_type", "").lower().strip() == "none":
        print("[Offline] skipped (b_type=none)")
        return

    X_np = np.asarray(X_ctx_off, dtype=np.float32)
    N = int(X_np.shape[0])
    if N < 2:
        print("[Offline] skipped (N_off<2)")
        return

    epochs = int(_get_cfg(cfg, "offline_epochs", 0))
    if epochs <= 0:
        print("[Offline] skipped (offline_epochs<=0)")
        return

    router.unfreeze_B(lr_b=float(_get_cfg(cfg, "offline_lr_B", 1e-3)))
    wd = float(_get_cfg(cfg, "supcon_weight_decay", 0.0))
    if wd > 0:
        router.opt_b = torch.optim.AdamW(router.B.parameters(), lr=float(_get_cfg(cfg, "offline_lr_B", 1e-3)), weight_decay=wd)

    if getattr(router, "opt_b", None) is None:
        print("[Offline] skipped (no B params)")
        return

    seed = int(_get_cfg(cfg, "seed", 0))
    rng = np.random.RandomState(seed + 2024)

    X_t = torch.from_numpy(X_np).to(device=router.dev, dtype=torch.float32)

    bs = int(_get_cfg(cfg, "supcon_bs", 256))
    bs = min(bs, N)
    temp = float(_get_cfg(cfg, "supcon_temp", 0.07))
    grad_clip = float(_get_cfg(cfg, "supcon_grad_clip", 0.0))

    sem_topk = int(_get_cfg(cfg, "supcon_sem_topk", 8))
    sem_thresh = float(_get_cfg(cfg, "supcon_sem_thresh", 0.6))
    hash_bits = int(_get_cfg(cfg, "supcon_sem_hash_bits", 16))

    labels_np = _simhash_labels(X_np, seed=seed + 9917, n_bits=hash_bits)

    min_classes = int(_get_cfg(cfg, "balance_min_classes", 8))
    per_class = int(_get_cfg(cfg, "balance_per_class", 0)) or max(1, bs // max(1, min_classes))

    loss_fn = SemanticMultiPositiveSupConLoss(temperature=temp, pos_topk=sem_topk, pos_sim_thresh=sem_thresh)

    print(f"[Offline] semantic-simhash epochs={epochs} bs={bs} temp={temp} topk={sem_topk} thresh={sem_thresh} bits={hash_bits}")

    router.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        steps = 0
        avg_pos = 0.0

        n_batches = max(1, int(np.ceil(N / bs)))
        for _ in range(n_batches):
            idx_np = sample_balanced_batch_indices(labels_np, bs=bs, rng=rng, min_classes=min_classes, per_class=per_class)
            idx = torch.as_tensor(idx_np, device=router.dev, dtype=torch.long)
            xb = X_t[idx]

            router.opt_b.zero_grad(set_to_none=True)
            z = _proj(router, xb)
            loss = loss_fn(z, xb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(router.B.parameters(), max_norm=grad_clip)
            router.opt_b.step()

            tot += float(loss.item())
            steps += 1
            with torch.no_grad():
                x = F.normalize(xb, dim=1)
                sim_raw = x @ x.T
                sim_raw = sim_raw.masked_fill(torch.eye(sim_raw.shape[0], device=sim_raw.device, dtype=torch.bool), -1e9)
                k = min(sem_topk, sim_raw.shape[0] - 1)
                if k > 0:
                    vals, _ = sim_raw.topk(k=k, dim=1)
                    pos_cnt = (vals >= sem_thresh).sum(dim=1)
                    pos_cnt = torch.maximum(pos_cnt, torch.ones_like(pos_cnt))
                    avg_pos += float(pos_cnt.float().mean().item())

        print(f"[Offline] epoch={ep:3d} loss={tot/max(1,steps):.4f} avg_pos={avg_pos/max(1,steps):.1f}")

    router.freeze_B()
    router.eval()
