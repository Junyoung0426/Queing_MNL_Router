from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from queue_config import QueueConfig


def _proj(router, x: torch.Tensor) -> torch.Tensor:
    if hasattr(router, "B_projection"):
        return router.B_projection(x)
    if hasattr(router, "forward_ctx_supcon"):
        return router.forward_ctx_supcon(x)
    return router.B(x)


def _get_cfg(cfg: QueueConfig | None, name: str, default):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _util_profile(u: torch.Tensor, mean_center: bool) -> torch.Tensor:
    if mean_center:
        u = u - u.mean(dim=1, keepdim=True)
    return F.normalize(u, dim=1)


def supcon_loss_posmask_with_denmask(
    z: torch.Tensor,
    pos_w: torch.Tensor,
    den_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    B = int(z.shape[0])
    if B < 2:
        return z.sum() * 0.0

    eye = torch.eye(B, device=z.device, dtype=torch.bool)

    den = den_mask.to(device=z.device, dtype=torch.bool).masked_fill(eye, False)
    pw = pos_w.to(device=z.device, dtype=z.dtype).masked_fill(eye, 0.0)

    sim = (z @ z.T) / float(temperature)
    sim = sim.masked_fill(~den, -1e9)

    pos_sum = pw.sum(dim=1)
    valid = pos_sum > 0
    if not bool(valid.any().item()):
        return z.sum() * 0.0

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss_i = -(log_prob * pw).sum(dim=1) / pos_sum.clamp_min(1e-12)
    return loss_i[valid].mean()


def offline_pretrain_B_supcon(
    router,
    X_ctx_off,
    util_off=None,
    n_models=None,
    cfg: QueueConfig | None = None,
):
    if str(_get_cfg(cfg, "b_type", "")).lower().strip() == "none":
        print("[Offline] skipped (b_type=none)")
        return

    if util_off is None:
        print("[Offline] skipped (util_off required)")
        return

    X_np = np.asarray(X_ctx_off, dtype=np.float32)
    U_np = np.asarray(util_off, dtype=np.float32)
    N = int(X_np.shape[0])
    if N < 2:
        print("[Offline] skipped (N_off<2)")
        return

    epochs = int(_get_cfg(cfg, "offline_epochs", 0))
    if epochs <= 0:
        print("[Offline] skipped (offline_epochs<=0)")
        return

    lr = float(_get_cfg(cfg, "offline_lr_B", 1e-3))
    router.unfreeze_B(lr_b=lr)

    wd = float(_get_cfg(cfg, "supcon_weight_decay", 0.0))
    if wd > 0.0:
        params = [p for p in router.B.parameters() if p.requires_grad]
        if len(params) == 0:
            print("[Offline] skipped (no trainable B params)")
            return
        router.opt_b = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    if getattr(router, "opt_b", None) is None:
        print("[Offline] skipped (no B optimizer)")
        return

    X_t = torch.from_numpy(X_np).to(device=router.dev, dtype=torch.float32)
    U_t = torch.from_numpy(U_np).to(device=router.dev, dtype=torch.float32)

    bs = int(_get_cfg(cfg, "supcon_bs", 256))
    bs = min(bs, N)
    temp = float(_get_cfg(cfg, "supcon_temp", 0.07))
    grad_clip = float(_get_cfg(cfg, "supcon_grad_clip", 0.0))

    tau_pos = float(_get_cfg(cfg, "supcon_uc_tau_pos", 0.9))
    tau_neg = float(_get_cfg(cfg, "supcon_uc_tau_neg", 0.1))
    mean_center = bool(_get_cfg(cfg, "supcon_uc_mean_center", True))
    neg_cap = int(_get_cfg(cfg, "supcon_uc_neg_cap", 64))
    require_neg = bool(_get_cfg(cfg, "supcon_uc_require_neg", True))

    loader = DataLoader(
        TensorDataset(X_t, U_t),
        batch_size=bs,
        shuffle=True,
        drop_last=False,
    )

    print(
        f"[Offline] utilcos_singlepos_top1_negTopK "
        f"epochs={epochs} bs={bs} temp={temp} tau_pos={tau_pos} tau_neg={tau_neg} neg_cap={neg_cap}"
    )

    router.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        steps = 0
        valid_rows = 0
        total_rows = 0
        avg_neg = 0.0

        for xb, ub in loader:
            router.opt_b.zero_grad(set_to_none=True)

            z = _proj(router, xb)

            uprof = _util_profile(ub, mean_center=mean_center)
            c = uprof @ uprof.T
            B = int(c.shape[0])
            eye = torch.eye(B, device=c.device, dtype=torch.bool)

            pos_cand = (c > tau_pos) & (~eye)
            neg_cand = (c < tau_neg) & (~eye)

            if neg_cap > 0:
                neg_mask = torch.zeros((B, B), device=c.device, dtype=torch.bool)
                neg_score = (-c).masked_fill(~neg_cand, -1e9)
                k = min(int(neg_cap), B - 1)
                if k > 0:
                    vals, idx = torch.topk(neg_score, k=k, dim=1, largest=True)
                    keep = vals > -1e8
                    neg_mask.scatter_(1, idx, keep)
            else:
                neg_mask = neg_cand

            pos_w = torch.zeros((B, B), device=c.device, dtype=torch.float32)
            den_mask = torch.zeros((B, B), device=c.device, dtype=torch.bool)

            for i in range(B):
                pidx = torch.where(pos_cand[i])[0]
                if pidx.numel() == 0:
                    continue

                nidx = torch.where(neg_mask[i])[0]
                if require_neg and nidx.numel() == 0:
                    continue

                j = pidx[torch.argmax(c[i, pidx])]
                pos_w[i, j] = 1.0
                den_mask[i, j] = True

                if nidx.numel() > 0:
                    nidx = nidx[nidx != j]
                    if nidx.numel() > 0:
                        den_mask[i, nidx] = True

            loss = supcon_loss_posmask_with_denmask(z, pos_w, den_mask, temperature=temp)
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(router.B.parameters(), max_norm=float(grad_clip))
            router.opt_b.step()

            tot += float(loss.item())
            steps += 1

            with torch.no_grad():
                row_has_pos = (pos_w.sum(dim=1) > 0)
                valid_rows += int(row_has_pos.sum().item())
                total_rows += int(B)
                avg_neg += float(neg_mask.sum(dim=1).float().mean().item())

        ok_ratio = float(valid_rows) / float(max(1, total_rows))
        print(
            f"[Offline] epoch={ep:3d} loss={tot/max(1,steps):.4f} ok_ratio={ok_ratio:.3f} avg_neg={avg_neg/max(1,steps):.2f}"
        )

    router.freeze_B()
    router.eval()
