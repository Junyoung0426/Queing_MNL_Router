# mnl_router.py
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNLRouter(nn.Module):

    def __init__(
        self,
        d_ctx: int,
        n_models: int,
        d_proj: int,
        lambda_0: float = 1.0,
        supcon_temp: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        b_type: str = "linear",
        b_hidden_mult: int = 2,
        theta_solver: str = "lbfgs",
        lbfgs_max_iter: int = 40,
        lbfgs_history_size: int = 20,
        lbfgs_line_search: str = "strong_wolfe",
        hist_init_capacity: int = 2048,

        lr_b: float = 1e-3,
    ):
        super().__init__()
        self.d_ctx = int(d_ctx)
        self.n_models = int(n_models)
        self.d_proj = int(d_proj)

        self.lambda_0 = float(lambda_0)
        self.supcon_temp = float(supcon_temp)

        self.d_final_feature = self.d_proj
        self.d = self.d_final_feature

        init_dev = torch.device(device)

        A0 = torch.zeros((self.n_models, self.d_proj), device=init_dev, dtype=torch.float32)
        self.register_buffer("a_table", A0)
        self._ak_ready = False

        b_type = str(b_type).lower().strip()
        self.b_type = b_type
        if b_type == "linear":
            self.B = nn.Linear(self.d_ctx, self.d_proj, bias=False)
        elif b_type == "mlp":
            d_hidden = self.d_proj * int(b_hidden_mult)
            self.B = nn.Sequential(
                nn.Linear(self.d_ctx, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, self.d_proj),

            )
        else:
            self.B = nn.Linear(self.d_ctx, self.d_proj, bias=False)

        self.to(init_dev)

        self.theta = nn.Parameter(torch.zeros(self.d_final_feature, device=init_dev))
        self.register_buffer(
            "V_inv",
            (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=init_dev),
        )

        self.opt_b: Optional[torch.optim.Optimizer] = None

        self.theta_solver = theta_solver
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = lbfgs_line_search

        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)
        self._Z_hist: Optional[torch.Tensor] = None
        self._y_hist: Optional[torch.Tensor] = None

        self.freeze_B()

    @property
    def dev(self) -> torch.device:
        return self.theta.device

    @property
    def device(self) -> torch.device:
        return self.theta.device

    @staticmethod
    def _clip_norm(x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        scale = norm.clamp_min(1.0)
        return x / scale

    def _split_input(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.dev)
        d_in = X.shape[-1]

        if d_in == self.d_ctx + 1:
            x_ctx = X[..., : self.d_ctx]
            m_raw = X[..., self.d_ctx]
            m_idx = torch.round(m_raw).long().clamp(0, self.n_models - 1)
            return x_ctx, m_idx

        if d_in == self.d_ctx + self.n_models:
            x_ctx = X[..., : self.d_ctx]
            onehot = X[..., self.d_ctx:]
            m_idx = onehot.argmax(dim=-1).long().clamp(0, self.n_models - 1)
            return x_ctx, m_idx

        x_ctx = X[..., : self.d_ctx]
        if d_in > self.d_ctx:
            m_raw = X[..., -1]
            m_idx = torch.round(m_raw).long().clamp(0, self.n_models - 1)
        else:
            shape = x_ctx.shape[:-1]
            m_idx = torch.zeros(shape, device=self.dev, dtype=torch.long)
        return x_ctx, m_idx

    @torch.no_grad()
    def set_a_table(self, A: torch.Tensor, normalize: bool = True):
        A = A.to(self.a_table.device).float()
        target_elems = self.n_models * self.d_proj
        flat = A.reshape(-1)
        if flat.numel() < target_elems:
            pad = torch.zeros(target_elems - flat.numel(), device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=0)
        elif flat.numel() > target_elems:
            flat = flat[:target_elems]
        A = flat.view(self.n_models, self.d_proj)
        if normalize:
            A = F.normalize(A, dim=-1)
        self.a_table.copy_(A)
        self._ak_ready = True

    def forward_proj(self, x_combined: torch.Tensor) -> torch.Tensor:
        if not self._ak_ready:
            pass

        x_ctx, m_idx = self._split_input(x_combined)
        z_ctx = self.B(x_ctx)
        z_model = self.a_table[m_idx]
        z_final = (z_ctx * z_model)
        return z_final

    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        x_ctx = x_ctx.to(self.dev)
        z = self.B(x_ctx)
        return z

    def unfreeze_B(self, lr_b: float = 1e-3):
        for p in self.B.parameters():
            p.requires_grad = True
        self.opt_b = torch.optim.Adam(self.B.parameters(), lr=float(lr_b))

    def freeze_B(self):
        for p in self.B.parameters():
            p.requires_grad = False
        self.opt_b = None

    def reset_for_online(self):
        self.freeze_B()
        self.eval()
        with torch.no_grad():
            self.theta.zero_()
            self.V_inv.copy_((1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.dev))
        self._reset_history()

    def _reset_history(self):
        self._K = None
        self._T = 0
        self._Z_hist = None
        self._y_hist = None

    def supcon_loss_posmask(self, features: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:

        Bsz = features.shape[0]
        if Bsz < 2:
            return torch.tensor(0.0, device=features.device)

        feats = F.normalize(features, dim=-1)
        pos_mask = pos_mask.to(features.device).float()

        mask_self = torch.eye(Bsz, device=features.device)
        pos_mask = pos_mask * (1.0 - mask_self)

        sim = torch.matmul(feats, feats.T) / float(self.supcon_temp)
        logits = sim - mask_self * 1e9
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_weight_sum = pos_mask.sum(dim=1).clamp_min(1e-12)
        loss_i = -(log_prob * pos_mask).sum(dim=1) / pos_weight_sum
        return loss_i.mean()

    @torch.no_grad()
    def build_pos_mask_adaptive_topk(
        self,
        util_batch: torch.Tensor,
        max_k: int = 4,
        mode: str = "mass",
        delta: float = 1.0,
        beta: float = 5.0,
        q: float = 0.90,
    ) -> torch.Tensor:

        util_batch = util_batch.to(self.dev)
        Bsz, K = util_batch.shape
        if K != self.n_models:
            if K > self.n_models:
                util_batch = util_batch[:, : self.n_models]
            else:
                pad = torch.zeros((Bsz, self.n_models - K), device=self.dev, dtype=util_batch.dtype)
                util_batch = torch.cat([util_batch, pad], dim=1)
            Bsz, K = util_batch.shape

        max_k = int(min(max_k, K))
        top_vals, top_idx = torch.topk(util_batch, k=max_k, dim=1)
        top1 = top_vals[:, [0]]

        mode = mode.lower().strip()

        if mode == "margin":
            keep = (top_vals >= (top1 - float(delta)))
            keep[:, 0] = True

            delta_eff = max(float(delta), 1e-6)
            w = ((top_vals - (top1 - delta_eff)) / delta_eff).clamp(0.0, 1.0)
            w[:, 0] = 1.0
            w = w * keep.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

            membership = torch.zeros((Bsz, K), device=self.dev, dtype=torch.float32)
            membership.scatter_(1, top_idx, w)

            pos_weight = membership @ membership.T
            pos_weight.fill_diagonal_(0.0)
            return pos_weight

        if mode == "mass":
            v = top_vals - top_vals.max(dim=1, keepdim=True).values
            p = torch.softmax(float(beta) * v, dim=1)

            c = torch.cumsum(p, dim=1)
            cond = (c >= float(q))
            any_true = cond.any(dim=1)
            first = cond.int().argmax(dim=1)
            first = torch.where(any_true, first, torch.full_like(first, max_k - 1))
            k_i = first + 1

            ar = torch.arange(max_k, device=self.dev).unsqueeze(0)
            keep = (ar < k_i.unsqueeze(1))

            w = p * keep.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

            membership = torch.zeros((Bsz, K), device=self.dev, dtype=torch.float32)
            membership.scatter_(1, top_idx, w)

            pos_weight = membership @ membership.T
            pos_weight.fill_diagonal_(0.0)
            return pos_weight

        mode = "mass"
        v = top_vals - top_vals.max(dim=1, keepdim=True).values
        p = torch.softmax(float(beta) * v, dim=1)

        c = torch.cumsum(p, dim=1)
        cond = (c >= float(q))
        any_true = cond.any(dim=1)
        first = cond.int().argmax(dim=1)
        first = torch.where(any_true, first, torch.full_like(first, max_k - 1))
        k_i = first + 1

        ar = torch.arange(max_k, device=self.dev).unsqueeze(0)
        keep = (ar < k_i.unsqueeze(1))

        w = p * keep.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

        membership = torch.zeros((Bsz, K), device=self.dev, dtype=torch.float32)
        membership.scatter_(1, top_idx, w)

        pos_weight = membership @ membership.T
        pos_weight.fill_diagonal_(0.0)
        return pos_weight

    def get_scores(self, X_S: torch.Tensor) -> torch.Tensor:
        z = self.forward_proj(X_S)
        if X_S.dim() == 2:
            return z @ self.theta
        if X_S.dim() == 3:
            return torch.einsum("bkd,d->bk", z, self.theta)
        X_flat = X_S.reshape(-1, X_S.shape[-1])
        z_flat = self.forward_proj(X_flat)
        return z_flat @ self.theta

    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        d_feat = self.d_final_feature
        with torch.no_grad():
            V = 0.5 * (self.V_inv + self.V_inv.T)
            L = torch.linalg.cholesky(V)
            u = torch.randn(d_feat, M, device=self.device)
            return float(alpha_t) * (L @ u)

    def sample_optimistic_reward_batch(self, X_S_batch: torch.Tensor, noise_vectors: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.forward_proj(X_S_batch)
            mean = torch.einsum("bkd,d->bk", z, self.theta)
            unc = torch.einsum("bkd,dm->bkm", z, noise_vectors)
            uopt = (mean.unsqueeze(-1) + unc).amax(dim=-1)
            lse = torch.logsumexp(uopt, dim=1)
            return torch.sigmoid(lse)

    def _ensure_capacity(self, need_T: int):
        if self._Z_hist is None or self._y_hist is None:
            return
        if need_T <= self._cap:
            return

        new_cap = self._cap
        while new_cap < need_T:
            new_cap *= 2

        Z_new = torch.empty((new_cap, self._K, self.d_final_feature), device=self.dev, dtype=self._Z_hist.dtype)
        y_new = torch.empty((new_cap,), device=self.dev, dtype=self._y_hist.dtype)

        if self._T > 0:
            Z_new[: self._T].copy_(self._Z_hist[: self._T])
            y_new[: self._T].copy_(self._y_hist[: self._T])

        self._Z_hist = Z_new
        self._y_hist = y_new
        self._cap = new_cap

    def _init_history_if_needed(self, K: int):
        if self._K is None:
            self._K = int(K)
            self._Z_hist = torch.empty((self._cap, self._K, self.d_final_feature), device=self.dev, dtype=torch.float32)
            self._y_hist = torch.empty((self._cap,), device=self.dev, dtype=torch.long)
            self._T = 0
        else:
            if int(K) != self._K:
                self._reset_history()
                self._K = int(K)
                self._Z_hist = torch.empty((self._cap, self._K, self.d_final_feature), device=self.dev, dtype=torch.float32)
                self._y_hist = torch.empty((self._cap,), device=self.dev, dtype=torch.long)
                self._T = 0

    def _objective_vectorized(self, Z_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        T = Z_batch.shape[0]
        logits = torch.einsum("tkd,d->tk", Z_batch, self.theta)
        zero_col = torch.zeros((T, 1), device=self.dev, dtype=logits.dtype)
        full_logits = torch.cat([zero_col, logits], dim=1)
        nll = F.cross_entropy(full_logits, y_batch, reduction="sum")
        ridge = 0.5 * float(self.lambda_0) * torch.sum(self.theta * self.theta)
        return nll + ridge

    def _solve_theta_minimize(self) -> float:
        if self._T == 0:
            return 0.0
        solver = str(self.theta_solver).lower()

        assert self._Z_hist is not None and self._y_hist is not None
        Z_batch = self._Z_hist[: self._T]
        y_batch = self._y_hist[: self._T]

        if solver == "lbfgs":
            opt = torch.optim.LBFGS(
                [self.theta],
                lr=0.05,
                max_iter=self.lbfgs_max_iter,
                history_size=self.lbfgs_history_size,
                line_search_fn=self.lbfgs_line_search,
            )

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = self._objective_vectorized(Z_batch, y_batch)
                loss.backward()
                return loss

            loss = opt.step(closure)
            return float(loss.item()) if hasattr(loss, "item") else float(loss)

        opt = torch.optim.SGD([self.theta], lr=0.01)
        for _ in range(min(100, self.lbfgs_max_iter)):
            opt.zero_grad(set_to_none=True)
            loss = self._objective_vectorized(Z_batch, y_batch)
            loss.backward()
            opt.step()
        return float(loss.item()) if hasattr(loss, "item") else float(loss)

    def update(self, X_S: torch.Tensor, y_vec: torch.Tensor) -> Tuple[float, float]:
        self.eval()
        X_S = X_S.to(self.dev)
        y_vec = y_vec.to(self.dev).view(-1)

        max_val, arg = torch.max(y_vec, dim=0)
        target_idx = torch.where(
            max_val > 0,
            arg.long() + 1,
            torch.zeros((), device=self.dev, dtype=torch.long),
        )

        with torch.no_grad():
            z_final = self.forward_proj(X_S).contiguous()

        K = z_final.shape[0]
        self._init_history_if_needed(K)
        self._ensure_capacity(self._T + 1)

        assert self._Z_hist is not None and self._y_hist is not None
        self._Z_hist[self._T].copy_(z_final)
        self._y_hist[self._T] = target_idx
        self._T += 1

        loss_total = self._solve_theta_minimize()

        with torch.no_grad():
            for j in range(z_final.shape[0]):
                z = z_final[j]
                v = self.V_inv @ z
                denom = 1.0 + (z @ v)
                self.V_inv -= torch.outer(v, v) / denom

        return float(loss_total), 0.0

def build_X_S_from_context_idx(x_ctx: np.ndarray, S: List[int], device: torch.device) -> torch.Tensor:

    M = len(S)
    x_ctx_tensor = torch.from_numpy(x_ctx).float().to(device)
    x_rep = x_ctx_tensor.unsqueeze(0).expand(M, -1)
    idx = torch.tensor(S, device=device, dtype=torch.float32).unsqueeze(1)
    return torch.cat([x_rep, idx], dim=1)
