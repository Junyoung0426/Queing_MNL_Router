# mnl_router.py
from __future__ import annotations

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Router (query-only input) + model embedding table a_k.

    - Input at decision time: x_ctx only (no model idx appended)
    - Internally uses:
        z_ctx = B(x_ctx)                              (d_proj)
        a_k   = a_table[k]                            (d_proj)
        z_{k} = z_ctx ⊙ a_k                           (d_proj)
        u_k   = z_k^T theta                           (scalar, log-odds scale)

    Online:
      - theta only (full-history MLE + ridge via LBFGS)
      - V_inv Sherman–Morrison
    """

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

        b_type = str(b_type).lower().strip()
        if b_type not in ("linear", "mlp", "none"):
            raise ValueError(f"b_type must be one of ['linear','mlp','none'], got {b_type}")
        self.b_type = b_type

        # effective projection dim
        if self.b_type == "none":
            d_proj_eff = self.d_ctx
        else:
            d_proj_eff = int(d_proj)

        self.d_proj = int(d_proj_eff)
        self.d_final_feature = self.d_proj
        self.d = self.d_final_feature

        self.lambda_0 = float(lambda_0)
        self.supcon_temp = float(supcon_temp)
        self.lr_b_default = float(lr_b)

        init_dev = torch.device(device)

        # a_table: (n_models, d)
        A0 = torch.zeros((self.n_models, self.d_final_feature), device=init_dev, dtype=torch.float32)
        self.register_buffer("a_table", A0)
        self._ak_ready = False

        # B projection
        if self.b_type == "none":
            self.B = nn.Identity()
        elif self.b_type == "linear":
            self.B = nn.Linear(self.d_ctx, self.d_proj, bias=False)
        elif self.b_type == "mlp":
            d_hidden = self.d_proj * int(b_hidden_mult)
            self.B = nn.Sequential(
                nn.Linear(self.d_ctx, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, self.d_proj),
                nn.LayerNorm(self.d_proj),
            )
        else:
            self.B = nn.Identity()

        self.to(init_dev)

        # theta (float32) / V_inv (float64)
        self.theta = nn.Parameter(torch.zeros(self.d_final_feature, device=init_dev, dtype=torch.float32))
        self.register_buffer(
            "V_inv",
            (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=init_dev, dtype=torch.float64),
        )

        self.opt_b: Optional[torch.optim.Optimizer] = None

        self.theta_solver = str(theta_solver).lower().strip()
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = lbfgs_line_search

        # history buffer (stores per-step chosen-assortment features z_S and label y)
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)
        self._Z_hist: Optional[torch.Tensor] = None   # (cap, K, d)
        self._y_hist: Optional[torch.Tensor] = None   # (cap,)

        # default: freeze B
        self.freeze_B()

    @property
    def dev(self) -> torch.device:
        return self.theta.device

    @staticmethod
    def _clip_norm(x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        scale = norm.clamp_min(1.0)
        return x / scale

    # ---------- a_table ----------
    @torch.no_grad()
    def set_a_table(self, A: torch.Tensor, normalize: bool = True):
        A = A.to(self.a_table.device).float()
        if normalize:
            A = F.normalize(A, dim=-1)
        self.a_table.copy_(A)
        self._ak_ready = True

    # ---------- projection / feature construction ----------
    def B_projection(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (d_ctx,) or (B, d_ctx)
        return z_ctx: (d,) or (B, d)
        """
        x_ctx = x_ctx.to(self.dev)
        return self.B(x_ctx)

    def z_for_S(self, z_ctx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        z_ctx: (d,) or (B,d)
        S:     (K,) or (B,K)   (dtype long)
        return z_S: (K,d) or (B,K,d) where z_{t,k} = z_ctx ⊙ a_table[S_k]
        """
        if not self._ak_ready:
            raise RuntimeError("a_table not set. Call set_a_table(A) before routing.")

        S = S.to(self.dev).long().clamp(0, self.n_models - 1)
        a_S = self.a_table[S]  # (K,d) or (B,K,d)

        if z_ctx.dim() == 1:
            return a_S * z_ctx.unsqueeze(0)

        if a_S.dim() == 2:
            return z_ctx.unsqueeze(1) * a_S.unsqueeze(0)

        return z_ctx.unsqueeze(1) * a_S

    def z_for_S_from_ctx(self, x_ctx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        z_ctx = self.B_projection(x_ctx)
        return self.z_for_S(z_ctx, S)

    # ---------- scoring ----------
    def scores_for_S_from_z(self, z_S: torch.Tensor) -> torch.Tensor:
        z_S = z_S.to(self.dev)
        if z_S.dim() == 2:
            return z_S @ self.theta
        if z_S.dim() == 3:
            return torch.einsum("bkd,d->bk", z_S, self.theta)

    def logits_all_models(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        Fast path: compute u_k for all models without building (B,N,d) explicitly.
        For each k:
          u_k = (z_ctx ⊙ a_k)^T theta = a_k^T (z_ctx ⊙ theta)

        return:
          if x_ctx is (d_ctx,) -> (N,)
          if x_ctx is (B,d_ctx) -> (B,N)
        """
        z_ctx = self.B_projection(x_ctx)                  # (d) or (B,d)
        z_theta = z_ctx * self.theta                     # broadcast => (d) or (B,d)

        if z_theta.dim() == 1:
            return z_theta.unsqueeze(0) @ self.a_table.T  # (1,d)@(d,N)=(1,N)
        return z_theta @ self.a_table.T                   # (B,d)@(d,N)=(B,N)

    # ---------- TS noise ----------
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        d_feat = self.d_final_feature
        with torch.no_grad():
            V = 0.5 * (self.V_inv
                       + self.V_inv.T)  # float64
            eye = torch.eye(d_feat, device=self.dev, dtype=torch.float64)

            jitter = 1e-12
            L = None
            for _ in range(8):
                L_try, info = torch.linalg.cholesky_ex(V + jitter * eye)
                if int(info) == 0:
                    L = L_try
                    break
                jitter *= 10.0

            if L is None:
                e, v = torch.linalg.eigh(V + jitter * eye)
                e = e.clamp_min(jitter)
                V_spd = (v * e) @ v.T
                L = torch.linalg.cholesky(V_spd)

            u = torch.randn(d_feat, M, device=self.dev, dtype=torch.float64)
            noise = float(alpha_t) * (L @ u)  # float64
            return noise.to(dtype=torch.float32)

    def sample_optimistic_reward_from_zS(self, z_S_batch: torch.Tensor, noise_vectors: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = z_S_batch.to(self.dev)                               # (B,K,d)
            mean = torch.einsum("bkd,d->bk", z, self.theta)          # (B,K)
            unc = torch.einsum("bkd,dm->bkm", z, noise_vectors)      # (B,K,M)
            u_samp = mean.unsqueeze(-1) + unc                        # (B,K,M)
            u_optim = u_samp.max(dim=2).values                       # (B,K)
            lse = torch.logsumexp(u_optim, dim=1)                    # (B,)
            return torch.sigmoid(lse)                                # (B,)

    # ---------- SupCon (B only) ----------
    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        x_ctx = x_ctx.to(self.dev)
        return self.B(x_ctx)

    def unfreeze_B(self, lr_b: Optional[float] = None):
        params = list(self.B.parameters())
        if len(params) == 0:
            self.opt_b = None
            return
        for p in params:
            p.requires_grad = True
        lr_use = self.lr_b_default if lr_b is None else float(lr_b)
        self.opt_b = torch.optim.Adam(params, lr=lr_use)

    def freeze_B(self):
        params = list(self.B.parameters())
        for p in params:
            p.requires_grad = False
        self.opt_b = None

    # ---------- online reset ----------
    def reset_for_online(self):
        self.freeze_B()
        self.eval()
        with torch.no_grad():
            self.theta.zero_()
            self.V_inv.copy_(
                (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float64)
            )
        self._reset_history()

    def _reset_history(self):
        self._K = None
        self._T = 0
        self._Z_hist = None
        self._y_hist = None

    # ---------- online MLE (full-history) ----------
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
            return

        if int(K) != self._K:
            self._reset_history()
            self._K = int(K)
            self._Z_hist = torch.empty((self._cap, self._K, self.d_final_feature), device=self.dev, dtype=torch.float32)
            self._y_hist = torch.empty((self._cap,), device=self.dev, dtype=torch.long)
            self._T = 0

    def _objective_vectorized(self, Z_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Z_batch: (T, K, d)
        y_batch: (T,) with classes in {0..K}, where 0 means outside option
        """
        T = Z_batch.shape[0]
        logits = torch.einsum("tkd,d->tk", Z_batch, self.theta)                 # (T,K)
        zero_col = torch.zeros((T, 1), device=self.dev, dtype=logits.dtype)    # outside logit=0
        full_logits = torch.cat([zero_col, logits], dim=1)                     # (T,K+1)
        nll = F.cross_entropy(full_logits, y_batch, reduction="sum")
        ridge = 0.5 * float(self.lambda_0) * torch.sum(self.theta * self.theta)
        return nll + ridge

    def _solve_theta_minimize(self) -> float:
        if self._T == 0:
            return 0.0
        assert self._Z_hist is not None and self._y_hist is not None
        Z_batch = self._Z_hist[: self._T]
        y_batch = self._y_hist[: self._T]

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

    def update_from_zS(self, z_S: torch.Tensor, y_vec: torch.Tensor) -> Tuple[float, float]:
        """
        z_S: (K, d) - 선택된 Assortment의 특징 벡터들
        y_vec: (K+1,) - One-hot reward vector (0번 인덱스: Outside Option)
        """
        self.eval()
        z_S = z_S.to(self.dev).contiguous()
        y_vec = y_vec.to(self.dev).view(-1) # (K+1,)

        if z_S.dim() != 2:
            raise ValueError(f"z_S shape mismatch: expected (K,d), got {tuple(z_S.shape)}")

        K = z_S.shape[0]
        # (0: Outside, 1~K: Items)
        target_idx = torch.argmax(y_vec) 
        self._init_history_if_needed(K)
        self._ensure_capacity(self._T + 1)

        assert self._Z_hist is not None and self._y_hist is not None
        self._Z_hist[self._T].copy_(z_S)
        self._y_hist[self._T] = target_idx
        self._T += 1

        loss_total = self._solve_theta_minimize()

        # Sherman-Morrison Update 
        with torch.no_grad():
            for j in range(K):
                z = z_S[j].to(dtype=torch.float64)
                v = self.V_inv @ z
                denom = 1.0 + (z @ v)
                if (not torch.isfinite(denom)) or (denom <= 1e-12):
                    denom = torch.tensor(1e-12, device=self.dev, dtype=torch.float64)
                self.V_inv -= torch.outer(v, v) / denom
            self.V_inv.copy_(0.5 * (self.V_inv + self.V_inv.T))
            self.V_inv.diagonal().add_(1e-12)
        return float(loss_total), 0.0

    def update_from_ctx(self, x_ctx: Union[np.ndarray, torch.Tensor], S: List[int], y_vec: torch.Tensor) -> Tuple[float, float]:
        if isinstance(x_ctx, np.ndarray):
            x_ctx_t = torch.from_numpy(x_ctx).float().to(self.dev)
        else:
            x_ctx_t = x_ctx.to(self.dev).float()
        S_t = torch.tensor(S, device=self.dev, dtype=torch.long)
        z_S = self.z_for_S_from_ctx(x_ctx_t, S_t)  # (K,d)
        return self.update_from_zS(z_S, y_vec)
