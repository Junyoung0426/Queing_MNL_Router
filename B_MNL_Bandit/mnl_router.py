# mnl_router.py
from __future__ import annotations

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Online MNL Router using Sherman–Morrison updates.

      - theta: full-history MLE + ridge via L-BFGS (float32)
      - V_inv: approximate covariance inverse, updated via Sherman–Morrison (float32)
      - TS sampling: Cholesky of V_inv with jitter + eig fallback (float32)

    Notes:
      - V_inv may drift from SPD due to numerical errors. Optional SPD projection helps.
    """

    def __init__(
        self,
        d_ctx: int,
        n_models: int,
        d_proj: int,
        combine_mode: str,
        lambda_0: float = 1.0,
        supcon_temp: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        b_type: str = "linear",
        b_hidden_mult: int = 2,
        lbfgs_max_iter: int = 80,
        lbfgs_history_size: int = 50,
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

        if self.b_type == "none":
            d_proj_eff = self.d_ctx
        else:
            d_proj_eff = int(d_proj)

        self.d_proj = int(d_proj_eff)
        self.d_final_feature = int(d_proj_eff)
        self.d = self.d_final_feature

        cm = str(combine_mode).lower().strip()
        if cm not in ("mul", "add"):
            raise ValueError(f"combine_mode must be 'mul' or 'add', got {combine_mode}")
        self.combine_mode = cm

        self.lambda_0 = float(lambda_0)
        if self.lambda_0 <= 0.0:
            raise ValueError("lambda_0 must be > 0 for SPD initialization.")
        self.supcon_temp = float(supcon_temp)
        self.lr_b_default = float(lr_b)

        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = lbfgs_line_search


        init_dev = torch.device(device)

        # a_table: (n_models, d) float32
        A0 = torch.zeros((self.n_models, self.d_final_feature), device=init_dev, dtype=torch.float32)
        self.register_buffer("a_table", A0)
        self._ak_ready = False

        # B projection (float32)
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
        self.B = self.B.to(dtype=torch.float32)

        # theta: float32
        self.theta = nn.Parameter(torch.zeros(self.d_final_feature, device=init_dev, dtype=torch.float32))

        # V_inv: float32, initial V = lambda_0 I -> V_inv = (1/lambda_0) I
        V_inv_0 = (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=init_dev, dtype=torch.float32)
        self.register_buffer("V_inv", V_inv_0)

        # history buffers
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)
        self._Z_hist: Optional[torch.Tensor] = None  # (cap, K, d) float32
        self._y_hist: Optional[torch.Tensor] = None  # (cap,) long

        # diagnostics (TS 안정성 모니터링)
        self._ts_chol_fail_count: int = 0
        self._ts_last_jitter: float = 0.0

        self.opt_b: Optional[torch.optim.Optimizer] = None
        self.freeze_B()

    @property
    def dev(self) -> torch.device:
        return self.theta.device

    # --------- diagnostics getters ---------
    @property
    def ts_chol_fail_count(self) -> int:
        return int(self._ts_chol_fail_count)

    @property
    def ts_last_jitter(self) -> float:
        return float(self._ts_last_jitter)

    # ---------- a_table ----------
    @torch.no_grad()
    def set_a_table(self, A: torch.Tensor, normalize: bool = True):
        A = A.to(device=self.a_table.device, dtype=torch.float32)
        if normalize:
            A = F.normalize(A, dim=-1)
        self.a_table.copy_(A)
        self._ak_ready = True

    # ---------- projection / features ----------
    def B_projection(self, x_ctx: torch.Tensor) -> torch.Tensor:
        x_ctx = x_ctx.to(device=self.dev, dtype=torch.float32)
        return self.B(x_ctx)

    def z_for_S(self, z_ctx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        if not self._ak_ready:
            raise RuntimeError("a_table not set. Call set_a_table(A) before routing.")

        z_ctx = z_ctx.to(device=self.dev, dtype=torch.float32)
        S = S.to(self.dev).long().clamp(0, self.n_models - 1)
        a_S = self.a_table[S]  # float32

        if self.combine_mode == "mul":
            if z_ctx.dim() == 1:
                return a_S * z_ctx.unsqueeze(0)               # (K,d)
            if a_S.dim() == 2:
                return z_ctx.unsqueeze(1) * a_S.unsqueeze(0)  # (B,K,d)
            return z_ctx.unsqueeze(1) * a_S                   # (B,K,d)

        # add
        if z_ctx.dim() == 1:
            return a_S + z_ctx.unsqueeze(0)                   # (K,d)
        if a_S.dim() == 2:
            return z_ctx.unsqueeze(1) + a_S.unsqueeze(0)      # (B,K,d)
        return z_ctx.unsqueeze(1) + a_S                       # (B,K,d)

    def z_for_S_from_ctx(self, x_ctx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        z_ctx = self.B_projection(x_ctx)
        return self.z_for_S(z_ctx, S)

    # ---------- scoring ----------
    def scores_for_S_from_z(self, z_S: torch.Tensor) -> torch.Tensor:
        z_S = z_S.to(device=self.dev, dtype=torch.float32)
        if z_S.dim() == 2:
            return z_S @ self.theta
        if z_S.dim() == 3:
            return torch.einsum("bkd,d->bk", z_S, self.theta)
        raise ValueError(f"z_S dim must be 2 or 3, got {z_S.dim()}")

    def logits_all_models(self, x_ctx: torch.Tensor) -> torch.Tensor:
        z_ctx = self.B_projection(x_ctx)  # float32

        if self.combine_mode == "mul":
            z_theta = z_ctx * self.theta
            if z_theta.dim() == 1:
                return z_theta.unsqueeze(0) @ self.a_table.T  # (1,N)
            return z_theta @ self.a_table.T                   # (B,N)

        a_theta = (self.a_table @ self.theta)                 # (N,)
        if z_ctx.dim() == 1:
            base = torch.dot(z_ctx, self.theta)               # ()
            return base.unsqueeze(0) + a_theta.unsqueeze(0)   # (1,N)
        base = z_ctx @ self.theta                             # (B,)
        return base.unsqueeze(1) + a_theta.unsqueeze(0)       # (B,N)




    # ---------- TS noise ----------
    @torch.no_grad()
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        """
        noise ~ N(0, alpha^2 * V_inv).
        L L^T = V_inv, noise = alpha * (L @ u), u~N(0,I).
        """
        d_feat = int(self.d_final_feature)

        V = 0.5 * (self.V_inv + self.V_inv.T)
        eye = torch.eye(d_feat, device=self.dev, dtype=torch.float32)

        jitter = 1e-12
        L = None
        for _ in range(8):
            L_try, info = torch.linalg.cholesky_ex(V + jitter * eye)
            if int(info) == 0:
                L = L_try
                break
            jitter *= 10.0

        if L is None:
            # 기록 남김
            self._ts_chol_fail_count += 1
            self._ts_last_jitter = float(jitter)

            e, Q = torch.linalg.eigh(V + jitter * eye)
            e = e.clamp_min(jitter)
            V_spd = (Q * e) @ Q.T
            L = torch.linalg.cholesky(V_spd)
        else:
            self._ts_last_jitter = float(jitter)

        u = torch.randn(d_feat, M, device=self.dev, dtype=torch.float32)
        return float(alpha_t) * (L @ u)  # (d,M) float32

    @torch.no_grad()
    def sample_optimistic_reward_from_zS(self, z_S_batch: torch.Tensor, noise_vectors: torch.Tensor) -> torch.Tensor:
        z = z_S_batch.to(device=self.dev, dtype=torch.float32)       # (B,K,d)
        nv = noise_vectors.to(device=self.dev, dtype=torch.float32)  # (d,M)

        mean = torch.einsum("bkd,d->bk", z, self.theta)              # (B,K)
        unc = torch.einsum("bkd,dm->bkm", z, nv)                     # (B,K,M)
        u_samp = mean.unsqueeze(-1) + unc                            # (B,K,M)
        u_optim = u_samp.max(dim=2).values                           # (B,K)
        lse = torch.logsumexp(u_optim, dim=1)                        # (B,)
        return torch.sigmoid(lse)                                    # (B,) float32

    # ---------- SupCon (B only) ----------
    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        x_ctx = x_ctx.to(device=self.dev, dtype=torch.float32)
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
        for p in self.B.parameters():
            p.requires_grad = False
        self.opt_b = None

    # ---------- online reset ----------
    def reset_for_online(self):
        self.freeze_B()
        self.eval()
        with torch.no_grad():
            self.theta.zero_()
            self.V_inv.copy_(
                (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float32)
            )
        self._reset_history()
        self._ts_chol_fail_count = 0
        self._ts_last_jitter = 0.0

    def _reset_history(self):
        self._K = None
        self._T = 0
        self._Z_hist = None
        self._y_hist = None

    # ---------- history ----------
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

    # ---------- objective ----------
    def _objective_vectorized_ce(self, Z_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        T = int(Z_batch.shape[0])
        logits = torch.einsum("tkd,d->tk", Z_batch, self.theta)               # (T,K)
        zero_col = torch.zeros((T, 1), device=self.dev, dtype=torch.float32)  # outside logit=0
        full_logits = torch.cat([zero_col, logits], dim=1)                   # (T,K+1)
        nll = F.cross_entropy(full_logits, y_batch, reduction="sum")
        ridge = 0.5 * float(self.lambda_0) * torch.dot(self.theta, self.theta)
        return nll + ridge

    # ---------- solver ----------
    def _solve_theta_lbfgs(self, Z_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        opt = torch.optim.LBFGS(
            [self.theta],
            lr=0.5,
            max_iter=self.lbfgs_max_iter,
            history_size=self.lbfgs_history_size,
            line_search_fn=self.lbfgs_line_search,
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._objective_vectorized_ce(Z_batch, y_batch)
            loss.backward()
            return loss

        loss = opt.step(closure)
        return float(loss.item()) if hasattr(loss, "item") else float(loss)

    def _solve_theta_minimize(self) -> float:
        if self._T == 0:
            return 0.0
        assert self._Z_hist is not None and self._y_hist is not None
        Z_batch = self._Z_hist[: self._T]
        y_batch = self._y_hist[: self._T]
        return self._solve_theta_lbfgs(Z_batch, y_batch)

    # ---------- updates ----------
    def update_from_zS(self, z_S: torch.Tensor, y_vec: torch.Tensor) -> Tuple[float, float]:
        """
        z_S: (K, d) float32
        y_vec: (K+1,) one-hot, index 0 is outside option
        """
        self.eval()

        z_S = z_S.to(device=self.dev, dtype=torch.float32).contiguous()
        y_vec = y_vec.to(device=self.dev, dtype=torch.float32).view(-1)

        if z_S.dim() != 2:
            raise ValueError(f"z_S shape mismatch: expected (K,d), got {tuple(z_S.shape)}")

        K = int(z_S.shape[0])
        target_idx = torch.argmax(y_vec).long()

        self._init_history_if_needed(K)
        self._ensure_capacity(self._T + 1)

        assert self._Z_hist is not None and self._y_hist is not None
        self._Z_hist[self._T].copy_(z_S)
        self._y_hist[self._T] = target_idx
        self._T += 1

        # 1) theta update
        loss_total = self._solve_theta_minimize()

        # 2) V_inv update via Sherman–Morrison
        with torch.no_grad():
            for i in range(K):
                z = z_S[i]                    # (d,)
                v = self.V_inv @ z            # (d,)
                denom = 1.0 + (z @ v)         # scalar tensor

                denom = torch.nan_to_num(denom, nan=1e-12, posinf=1e12, neginf=1e-12)
                denom = denom.clamp_min(1e-12)

                self.V_inv -= torch.outer(v, v) / denom

            # symmetry projection (cheap)
            self.V_inv.copy_(0.5 * (self.V_inv + self.V_inv.T))

        return float(loss_total), 0.0

    def update_from_ctx(
        self,
        x_ctx: Union[np.ndarray, torch.Tensor],
        S: List[int],
        y_vec: torch.Tensor,
    ) -> Tuple[float, float]:
        if isinstance(x_ctx, np.ndarray):
            x_ctx_t = torch.from_numpy(x_ctx).to(device=self.dev, dtype=torch.float32)
        else:
            x_ctx_t = x_ctx.to(device=self.dev, dtype=torch.float32)

        S_t = torch.tensor(S, device=self.dev, dtype=torch.long)
        z_S = self.z_for_S_from_ctx(x_ctx_t, S_t)  # (K,d)
        return self.update_from_zS(z_S, y_vec)
