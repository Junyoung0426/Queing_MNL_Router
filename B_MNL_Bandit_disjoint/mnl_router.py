# B_MNL_Bandit_disjoint/mnl_router.py
from __future__ import annotations

from typing import List, Tuple, Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Online MNL Router (Disjoint per-model parameters)

    Offline:
      - B만 SupCon으로 pretrain (router.unfreeze_B -> offline_pretrain_B_supcon)
      - 완료 후 freeze_B()

    Online:
      - B는 고정(eval), theta만 active-set sparse update (LBFGS)
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
        lbfgs_max_iter: int = 20,
        lbfgs_history_size: int = 50,
        lbfgs_line_search: str = "strong_wolfe",
        hist_init_capacity: int = 2048,
        lr_b: float = 1e-3,
        # robustness
        dropout_rate: float = 0.2,
        noise_level: float = 0.05,
    ):
        super().__init__()

        self.d_ctx = int(d_ctx)
        self.n_models = int(n_models)

        # robustness params
        self.dropout_rate = float(dropout_rate)
        # noise_level을 "노이즈 벡터 L2 norm 스케일"로 해석한다
        self.noise_level = float(noise_level)

        # -----------------------
        # Projection Layer (B)
        # -----------------------
        b_type = str(b_type).lower().strip()
        if b_type not in ("linear", "mlp", "none"):
            raise ValueError(f"b_type must be one of ['linear','mlp','none'], got {b_type}")
        self.b_type = b_type

        if self.b_type == "none":
            self.B = nn.Identity()
            d_eff = self.d_ctx

        elif self.b_type == "linear":
            self.B = nn.Linear(self.d_ctx, int(d_proj), bias=False)
            d_eff = int(d_proj)

        elif self.b_type == "mlp":
            d_proj = int(self.d_ctx)
            d_hidden = int(d_proj) * int(b_hidden_mult)

            class _ResMLP(nn.Module):
                def __init__(self, d_in: int, d_hidden: int, p: float):
                    super().__init__()
                    self.fc1 = nn.Linear(d_in, d_hidden)
                    self.fc2 = nn.Linear(d_hidden, d_in)
                    self.act = nn.GELU()
                    self.drop = nn.Dropout(p=p)

                def forward(self, x):
                    y = self.fc1(x)
                    y = self.act(y)
                    y = self.drop(y)
                    y = self.fc2(y)
                    return x + y

            self.B = _ResMLP(int(self.d_ctx), d_hidden, self.dropout_rate)
            d_eff = int(d_proj)


        else:
            self.B = nn.Identity()
            d_eff = self.d_ctx

        self.d_proj = int(d_eff)
        self.d_final_feature = int(d_eff)
        self.d = self.d_final_feature

        # Hyperparameters
        self.lambda_0 = float(lambda_0)
        self.supcon_temp = float(supcon_temp)
        self.lr_b_default = float(lr_b)

        # L-BFGS config
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = str(lbfgs_line_search)

        init_dev = torch.device(device)
        self.to(init_dev)
        self.B = self.B.to(dtype=torch.float32)

        # -----------------------
        # Disjoint Params (theta_j per model)
        # -----------------------
        self.theta_list = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.d_final_feature, device=init_dev, dtype=torch.float32))
                for _ in range(self.n_models)
            ]
        )

        # V_inv: (N,d,d)
        V0 = (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=init_dev, dtype=torch.float32)
        self.register_buffer("V_inv", V0.unsqueeze(0).repeat(self.n_models, 1, 1))

        # -----------------------
        # History Storage
        # -----------------------
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)

        self._Z_hist = torch.empty((self._cap, self.d_final_feature), device=init_dev, dtype=torch.float32)
        self._S_hist: Optional[torch.Tensor] = None  # (cap, K)
        self._y_hist = torch.empty((self._cap,), device=init_dev, dtype=torch.long)

        # TS diagnostics
        self._ts_chol_fail_count: int = 0
        self._ts_last_jitter: float = 0.0

        # Optimizer for B
        self.opt_b: Optional[torch.optim.Optimizer] = None
        self._B_frozen: bool = False
        self.freeze_B()

    @property
    def dev(self) -> torch.device:
        return self.theta_list[0].device

    @property
    def theta(self) -> torch.Tensor:
        return torch.stack(list(self.theta_list), dim=0)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_B_frozen", False):
            self.B.eval()
        return self

    # -----------------------
    # Projection
    # -----------------------
    def B_projection(self, x_ctx: torch.Tensor) -> torch.Tensor:
        x_ctx = x_ctx.to(device=self.dev, dtype=torch.float32)
        z = self.B(x_ctx)

        if self.training and (not getattr(self, "_B_frozen", True)) and (self.dropout_rate > 0.0):
            z = F.dropout(z, p=self.dropout_rate, training=True)

        if self.training and (not getattr(self, "_B_frozen", True)) and (self.noise_level > 0.0):
            d = max(1, int(z.shape[-1]))
            noise_std = float(self.noise_level) / math.sqrt(float(d))
            z = z + torch.randn_like(z) * noise_std
        return z

    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        return self.B_projection(x_ctx)

    # -----------------------
    # Scoring helpers
    # -----------------------
    def scores_for_S_from_ctx(self, x_ctx: Union[np.ndarray, torch.Tensor], S: Union[List[int], torch.Tensor]):
        if isinstance(x_ctx, np.ndarray):
            x_t = torch.from_numpy(x_ctx).to(device=self.dev, dtype=torch.float32)
        else:
            x_t = x_ctx.to(device=self.dev, dtype=torch.float32)

        z = self.B_projection(x_t)
        S_t = torch.as_tensor(S, device=self.dev, dtype=torch.long).clamp(0, self.n_models - 1)

        Theta = self.theta

        if z.dim() == 1:
            th = Theta[S_t]
            return (th * z.unsqueeze(0)).sum(dim=-1)

        if S_t.dim() == 1:
            S_t = S_t.unsqueeze(0).expand(z.shape[0], -1)
        th = Theta[S_t]
        return (th * z.unsqueeze(1)).sum(dim=-1)

    def logits_all_models(self, x_ctx: torch.Tensor) -> torch.Tensor:
        z = self.B_projection(x_ctx.to(device=self.dev, dtype=torch.float32))
        Theta = self.theta
        if z.dim() == 1:
            return z.unsqueeze(0) @ Theta.T
        return z @ Theta.T

    # -----------------------
    # Thompson Sampling
    # -----------------------
    @torch.no_grad()
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        V = 0.5 * (self.V_inv + self.V_inv.transpose(-2, -1))
        eye = torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float32).unsqueeze(0)

        jitter = 1e-12
        L = None

        for _ in range(8):
            L_try, info_try = torch.linalg.cholesky_ex(V + jitter * eye)
            if int((info_try != 0).sum().item()) == 0:
                L = L_try
                break
            jitter *= 10.0

        if L is None:
            V_fix = (V + jitter * eye).clone()
            bad = torch.linalg.cholesky_ex(V_fix)[1] != 0
            self._ts_chol_fail_count += int(bad.sum().item())
            self._ts_last_jitter = float(jitter)

            bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
            for j in bad_idx:
                e, Q = torch.linalg.eigh(V_fix[j])
                e = e.clamp_min(jitter)
                V_spd = (Q * e) @ Q.T
                V_fix[j] = 0.5 * (V_spd + V_spd.T)
            L = torch.linalg.cholesky(V_fix)
        else:
            self._ts_last_jitter = float(jitter)

        u = torch.randn(self.n_models, self.d_final_feature, M, device=self.dev, dtype=torch.float32)
        return float(alpha_t) * (L @ u)

    @torch.no_grad()
    def sample_optimistic_reward_from_zS(self, z_ctx_batch, S_batch, noise_vectors):
        z = z_ctx_batch.to(device=self.dev, dtype=torch.float32)
        S = S_batch.to(device=self.dev, dtype=torch.long).clamp(0, self.n_models - 1)
        nv = noise_vectors.to(device=self.dev, dtype=torch.float32)

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if S.dim() == 1:
            S = S.unsqueeze(0)

        theta_opt = self.theta.unsqueeze(-1) + nv  # (N,d,M)
        theta_opt_S = theta_opt[S]                 # (B,K,d,M)

        u_samp = (z.unsqueeze(1).unsqueeze(-1) * theta_opt_S).sum(dim=2)  # (B,K,M)
        u_optim = u_samp.max(dim=2).values
        lse = torch.logsumexp(u_optim, dim=1)
        return torch.sigmoid(lse)

    # -----------------------
    # History management
    # -----------------------
    def _ensure_capacity(self, need_T: int):
        if need_T <= self._cap:
            return
        new_cap = self._cap
        while new_cap < need_T:
            new_cap *= 2

        Z_new = torch.empty((new_cap, self.d_final_feature), device=self.dev, dtype=self._Z_hist.dtype)
        y_new = torch.empty((new_cap,), device=self.dev, dtype=self._y_hist.dtype)

        if self._T > 0:
            Z_new[: self._T].copy_(self._Z_hist[: self._T])
            y_new[: self._T].copy_(self._y_hist[: self._T])

        self._Z_hist = Z_new
        self._y_hist = y_new

        if self._S_hist is not None:
            K = int(self._S_hist.shape[1])
            S_new = torch.empty((new_cap, K), device=self.dev, dtype=self._S_hist.dtype)
            if self._T > 0:
                S_new[: self._T].copy_(self._S_hist[: self._T])
            self._S_hist = S_new

        self._cap = new_cap

    def _init_history_if_needed(self, K: int):
        if self._K is None:
            self._K = int(K)
            self._S_hist = torch.empty((self._cap, self._K), device=self.dev, dtype=torch.long)
            self._T = 0
            return
        if int(K) != self._K:
            self._reset_history()
            self._K = int(K)
            self._S_hist = torch.empty((self._cap, self._K), device=self.dev, dtype=torch.long)
            self._T = 0

    def _reset_history(self):
        self._K = None
        self._T = 0
        self._S_hist = None

    # -----------------------
    # Theta optimization (active only, full history)
    # -----------------------
    def _solve_theta_full_history_active_only(self, active_indices: List[int]) -> float:
        T = int(self._T)
        if T == 0 or len(active_indices) == 0:
            return 0.0
        assert self._S_hist is not None

        active_indices = sorted(set(int(i) for i in active_indices))
        active_set = set(active_indices)

        active_tensor = torch.tensor(active_indices, device=self.dev, dtype=torch.long)
        is_relevant = torch.isin(self._S_hist[:T], active_tensor)
        mask = is_relevant.any(dim=1)
        if not mask.any():
            return 0.0

        Z_sub = self._Z_hist[:T][mask]
        S_sub = self._S_hist[:T][mask]
        y_sub = self._y_hist[:T][mask]
        T_sub = Z_sub.shape[0]

        targets = [self.theta_list[i] for i in active_indices]

        opt = torch.optim.LBFGS(
            targets,
            lr=1.0,
            max_iter=self.lbfgs_max_iter,
            history_size=self.lbfgs_history_size,
            line_search_fn=self.lbfgs_line_search,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            opt.zero_grad(set_to_none=True)

            theta_mix = []
            for j in range(self.n_models):
                pj = self.theta_list[j]
                theta_mix.append(pj if j in active_set else pj.detach())
            Theta_mat = torch.stack(theta_mix, dim=0)  # (N,d)

            theta_S = Theta_mat[S_sub]  # (T_sub,K,d)
            logits = (theta_S * Z_sub.unsqueeze(1)).sum(dim=-1)  # (T_sub,K)

            zero_col = torch.zeros((T_sub, 1), device=self.dev, dtype=torch.float32)
            full_logits = torch.cat([zero_col, logits], dim=1)  # (T_sub,K+1)

            nll = F.cross_entropy(full_logits, y_sub, reduction="sum")
            ridge = 0.5 * float(self.lambda_0) * sum((p ** 2).sum() for p in targets)

            loss = nll + ridge
            loss.backward()
            return loss

        loss = opt.step(closure)
        return float(loss.item()) if hasattr(loss, "item") else float(loss)

    # -----------------------
    # Online update interface
    # -----------------------
    def update_from_zS(self, z_S_or_z: torch.Tensor, S: Union[List[int], torch.Tensor], y_vec: torch.Tensor) -> Tuple[float, float]:
        self.train()
        self.B.eval()  # online에서 B는 고정

        if z_S_or_z.dim() == 1:
            z = z_S_or_z
        elif z_S_or_z.dim() == 2:
            z = z_S_or_z[0]
        else:
            raise ValueError(f"z shape error: {z_S_or_z.shape}")

        if isinstance(S, list):
            S_t = torch.tensor(S, device=self.dev, dtype=torch.long)
        else:
            S_t = S.to(device=self.dev, dtype=torch.long)
        S_t = S_t.clamp(0, self.n_models - 1)

        K = int(S_t.numel())
        y_idx = torch.argmax(y_vec.to(device=self.dev)).long()

        z = z.to(device=self.dev, dtype=torch.float32).view(-1)
        

        # append history
        self._init_history_if_needed(K)
        self._ensure_capacity(self._T + 1)

        self._Z_hist[self._T].copy_(z)
        self._S_hist[self._T].copy_(S_t)
        self._y_hist[self._T] = y_idx
        self._T += 1

        current_active = torch.unique(S_t).tolist()

        # covariance update (active only)
        with torch.no_grad():
            for idx in current_active:
                V = self.V_inv[idx]
                v = V @ z
                denom = (1.0 + (z @ v)).clamp_min(1e-12)
                self.V_inv[idx] = V - torch.outer(v, v) / denom
                self.V_inv[idx].copy_(0.5 * (self.V_inv[idx] + self.V_inv[idx].T))

        loss_total = self._solve_theta_full_history_active_only(current_active)
        return float(loss_total), 0.0

    def update_from_ctx(self, x_ctx, S, y_vec):
        if isinstance(x_ctx, np.ndarray):
            x_t = torch.from_numpy(x_ctx).to(device=self.dev, dtype=torch.float32)
        else:
            x_t = x_ctx.to(device=self.dev, dtype=torch.float32)

        with torch.no_grad():
            z = self.B_projection(x_t).view(-1)
        return self.update_from_zS(z, S, y_vec)

    # -----------------------
    # Freeze/unfreeze
    # -----------------------
    def freeze_B(self):
        for p in self.B.parameters():
            p.requires_grad_(False)
        self.opt_b = None
        self._B_frozen = True
        self.B.eval()

    def unfreeze_B(self, lr_b=None):
        params = list(self.B.parameters())
        if len(params) == 0:
            return
        for p in params:
            p.requires_grad_(True)

        self._B_frozen = False
        lr_use = self.lr_b_default if lr_b is None else float(lr_b)
        self.opt_b = torch.optim.Adam(params, lr=lr_use)
        self.B.train()

    def reset_for_online(self):
        self.freeze_B()
        self.eval()
        with torch.no_grad():
            for p in self.theta_list:
                p.zero_()
            V0 = (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float32)
            self.V_inv.copy_(V0.unsqueeze(0).repeat(self.n_models, 1, 1))

        self._reset_history()
        self._ts_chol_fail_count = 0
        self._ts_last_jitter = 0.0
