# B_MNL_Bandit_disjoint/mnl_router.py
from __future__ import annotations

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Online MNL Router (Disjoint per-model parameters)
    
    [Final Policy: Full History Objective + Active-Set Sparse Update]
      1. Storage: 매 스텝 데이터(Context, Assortment, Choice)를 히스토리에 저장.
      2. V_inv Update: Sherman-Morrison을 사용해 Active Model에 대해서만 공분산 갱신.
      3. Theta Update: 
         - 전체 히스토리 중 '이번 Active 모델들이 등장했던' 데이터만 필터링(Masking).
         - 해당 데이터에 대해 L-BFGS를 수행하되, Active Parameter만 업데이트하고 나머지는 detach.
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
        normalize_z: bool = False,
    ):
        super().__init__()

        self.d_ctx = int(d_ctx)
        self.n_models = int(n_models)

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
            self.B = nn.Linear(self.d_ctx, d_proj, bias=False)
            d_eff = int(d_proj)
        elif self.b_type == "mlp":
            d_hidden = d_proj * int(b_hidden_mult)
            self.B = nn.Sequential(
                nn.Linear(self.d_ctx, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_proj),
                nn.LayerNorm(d_proj),
            )
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
        self.normalize_z = bool(normalize_z)
        
        # L-BFGS Config
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

        # V_inv: (N,d,d) - Thompson Sampling용 공분산 역행렬
        V0 = (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=init_dev, dtype=torch.float32)
        self.register_buffer("V_inv", V0.unsqueeze(0).repeat(self.n_models, 1, 1))

        # -----------------------
        # History Storage
        # -----------------------
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)

        self._Z_hist = torch.empty((self._cap, self.d_final_feature), device=init_dev, dtype=torch.float32)
        self._S_hist: Optional[torch.Tensor] = None  # Will be (cap, K)
        self._y_hist = torch.empty((self._cap,), device=init_dev, dtype=torch.long)

        # TS Diagnostics
        self._ts_chol_fail_count: int = 0
        self._ts_last_jitter: float = 0.0

        # Optimizer for B (Representation Learning) - Optional
        self.opt_b: Optional[torch.optim.Optimizer] = None
        self._B_frozen: bool = False
        self.freeze_B()

    # -----------------------
    # Properties & Helpers
    # -----------------------
    @property
    def dev(self) -> torch.device:
        return self.theta_list[0].device
        
    @property
    def theta(self) -> torch.Tensor:
        """Read-only stacked theta (N, d)"""
        return torch.stack(list(self.theta_list), dim=0)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_B_frozen", False):
            self.B.eval()
        return self

    # -----------------------
    # Projection & Scoring
    # -----------------------
    def B_projection(self, x_ctx: torch.Tensor) -> torch.Tensor:
        x_ctx = x_ctx.to(device=self.dev, dtype=torch.float32)
        z = self.B(x_ctx)
        if self.normalize_z:
            z = F.normalize(z, dim=-1)
        return z

    def scores_for_S_from_ctx(self, x_ctx: Union[np.ndarray, torch.Tensor], S: Union[List[int], torch.Tensor]):
        if isinstance(x_ctx, np.ndarray):
            x_t = torch.from_numpy(x_ctx).to(device=self.dev, dtype=torch.float32)
        else:
            x_t = x_ctx.to(device=self.dev, dtype=torch.float32)

        z = self.B_projection(x_t)  # (d)
        S_t = torch.as_tensor(S, device=self.dev, dtype=torch.long).clamp(0, self.n_models - 1)

        Theta = self.theta  # (N,d)

        if z.dim() == 1:
            th = Theta[S_t]  # (K,d)
            return (th * z.unsqueeze(0)).sum(dim=-1)  # (K,)

        # Batch case handling
        if S_t.dim() == 1:
            S_t = S_t.unsqueeze(0).expand(z.shape[0], -1)
        th = Theta[S_t]
        return (th * z.unsqueeze(1)).sum(dim=-1)

    def logits_all_models(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        Helper for queue_env (debugging/logging).
        Returns raw scores for ALL models: (Batch, N)
        """
        z = self.B_projection(x_ctx.to(device=self.dev, dtype=torch.float32))
        Theta = self.theta  # (N,d)
        if z.dim() == 1:
            return z.unsqueeze(0) @ Theta.T
        return z @ Theta.T

    # -----------------------
    # Thompson Sampling Logic
    # -----------------------
    @torch.no_grad()
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        """Sample noise from N(0, alpha^2 * V^-1)"""
        V = 0.5 * (self.V_inv + self.V_inv.transpose(-2, -1))
        eye = torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float32).unsqueeze(0)
        
        jitter = 1e-12
        L = None
        
        # Try Cholesky with increasing jitter
        for _ in range(8):
            L_try, info_try = torch.linalg.cholesky_ex(V + jitter * eye)
            if int((info_try != 0).sum().item()) == 0:
                L = L_try
                break
            jitter *= 10.0
        
        # Fallback to Eigendecomposition if Cholesky fails completely
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
        """Compute optimistic reward (sigmoid of optimistic score)"""
        z = z_ctx_batch.to(device=self.dev, dtype=torch.float32)
        S = S_batch.to(device=self.dev, dtype=torch.long).clamp(0, self.n_models - 1)
        nv = noise_vectors.to(device=self.dev, dtype=torch.float32)

        if z.dim() == 1: z = z.unsqueeze(0)
        if S.dim() == 1: S = S.unsqueeze(0)

        theta_opt = self.theta.unsqueeze(-1) + nv  # (N,d,M)
        theta_opt_S = theta_opt[S]                 # (B,K,d,M)

        u_samp = (z.unsqueeze(1).unsqueeze(-1) * theta_opt_S).sum(dim=2) # (B,K,M)
        u_optim = u_samp.max(dim=2).values         # (B,K)
        lse = torch.logsumexp(u_optim, dim=1)
        return torch.sigmoid(lse)

    # -----------------------
    # History Management
    # -----------------------
    def _ensure_capacity(self, need_T: int):
        if need_T <= self._cap: return
        new_cap = self._cap
        while new_cap < need_T: new_cap *= 2
        
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
            if self._T > 0: S_new[: self._T].copy_(self._S_hist[: self._T])
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
    # Optimization Core (Active-Only Full History)
    # -----------------------
    def _solve_theta_full_history_active_only(self, active_indices: List[int]) -> float:
        """
        [Optimization Core]
        Full-History L-BFGS를 수행하되,
        1) Masking: Active Model이 포함된 과거 데이터만 필터링하여 연산 속도 최적화.
        2) Partial Update: Active Parameter만 Optimizer에 등록하고 나머지는 detach하여 고정.
        """
        T = int(self._T)
        if T == 0 or len(active_indices) == 0:
            return 0.0
        assert self._S_hist is not None

        # 1. Clean Active Set
        active_indices = sorted(set(int(i) for i in active_indices))
        active_set = set(active_indices)
        
        # 2. Relevant History Masking
        # 전체 히스토리 중, 이번에 업데이트할 모델(active_indices)이 
        # Assortment(S)에 하나라도 포함되어 있는 행(row)만 추출
        active_tensor = torch.tensor(active_indices, device=self.dev, dtype=torch.long)
        
        # is_in: (T, K) -> True if element is in active_set
        is_relevant = torch.isin(self._S_hist[:T], active_tensor)
        mask = is_relevant.any(dim=1) # (T,) -> True if row contains any active model
        
        if not mask.any():
            return 0.0

        # 데이터 슬라이싱 (Sub-sampled Data)
        Z_sub = self._Z_hist[:T][mask]  # (T_sub, d)
        S_sub = self._S_hist[:T][mask]  # (T_sub, K)
        y_sub = self._y_hist[:T][mask]  # (T_sub,) 
        T_sub = Z_sub.shape[0]

        # 3. Setup Optimizer for Active Params Only
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

            # 4. Construct Theta Matrix (Active=Grad, Inactive=Const)
            theta_mix = []
            for j in range(self.n_models):
                pj = self.theta_list[j]
                theta_mix.append(pj if j in active_set else pj.detach())
            Theta_mat = torch.stack(theta_mix, dim=0)  # (N, d)

            # 5. Compute Logits on Sub-sampled History
            # S_sub: (T_sub, K)
            theta_S = Theta_mat[S_sub]  # (T_sub, K, d)
            
            # Logits: (T_sub, K)
            logits = (theta_S * Z_sub.unsqueeze(1)).sum(dim=-1)

            # Add Outside Option (Score=0, Index=0)
            zero_col = torch.zeros((T_sub, 1), device=self.dev, dtype=torch.float32)
            full_logits = torch.cat([zero_col, logits], dim=1)  # (T_sub, K+1)

            # 6. Loss Calculation
            # y_sub: 0 means outside, 1..K means k-th item in assortment
            nll = F.cross_entropy(full_logits, y_sub, reduction="sum")

            # 7. Ridge Regularization (Only for active targets)
            # Inactive params are detached, so they don't contribute to gradient
            ridge = 0.5 * float(self.lambda_0) * sum((p ** 2).sum() for p in targets)

            loss = nll + ridge
            loss.backward()
            return loss

        loss = opt.step(closure)
        return float(loss.item()) if hasattr(loss, "item") else float(loss)

    # -----------------------
    # Main Update Interface
    # -----------------------
    def update_from_zS(
        self,
        z_S_or_z: torch.Tensor,
        S: Union[List[int], torch.Tensor],
        y_vec: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Step update using z (representation), S (assortment), y (outcome).
        1. Add to History.
        2. Update V_inv (Sherman-Morrison) for Active Models.
        3. Optimize Theta (Full-History L-BFGS) for Active Models.
        """
        self.train()
        self.B.eval()

        # Input Normalization
        if z_S_or_z.dim() == 1: z = z_S_or_z
        elif z_S_or_z.dim() == 2: z = z_S_or_z[0]
        else: raise ValueError(f"z shape error: {z_S_or_z.shape}")

        if isinstance(S, list): S_t = torch.tensor(S, device=self.dev, dtype=torch.long)
        else: S_t = S.to(device=self.dev, dtype=torch.long)
        S_t = S_t.clamp(0, self.n_models - 1)
        
        K = int(S_t.numel())
        # y_vec: one-hot of size K+1 (0=outside, 1..K=items)
        y_idx = torch.argmax(y_vec.to(device=self.dev)).long()

        z = z.to(device=self.dev, dtype=torch.float32).view(-1)
        if self.normalize_z: z = F.normalize(z, dim=-1)

        # 1. Append to History
        self._init_history_if_needed(K)
        self._ensure_capacity(self._T + 1)
        
        self._Z_hist[self._T].copy_(z)
        self._S_hist[self._T].copy_(S_t)
        self._y_hist[self._T] = y_idx
        self._T += 1

        # Identify Active Models in this step
        current_active = torch.unique(S_t).tolist()

        # 2. Update Covariance (V_inv) - Active Only (Sherman-Morrison)
        # This is needed for Thompson Sampling regardless of Loss optimization
        with torch.no_grad():
            for idx in current_active:
                V = self.V_inv[idx]
                v = V @ z
                denom = (1.0 + (z @ v)).clamp_min(1e-12)
                self.V_inv[idx] = V - torch.outer(v, v) / denom
                self.V_inv[idx].copy_(0.5 * (self.V_inv[idx] + self.V_inv[idx].T))

        # 3. Optimize Theta (Full History, Active Only)
        loss_total = self._solve_theta_full_history_active_only(current_active)

        return float(loss_total), 0.0

    def update_from_ctx(self, x_ctx, S, y_vec):
        """Wrapper to update from raw context x"""
        if isinstance(x_ctx, np.ndarray):
            x_t = torch.from_numpy(x_ctx).to(device=self.dev, dtype=torch.float32)
        else:
            x_t = x_ctx.to(device=self.dev, dtype=torch.float32)
            
        with torch.no_grad():
            z = self.B_projection(x_t).view(-1)
        
        return self.update_from_zS(z, S, y_vec)

    # -----------------------
    # Utilities (Freeze/Reset)
    # -----------------------
    def freeze_B(self):
        for p in self.B.parameters(): p.requires_grad_(False)
        self.opt_b = None
        self._B_frozen = True
        self.B.eval()

    def unfreeze_B(self, lr_b=None):
        params = list(self.B.parameters())
        if len(params) == 0: return
        for p in params: p.requires_grad_(True)
        self._B_frozen = False
        lr_use = self.lr_b_default if lr_b is None else float(lr_b)
        self.opt_b = torch.optim.Adam(params, lr=lr_use)
        self.B.train()

    def reset_for_online(self):
        self.freeze_B()
        self.eval()
        with torch.no_grad():
            for p in self.theta_list: p.zero_()
            V0 = (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.dev, dtype=torch.float32)
            self.V_inv.copy_(V0.unsqueeze(0).repeat(self.n_models, 1, 1))
        self._reset_history()
        self._ts_chol_fail_count = 0
        self._ts_last_jitter = 0.0

        # ---------- SupCon (B only) ----------
    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        x_ctx = x_ctx.to(device=self.dev, dtype=torch.float32)
        return self.B(x_ctx)