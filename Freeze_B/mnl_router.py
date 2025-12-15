# mnl_router.py
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Offline:
      - B만 SupCon으로 학습한다 (theta/V_inv 업데이트 안 한다)
      - model_emb/model_proj는 기본 freeze

    Online:
      - B, model parts freeze
      - theta만 0..t full-history MLE(+ridge)를 LBFGS로 minimize 한다
      - V_inv는 현재 라운드 feature로 Sherman–Morrison 업데이트한다

    입력 형식:
      1) index-format:  x = [x_ctx (d_ctx), model_idx (1)]          => (d_ctx+1)
      2) onehot-format: x = [x_ctx (d_ctx), onehot(model) (K)]      => (d_ctx+n_models)

    Interaction:
      z_ctx = B(x_ctx)                     (d_proj)
      z_model = model_proj(model_emb(idx)) (d_proj)   (default: frozen)
      z_final = clip_norm(z_ctx * z_model) (d_proj)
      score = z_final^T theta
    """

    def __init__(
        self,
        d_ctx: int,
        n_models: int,
        d_proj: int,
        lambda_0: float = 1.0,
        supcon_temp: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # B 타입
        b_type: str = "mlp",          # "linear" or "mlp"
        b_hidden_mult: int = 2,       # mlp hidden = d_proj * b_hidden_mult
        # model embedding (고정 사용)
        d_model_emb: int = 128,
        # Online LBFGS
        theta_solver: str = "lbfgs",
        lbfgs_max_iter: int = 30,
        lbfgs_history_size: int = 10,
        lbfgs_line_search: str = "strong_wolfe",
        hist_init_capacity: int = 2048,
        # Offline optimizer (B만)
        lr_b: float = 1e-3,
        buffer_max_size: int = 5000,
    ):
        super().__init__()
        self.d_ctx = int(d_ctx)
        self.n_models = int(n_models)
        self.d_proj = int(d_proj)
        self.device = torch.device(device)

        self.lambda_0 = float(lambda_0)
        self.supcon_temp = float(supcon_temp)

        self.d_final_feature = self.d_proj
        self.d = self.d_final_feature  # queue_env에서 쓰는 d와 맞춘다

        # -------- Model Identity Embedding (default: freeze) --------
        self.d_model_emb = int(d_model_emb)
        self.model_emb = nn.Embedding(self.n_models, self.d_model_emb).to(self.device)
        self.model_proj = nn.Linear(self.d_model_emb, self.d_proj, bias=False).to(self.device)

        # -------- Context Encoder B --------
        b_type = b_type.lower().strip()
        self.b_type = b_type
        if b_type == "linear":
            self.B = nn.Linear(self.d_ctx, self.d_proj, bias=False).to(self.device)
        elif b_type == "mlp":
            d_hidden = self.d_proj * int(b_hidden_mult)
            self.B = nn.Sequential(
                nn.Linear(self.d_ctx, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, self.d_proj),
                nn.LayerNorm(self.d_proj),
            ).to(self.device)
        else:
            raise ValueError(f"b_type must be 'linear' or 'mlp', got {b_type}")

        # -------- Online bandit parameters --------
        self.theta = nn.Parameter(torch.zeros(self.d_final_feature, device=self.device))

        self.register_buffer(
            "V_inv",
            (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.device)
        )

        # -------- Offline SupCon buffer (지금 파이프라인에선 사실상 미사용) --------
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[int] = []
        self.buffer_max_size = int(buffer_max_size)

        # -------- Offline optimizer: B만 --------
        self.opt_b: Optional[torch.optim.Optimizer] = torch.optim.Adam(
            self.B.parameters(), lr=float(lr_b)
        )

        # -------- Online LBFGS settings --------
        self.theta_solver = theta_solver
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = lbfgs_line_search

        # -------- Online history buffer: (T,K,d_proj) --------
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)
        self._Z_hist: Optional[torch.Tensor] = None
        self._y_hist: Optional[torch.Tensor] = None  # (T,), 0..K (outside=0)

        self._freeze_model_parts()

    # ======================================================
    # Utils: norm clipping (||x||<=1)
    # ======================================================
    def _clip_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        scale = torch.clamp(norm, min=1.0)
        return x / scale

    # ======================================================
    # Input parsing: index-format or onehot-format
    # ======================================================
    def _split_input(self, x_combined: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
          x_ctx: (..., d_ctx)
          m_idx: (...,) long
        """
        if x_combined.dim() == 1:
            x_combined = x_combined.unsqueeze(0)
        x_combined = x_combined.to(self.device)

        D = x_combined.shape[-1]
        if D == self.d_ctx + 1:
            x_ctx = x_combined[..., :self.d_ctx]
            m_raw = x_combined[..., self.d_ctx]
            # float로 들어오는 model idx 방어적으로 반올림
            m_idx = torch.round(m_raw).long()
        elif D == self.d_ctx + self.n_models:
            x_ctx = x_combined[..., :self.d_ctx]
            onehot = x_combined[..., self.d_ctx:]
            m_idx = torch.argmax(onehot, dim=-1).long()
        else:
            raise ValueError(
                f"Input dim mismatch. got {D}, expected {self.d_ctx+1} (idx) or {self.d_ctx+self.n_models} (onehot)"
            )

        m_idx = m_idx.clamp(min=0, max=self.n_models - 1)
        return x_ctx, m_idx

    # ======================================================
    # Projection: z_final = z_ctx * z_model
    # ======================================================
    def forward_proj(self, x_combined: torch.Tensor) -> torch.Tensor:
        x_ctx, m_idx = self._split_input(x_combined)

        z_ctx = self._clip_norm(self.B(x_ctx))

        m_vec = self.model_emb(m_idx)
        z_model = self._clip_norm(self.model_proj(m_vec))

        z_final = self._clip_norm(z_ctx * z_model)
        return z_final

    # Offline SupCon용: context만
    def forward_ctx_supcon(self, x_ctx: torch.Tensor) -> torch.Tensor:
        if x_ctx.dim() == 1:
            x_ctx = x_ctx.unsqueeze(0)
        x_ctx = x_ctx.to(self.device)
        z = self.B(x_ctx)
        return F.normalize(z, dim=-1)

    # ======================================================
    # Freeze / Unfreeze
    # ======================================================
    def _freeze_model_parts(self):
        for p in self.model_emb.parameters():
            p.requires_grad = False
        for p in self.model_proj.parameters():
            p.requires_grad = False

    def unfreeze_B(self, lr_b: float = 1e-3):
        print(f"[MNLRouter] Unfreezing B only (lr={lr_b})...")
        for p in self.B.parameters():
            p.requires_grad = True
        self._freeze_model_parts()
        self.opt_b = torch.optim.Adam(self.B.parameters(), lr=float(lr_b))

    def freeze_B(self):
        print("[MNLRouter] Freezing B & model parts...")
        for p in self.B.parameters():
            p.requires_grad = False
        self._freeze_model_parts()
        self.opt_b = None

    def reset_for_online(self):
        print("[MNLRouter] Resetting theta & V_inv & history for online phase...")
        with torch.no_grad():
            self.theta.zero_()
            self.V_inv.copy_(
                (1.0 / self.lambda_0) * torch.eye(self.d_final_feature, device=self.device)
            )
        self._reset_history()

    def _reset_history(self):
        self._K = None
        self._T = 0
        self._Z_hist = None
        self._y_hist = None

    # ======================================================
    # MNL scoring
    # ======================================================
    def get_scores(self, X_S: torch.Tensor) -> torch.Tensor:
        z = self.forward_proj(X_S)
        if X_S.dim() == 2:
            return z @ self.theta
        if X_S.dim() == 3:
            return torch.einsum("bid,d->bi", z, self.theta)
        raise ValueError("X_S must be 2D or 3D tensor.")

    # ======================================================
    # TS / Optimistic reward
    # ======================================================
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        d_feat = self.d_final_feature
        with torch.no_grad():
            try:
                L = torch.linalg.cholesky(self.V_inv)
            except RuntimeError:
                L = torch.linalg.cholesky(
                    self.V_inv + 1e-6 * torch.eye(d_feat, device=self.device)
                )
            u = torch.randn(d_feat, M, device=self.device)
            return float(alpha_t) * (L @ u)

    def sample_optimistic_reward_batch(
        self,
        X_S_batch: torch.Tensor,      # (B, K, d_in)
        noise_vectors: torch.Tensor,  # (d, M)
    ) -> torch.Tensor:
        with torch.no_grad():
            z = self.forward_proj(X_S_batch)                           # (B, K, d)
            mean = torch.einsum("bkd,d->bk", z, self.theta)            # (B, K)
            unc = torch.einsum("bkd,dm->bkm", z, noise_vectors)        # (B, K, M)
            uopt = (mean.unsqueeze(-1) + unc).amax(dim=-1)             # (B, K)

            max_u = uopt.max(dim=1, keepdim=True).values               # (B, 1)
            exps = torch.exp(uopt - max_u)
            sumexp = exps.sum(dim=1)
            denom = torch.exp(-max_u.squeeze(1)) + sumexp
            return sumexp / denom

    # ======================================================
    # Offline: SupCon only (B만 업데이트)
    # ======================================================
    def supcon_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=self.device)

        labels = labels.view(-1).to(self.device)
        feats = F.normalize(features, dim=-1)

        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(B, device=self.device)

        sim = torch.matmul(feats, feats.T) / self.supcon_temp
        logits = sim - mask_self * 1e9

        mask_pos = mask_pos - mask_self
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_count = mask_pos.sum(dim=1).clamp_min(1.0)
        loss_i = -(log_prob * mask_pos).sum(dim=1) / pos_count
        return loss_i.mean()

    def supcon_step(
        self,
        X_batch: torch.Tensor,   # (B, d_ctx) or (B, d_ctx+1) or (B, d_ctx+n_models)
        y_batch: torch.Tensor,   # (B,) winner index
        lambda_sc: float = 1.0,
    ) -> float:
        if lambda_sc <= 0.0:
            return 0.0
        if self.opt_b is None:
            raise RuntimeError("Run unfreeze_B() first.")

        self.train()
        self.opt_b.zero_grad(set_to_none=True)

        X_batch = X_batch.to(self.device)
        x_ctx = X_batch[..., :self.d_ctx]

        z = self.forward_ctx_supcon(x_ctx)
        loss = self.supcon_loss(z, y_batch)
        (float(lambda_sc) * loss).backward()
        self.opt_b.step()
        return float(loss.item())

    # ======================================================
    # Online: full-history MLE for theta only (LBFGS)
    # ======================================================
    def _ensure_capacity(self, need_T: int):
        if self._Z_hist is None or self._y_hist is None:
            return
        if need_T <= self._cap:
            return

        new_cap = self._cap
        while new_cap < need_T:
            new_cap *= 2

        Z_new = torch.empty(
            (new_cap, self._K, self.d_final_feature),
            device=self.device,
            dtype=self._Z_hist.dtype
        )
        y_new = torch.empty((new_cap,), device=self.device, dtype=self._y_hist.dtype)

        if self._T > 0:
            Z_new[: self._T].copy_(self._Z_hist[: self._T])
            y_new[: self._T].copy_(self._y_hist[: self._T])

        self._Z_hist = Z_new
        self._y_hist = y_new
        self._cap = new_cap

    def _init_history_if_needed(self, K: int):
        if self._K is None:
            self._K = int(K)
            self._Z_hist = torch.empty(
                (self._cap, self._K, self.d_final_feature),
                device=self.device,
                dtype=torch.float32
            )
            self._y_hist = torch.empty((self._cap,), device=self.device, dtype=torch.long)
            self._T = 0
        else:
            if int(K) != self._K:
                raise ValueError(f"|S_t|가 고정이 아니다. expected K={self._K}, got K={K}")

    def _objective_vectorized(self, Z_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        T = Z_batch.shape[0]
        logits = torch.einsum("tkd,d->tk", Z_batch, self.theta)            # (T, K)
        zero_col = torch.zeros((T, 1), device=self.device, dtype=logits.dtype)
        full_logits = torch.cat([zero_col, logits], dim=1)                 # (T, K+1)

        nll = F.cross_entropy(full_logits, y_batch, reduction="sum")
        ridge = 0.5 * self.lambda_0 * torch.sum(self.theta * self.theta)
        return nll + ridge

    def _solve_theta_minimize(self) -> float:
        if self._T == 0:
            return 0.0
        if self.theta_solver.lower() != "lbfgs":
            raise ValueError("theta_solver는 현재 lbfgs만 지원한다")

        assert self._Z_hist is not None and self._y_hist is not None
        Z_batch = self._Z_hist[: self._T]
        y_batch = self._y_hist[: self._T]

        opt = torch.optim.LBFGS(
            [self.theta],
            lr=1.0,
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

    def update(
        self,
        X_S: torch.Tensor,     # (K, d_in)
        y_vec: torch.Tensor,   # one-hot(K,) or all-zero(outside)
        batch_supcon_size: int = 64,  # not used
        lambda_sc: float = 0.0,       # not used
    ) -> Tuple[float, float]:
        self.train()
        X_S = X_S.to(self.device)
        y_vec = y_vec.to(self.device)

        # target idx (outside=0)
        max_val, arg = torch.max(y_vec, dim=0)
        target_idx = torch.where(
            max_val > 0,
            arg.long() + 1,
            torch.zeros((), device=self.device, dtype=torch.long)
        )

        z_final = self.forward_proj(X_S).detach().contiguous()  # (K, d)
        K = z_final.shape[0]
        self._init_history_if_needed(K)

        self._ensure_capacity(self._T + 1)
        assert self._Z_hist is not None and self._y_hist is not None

        self._Z_hist[self._T].copy_(z_final)
        self._y_hist[self._T] = target_idx
        self._T += 1

        loss_total = self._solve_theta_minimize()

        # V_inv 업데이트 (current z only)
        with torch.no_grad():
            for j in range(z_final.shape[0]):
                z = z_final[j]
                v = self.V_inv @ z
                denom = 1.0 + z @ v
                self.V_inv -= torch.outer(v, v) / denom

        return float(loss_total), 0.0

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def build_X_S_from_context_idx(
    x_ctx: np.ndarray,
    S: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    index-format: (d_ctx + 1)  [x_ctx | model_idx]
    """
    M = len(S)
    x_ctx_tensor = torch.from_numpy(x_ctx).float().to(device)
    x_rep = x_ctx_tensor.unsqueeze(0).expand(M, -1)  # (M, d_ctx)
    idx = torch.tensor(S, device=device, dtype=torch.float32).unsqueeze(1)  # (M,1)
    return torch.cat([x_rep, idx], dim=1)  # (M, d_ctx+1)
