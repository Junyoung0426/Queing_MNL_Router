# mnl_router_nonB.py

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Vanilla linear MNL router (no projection B, no SupCon).

    - 입력 feature x: context + model one-hot  → R^{d_in}
    - θ ∈ R^{d_in}
    - V_inv ≈ (λ_0 I + Σ x x^T)^{-1}  (Sherman–Morrison)
    - Thompson sampling + optimistic reward R^e(x,S) 구현.

    핵심:
      Algorithm 1 Line 10:
        θ̂_t = argmin_θ  -∑_{i=1}^t log p(y_i | x_i,S_i; θ) + (λ_0/2)||θ||^2
        매 라운드마다 LBFGS로 minimize 한다.

    GPU 최적화:
      - 히스토리를 (T, K, d) 텐서 버퍼로 누적한다.
      - closure에서 한 번에 matmul + cross_entropy로 objective를 계산한다.
    """

    def __init__(
        self,
        d_in: int,
        lambda_0: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        theta_solver: str = "lbfgs",
        lbfgs_max_iter: int = 50,
        lbfgs_history_size: int = 50,
        lbfgs_line_search: str = "strong_wolfe",
        hist_init_capacity: int = 2048,
    ):
        super().__init__()

        self.d_in = d_in
        self.d = d_in
        self.device = torch.device(device)
        self.lambda_0 = float(lambda_0)

        # θ ∈ R^d
        self.theta = nn.Parameter(torch.zeros(self.d, device=self.device))

        # V_0^{-1} = (1/λ_0) I
        self.V_inv = (1.0 / self.lambda_0) * torch.eye(self.d, device=self.device)

        # θ minimize solver
        self.theta_solver = theta_solver
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.lbfgs_line_search = lbfgs_line_search

        # ---------- History buffer (GPU-friendly) ----------
        # X_hist: (cap, K, d)  (K는 첫 update에서 확정)
        # y_hist: (cap,)       (정답 index: outside=0, inside=1..K)
        self._K: Optional[int] = None
        self._T: int = 0
        self._cap: int = int(hist_init_capacity)

        self._X_hist: Optional[torch.Tensor] = None
        self._y_hist: Optional[torch.Tensor] = None

    # ---------------- feature preprocess ----------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        ||x||_2 <= 1 가정 만족시키기 위해 norm>1이면 스케일링, norm<=1이면 그대로 둔다.
        """
        x = x.to(self.device)
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        scale = torch.clamp(norm, min=1.0)
        return x / scale

    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        return self._preprocess(x)

    # ---------------- MNL scoring ----------------
    def get_scores(self, X_S: torch.Tensor) -> torch.Tensor:
        """
        X_S: (|S|, d) or (B, |S|, d)
        """
        feats = self._preprocess(X_S)
        logits = feats @ self.theta
        return logits

    # ---------------- θ noise 샘플링 ----------------
    def sample_theta_noise(self, alpha_t: float, M: int = 4) -> torch.Tensor:
        """
        θ̃^(i) ~ N(θ̂, α_t^2 V^{-1}) 의 noise만 샘플링한다.
        return: (d, M)
        """
        with torch.no_grad():
            try:
                L = torch.linalg.cholesky(self.V_inv)
            except RuntimeError:
                L = torch.linalg.cholesky(
                    self.V_inv + 1e-6 * torch.eye(self.d, device=self.device)
                )

            u = torch.randn(self.d, M, device=self.device)
            noise_vectors = float(alpha_t) * (L @ u)
        return noise_vectors

    # ---------------- Optimistic reward R^e(x,S) (single) ----------------
    def sample_optimistic_reward(
        self,
        X_S: torch.Tensor,
        alpha_t: float,
        M: int = 4,
        noise_vectors: Optional[torch.Tensor] = None,
    ) -> float:

        with torch.no_grad():
            feats = self._preprocess(X_S)  # (K, d)

            if noise_vectors is None:
                noise_vectors = self.sample_theta_noise(alpha_t=alpha_t, M=M)

            if noise_vectors.shape[0] != feats.shape[-1]:
                noise_vectors = noise_vectors[: feats.shape[-1]]

            mean_scores = feats @ self.theta                         # (K,)
            uncertainty = feats @ noise_vectors                      # (K, M)
            sampled_scores = mean_scores.unsqueeze(1) + uncertainty  # (K, M)

            optimistic_u = torch.max(sampled_scores, dim=1).values    # (K,)

            max_u = torch.max(optimistic_u)
            exps = torch.exp(optimistic_u - max_u)
            denom = torch.exp(-max_u) + exps.sum()
            return float((exps.sum() / denom).item())

    # ---------------- Optimistic reward R^e(x,S) (batch) ----------------
    def sample_optimistic_reward_batch(
        self,
        X_S_batch: torch.Tensor,      # (B, K, d_in)
        noise_vectors: torch.Tensor,  # (d, M)
    ) -> torch.Tensor:

        with torch.no_grad():
            feats = self._preprocess(X_S_batch)  # (B, K, d)

            mean = torch.matmul(feats, self.theta)                      # (B, K)
            unc = torch.einsum("bkd,dm->bkm", feats, noise_vectors)     # (B, K, M)
            uopt = (mean.unsqueeze(-1) + unc).amax(dim=-1)              # (B, K)

            max_u = uopt.max(dim=1, keepdim=True).values                # (B, 1)
            exps = torch.exp(uopt - max_u)                              # (B, K)
            sumexp = exps.sum(dim=1)                                    # (B,)
            denom = torch.exp(-max_u.squeeze(1)) + sumexp               # (B,)
            return sumexp / denom                                       # (B,)

    # ---------------- history buffer helpers ----------------
    def _ensure_capacity(self, need_T: int):
        if self._X_hist is None or self._y_hist is None:
            return
        if need_T <= self._cap:
            return

        new_cap = self._cap
        while new_cap < need_T:
            new_cap *= 2

        X_new = torch.empty(
            (new_cap, self._K, self.d),
            device=self.device,
            dtype=self._X_hist.dtype,
        )
        y_new = torch.empty(
            (new_cap,),
            device=self.device,
            dtype=self._y_hist.dtype,
        )

        if self._T > 0:
            X_new[: self._T].copy_(self._X_hist[: self._T])
            y_new[: self._T].copy_(self._y_hist[: self._T])

        self._X_hist = X_new
        self._y_hist = y_new
        self._cap = new_cap

    def _init_history_if_needed(self, K: int):
        if self._K is None:
            self._K = int(K)
            self._X_hist = torch.empty(
                (self._cap, self._K, self.d),
                device=self.device,
                dtype=torch.float32,
            )
            self._y_hist = torch.empty(
                (self._cap,),
                device=self.device,
                dtype=torch.long,
            )
            self._T = 0
        else:
            if int(K) != self._K:
                raise ValueError(
                    f"|S_t|가 고정이 아니다. expected K={self._K}, got K={K}"
                )

    # ---------------- Line 10 objective ----------------
    def _objective(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
        X_batch: (T, K, d)
        y_batch: (T,)  [0..K] (outside=0)
        """
        T = X_batch.shape[0]
        logits = torch.matmul(X_batch, self.theta)                      # (T, K)
        zero_col = torch.zeros((T, 1), device=self.device, dtype=logits.dtype)
        full_logits = torch.cat([zero_col, logits], dim=1)              # (T, K+1)

        nll = F.cross_entropy(full_logits, y_batch, reduction="sum")
        ridge = 0.5 * self.lambda_0 * torch.sum(self.theta * self.theta)
        return nll + ridge

    def _solve_theta_minimize(self) -> float:
        """
        매 라운드 θ̂_t를 LBFGS로 minimize 한다.
        """
        if self._T == 0:
            return 0.0
        if self.theta_solver.lower() != "lbfgs":
            raise ValueError("theta_solver는 현재 lbfgs만 사용")

        assert self._X_hist is not None and self._y_hist is not None

        X_batch = self._X_hist[: self._T]  # view
        y_batch = self._y_hist[: self._T]  # view

        opt = torch.optim.LBFGS(
            [self.theta],
            lr=1.0,
            max_iter=self.lbfgs_max_iter,
            history_size=self.lbfgs_history_size,
            line_search_fn=self.lbfgs_line_search,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._objective(X_batch, y_batch)
            loss.backward()
            return loss

        loss = opt.step(closure)
        return float(loss.item()) if hasattr(loss, "item") else float(loss)

    # ---------------- Joint Update (θ + V_inv) ----------------
    def update(
        self,
        X_S: torch.Tensor,        # (K, d_in)
        y_vec: torch.Tensor,      # one-hot(K,) or all-zero(outside)
    ) -> Tuple[float, float]:
        """
        1) (feats_t, target_idx)를 히스토리 텐서 버퍼에 append
        2) θ̂_t = argmin(...) 를 LBFGS로 minimize
        3) V_inv를 Sherman–Morrison으로 업데이트
        """
        self.train()

        feats_t = self._preprocess(X_S).detach().contiguous()  # (K, d)
        K = feats_t.shape[0]
        self._init_history_if_needed(K)

        # y_vec -> target index (outside=0, inside=1..K)  (동기화 .item() 안 쓴다)
        y_vec = y_vec.to(self.device)
        max_val, arg = torch.max(y_vec, dim=0)  # 0-d tensor
        zero = torch.zeros((), device=self.device, dtype=torch.long)
        target = torch.where(max_val > 0, arg.long() + 1, zero)

        # append to history buffer
        self._ensure_capacity(self._T + 1)
        assert self._X_hist is not None and self._y_hist is not None
        self._X_hist[self._T].copy_(feats_t)
        self._y_hist[self._T] = target
        self._T += 1

        # Line 10 minimize
        loss_val = self._solve_theta_minimize()

        # Line 11 V_inv 업데이트 (Sherman–Morrison)
        with torch.no_grad():
            for j in range(feats_t.shape[0]):
                x = feats_t[j]
                v = self.V_inv @ x
                denom = 1.0 + x @ v
                self.V_inv -= torch.outer(v, v) / denom

        return loss_val, 0.0

    def add_to_buffer(self, x_feature: torch.Tensor, y_idx: int):
        return

    def reset_history(self):
        self._T = 0


# ----------------- 유틸 함수 -----------------

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

