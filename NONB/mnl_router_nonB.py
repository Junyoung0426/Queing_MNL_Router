# mnl_router_NONB.py

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
    - V_inv ≈ (λ I + Σ x x^T)^{-1}  (Sherman–Morrison)
    - Thompson sampling + optimistic reward Re(x,S) 구현.
    """

    def __init__(
        self,
        d_in: int,
        d_proj: int,           # 기존 인터페이스 유지용 (사용 안 함)
        lam_ridge: float = 1.0,
        supcon_temp: float = 0.07,  # 기존 인터페이스 유지용 (사용 안 함)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.d_in = d_in
        self.d = d_in              # 논문에서의 d
        self.device = torch.device(device)
        self.lam_ridge = lam_ridge

        # θ ∈ R^d
        self.theta = nn.Parameter(torch.zeros(self.d, device=self.device))

        # V^{-1}_0 = (1/λ) I
        self.V_inv = (1.0 / lam_ridge) * torch.eye(self.d, device=self.device)

        # 간단한 SGD (weight_decay 로 ridge 포함)
        self.opt_theta = torch.optim.SGD(
            [self.theta], lr=0.05, weight_decay=lam_ridge
        )

    # --------------- feature 전처리 (정규화) ----------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in) → (..., d_in)

        논문은 ∥x_-j∥_2 ≤ 1 을 가정하므로, 간단히 ℓ2 정규화. 
        (기존 forward_proj 의 normalize 역할만 남김)
        """
        x = x.to(self.device)
        return F.normalize(x, dim=-1)

    # 과거 코드 호환용 alias
    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        return self._preprocess(x)

    # --------------- MNL Scoring ----------------
    def get_scores(self, X_S: torch.Tensor) -> torch.Tensor:
        """
        X_S: (|S|, d_in) or (B, |S|, d_in)
        return:
          - (|S|,)   if X_S.dim() == 2
          - (B, |S|) if X_S.dim() == 3
        """
        if X_S.dim() == 2:
            feats = self._preprocess(X_S)              # (|S|, d)
            logits = feats @ self.theta                # (|S|,)
        elif X_S.dim() == 3:
            feats = self._preprocess(X_S)              # (B, |S|, d)
            logits = torch.einsum("bid,d->bi", feats, self.theta)
        else:
            raise ValueError("X_S must be 2D or 3D tensor.")
        return logits

    # --------------- θ noise 샘플링 ----------------
    def sample_theta_noise(
        self,
        alpha_t: float,
        M: int = 4,
    ) -> torch.Tensor:
        """
        θ̃^(i) ~ N(θ̂, α_t^2 V^{-1}) 의 noise 부분만 미리 샘플링.

        return:
          noise_vectors: (d, M)
        """
        with torch.no_grad():
            try:
                L = torch.linalg.cholesky(self.V_inv)
            except RuntimeError:
                L = torch.linalg.cholesky(
                    self.V_inv + 1e-6 * torch.eye(self.d, device=self.device)
                )

            u = torch.randn(self.d, M, device=self.device)   # (d, M)
            noise_vectors = alpha_t * (L @ u)                # (d, M)
        return noise_vectors

    # --------------- Optimistic reward Re(x, S) ----------------
    def sample_optimistic_reward(
        self,
        X_S: torch.Tensor,
        alpha_t: float,
        M: int = 4,
        noise_vectors: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Unknown-horizon 알고리즘에서 û_{etj}(x), Re(x, S) 근사. (Algorithm 1)
        """
        with torch.no_grad():
            feats = self._preprocess(X_S)  # (|S|, d)

            if noise_vectors is None:
                noise_vectors = self.sample_theta_noise(alpha_t=alpha_t, M=M)

            # 안전하게 차원 정렬
            if noise_vectors.shape[0] != feats.shape[-1]:
                noise_vectors = noise_vectors[: feats.shape[-1]]

            mean_scores = feats @ self.theta                         # (|S|,)
            uncertainty = feats @ noise_vectors                      # (|S|, M)
            sampled_scores = mean_scores.unsqueeze(1) + uncertainty  # (|S|, M)

            optimistic_u = torch.max(sampled_scores, dim=1).values   # (|S|,)

            max_u = torch.max(optimistic_u)
            exps = torch.exp(optimistic_u - max_u)
            denom = torch.exp(-max_u) + exps.sum()
            R_tilde = (exps.sum() / denom).item()
            return R_tilde

    # --------------- Joint Update (θ + V_inv) ----------------
    def update(
        self,
        X_S: torch.Tensor,        # (|S|, d_in)
        y_vec: torch.Tensor,      # one-hot (|S|), all-zero면 outside option
        batch_supcon_size: int = 0,  # 인터페이스 유지용 (무시)
        lambda_sc: float = 0.0,      # 인터페이스 유지용 (무시)
    ) -> Tuple[float, float]:
        """
        한 라운드 (x_t, S_t, y_t)에 대해
        - θ: MNL NLL loss로 1-step SGD 업데이트
        - V_inv: 현재 x 기준 Sherman-Morrison 업데이트
        - SupCon은 사용하지 않으므로 항상 0.0 반환
        """
        self.train()

        feats = self._preprocess(X_S)            # (|S|, d)
        logits = feats @ self.theta              # (|S|,)

        zero_tensor = torch.tensor([0.0], device=self.device)
        full_logits = torch.cat([zero_tensor, logits], dim=0)  # (|S|+1,)

        log_probs = torch.log_softmax(full_logits, dim=0)

        if y_vec.sum() == 0:
            target_idx = 0
        else:
            target_idx = torch.argmax(y_vec) + 1

        loss_mnl = -log_probs[target_idx]

        self.opt_theta.zero_grad()
        loss_mnl.backward()
        self.opt_theta.step()

        # V_inv 업데이트 (Sherman–Morrison)
        with torch.no_grad():
            feats_det = feats.detach()
            for j in range(feats_det.shape[0]):
                x = feats_det[j]            # (d,)
                v = self.V_inv @ x          # (d,)
                denom = 1.0 + x @ v         # scalar
                self.V_inv -= torch.outer(v, v) / denom

        # 두 번째 리턴값은 SupCon loss 자리에 항상 0.0
        return float(loss_mnl.item()), 0.0

    # --------------- SupCon buffer: no-op stub ----------------
    def add_to_buffer(self, x_feature: torch.Tensor, y_idx: int):
        """
        과거 SupCon 인터페이스 호환용 no-op.
        """
        return


# ----------------- 기존 유틸은 그대로 사용 -----------------
def build_X_S_from_context(
    x_ctx: np.ndarray,   # (d_ctx,)
    S: List[int],        # 모델 index 리스트
    N_models: int,
    device: torch.device,
) -> torch.Tensor:
    M = len(S)

    x_ctx_tensor = torch.from_numpy(x_ctx).float().to(device)
    x_repeated = x_ctx_tensor.unsqueeze(0).expand(M, -1)  # (M, d_ctx)

    indices = torch.tensor(S, device=device)
    one_hots = F.one_hot(indices, num_classes=N_models).float()  # (M, N_models)

    X_S = torch.cat([x_repeated, one_hots], dim=1)  # (M, d_ctx + N_models)
    return X_S


def predict_model(
    router: MNLRouter,
    x_ctx: np.ndarray,
    N_models: int,
    device: torch.device,
) -> int:
    router.eval()
    with torch.no_grad():
        X_all = build_X_S_from_context(
            x_ctx, list(range(N_models)), N_models, device
        )
        logits = router.get_scores(X_all)  # (N_models,)
        k_hat = int(torch.argmax(logits).item())
    return k_hat
