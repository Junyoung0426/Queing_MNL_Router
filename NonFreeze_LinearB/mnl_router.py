from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNLRouter(nn.Module):
    """
    Shared-theta MNL Router + SupCon + V_inv (Neural Linear-style).

    입력 feature:
      - x_ctx: (d_ctx,)
      - model one-hot: (K,)
      => concat → (d_ctx + K,) = d_in

    구조:
      B: R^{d_in} -> R^{d_proj}  (MNL + SupCon joint gradient로 학습)
      theta: R^{d_proj}          (MNL 파라미터)
      V_inv: R^{d_proj x d_proj} (UCB/TS용 inverse covariance)
    """

    def __init__(
        self,
        d_in: int,
        d_proj: int,
        lam_ridge: float = 1.0,
        supcon_temp: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_proj = d_proj
        self.device = torch.device(device)
        self.supcon_temp = supcon_temp
        self.lam_ridge = lam_ridge

        # Projection B
        self.B = nn.Linear(d_in, d_proj, bias=False).to(self.device)

        # Shared theta (MNL 파라미터)
        self.theta = nn.Parameter(torch.zeros(d_proj, device=self.device))

        # V^{-1} 초기값 = (1/λ) I
        self.V_inv = (1.0 / lam_ridge) * torch.eye(d_proj, device=self.device)

        # SupCon replay buffer (CPU에 저장)
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[int] = []
        self.buffer_max_size = 2000

        # Optimizers
        self.opt_theta = torch.optim.Adam(
            [self.theta], lr=0.1, weight_decay=lam_ridge
        )
        self.opt_b = torch.optim.Adam(self.B.parameters(), lr=1e-3)

    # ----------------- Projection -----------------
    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        return: (..., d_proj) (ℓ2 정규화)
        """
        z = self.B(x)
        return F.normalize(z, dim=-1)

    # ----------------- MNL Scoring -----------------
    def get_scores(self, X_S: torch.Tensor) -> torch.Tensor:
        """
        X_S: (|S|, d_in) or (B, |S|, d_in)
        return:
          - (|S|,)   if X_S.dim() == 2
          - (B, |S|) if X_S.dim() == 3
        """
        if X_S.dim() == 2:
            z = self.forward_proj(X_S)          # (|S|, d_proj)
            logits = z @ self.theta             # (|S|,)
        elif X_S.dim() == 3:
            z = self.forward_proj(X_S)          # (B, |S|, d_proj)
            logits = torch.einsum("bid,d->bi", z, self.theta)
        else:
            raise ValueError("X_S must be 2D or 3D tensor.")
        return logits

    # ----------------- θ noise 샘플링 -----------------
    def sample_theta_noise(
        self,
        alpha_t: float,
        M: int = 4,
    ) -> torch.Tensor:
        """
        θ̃^(i) ~ N(θ̂, α_t^2 V^{-1}) 의 noise 부분만 미리 샘플링.

        return:
          noise_vectors: (d_proj, M)
        """
        with torch.no_grad():
            try:
                L = torch.linalg.cholesky(self.V_inv)
            except RuntimeError:
                L = torch.linalg.cholesky(
                    self.V_inv + 1e-6 * torch.eye(self.d_proj, device=self.device)
                )

            u = torch.randn(self.d_proj, M, device=self.device)  # (d_proj, M)
            noise_vectors = alpha_t * (L @ u)                    # (d_proj, M)
        return noise_vectors

    # ----------------- Optimistic reward Re(x, S) -----------------
    def sample_optimistic_reward(
        self,
        X_S: torch.Tensor,
        alpha_t: float,
        M: int = 4,
        noise_vectors: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Unknown-horizon 알고리즘에서 ũ_j, Re(x, S) 근사.
        """
        with torch.no_grad():
            z_s = self.forward_proj(X_S)  # (|S|, d_proj)

            if noise_vectors is None:
                noise_vectors = self.sample_theta_noise(alpha_t=alpha_t, M=M)

            mean_scores = z_s @ self.theta                         # (|S|,)
            uncertainty = z_s @ noise_vectors                       # (|S|, M)
            sampled_scores = mean_scores.unsqueeze(1) + uncertainty  # (|S|, M)
            optimistic_u = torch.max(sampled_scores, dim=1).values   # (|S|,)

            max_u = torch.max(optimistic_u)
            exps = torch.exp(optimistic_u - max_u)
            denom = torch.exp(-max_u) + exps.sum()
            R_tilde = (exps.sum() / denom).item()
            return R_tilde

    # ----------------- SupCon buffer -----------------
    def add_to_buffer(self, x_feature: torch.Tensor, y_idx: int):
        """
        x_feature: (d_in,) – context + one-hot(model)
        y_idx    : 선택된 모델 index
        """
        self.buffer_x.append(x_feature.detach().cpu())
        self.buffer_y.append(int(y_idx))
        if len(self.buffer_x) > self.buffer_max_size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)

    # ----------------- SupCon loss -----------------
    def supcon_loss(
        self,
        features: torch.Tensor,  # (B, d_proj)
        labels: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss (Khosla et al., 2020).
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=self.device)

        labels = labels.view(-1)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        mask_self = torch.eye(B, device=self.device)

        sim = torch.matmul(features, features.T) / self.supcon_temp  # (B,B)
        logits = sim - mask_self * 1e9                               # self-sim ~ -inf

        mask_pos = mask_pos - mask_self

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_count = mask_pos.sum(dim=1).clamp_min(1.0)
        loss_i = -(log_prob * mask_pos).sum(dim=1) / pos_count
        return loss_i.mean()

    # ----------------- Joint Update (θ + B + V_inv) -----------------
    def update(
        self,
        X_S: torch.Tensor,        # (|S|, d_in)
        y_vec: torch.Tensor,      # one-hot (|S|), all-zero면 outside option
        batch_supcon_size: int = 64,
        lambda_sc: float = 1.0,   # SupCon 가중치
    ) -> Tuple[float, float]:
        """
        한 라운드 (x_t, S_t, y_t)에 대해
        - θ: MNL NLL loss로 업데이트
        - B: MNL NLL + λ_sc * SupCon loss joint gradient로 업데이트
        - V_inv: 현재 z 기준 Sherman-Morrison 업데이트
        """
        self.train()

        # ----- 1. MNL Loss -----
        z_s = self.forward_proj(X_S)            # (|S|, d_proj)
        logits = z_s @ self.theta              # (|S|,)

        zero_tensor = torch.tensor([0.0], device=self.device)
        full_logits = torch.cat([zero_tensor, logits], dim=0)  # (|S|+1,)

        log_probs = torch.log_softmax(full_logits, dim=0)

        if y_vec.sum() == 0:
            # outside option
            target_idx = 0
        else:
            target_idx = torch.argmax(y_vec) + 1

        loss_mnl = -log_probs[target_idx]

        # ----- 2. SupCon Loss (buffer에서 미니배치) -----
        loss_sc = torch.tensor(0.0, device=self.device)
        if (lambda_sc > 0.0) and (len(self.buffer_x) >= batch_supcon_size):
            idxs = np.random.choice(len(self.buffer_x), batch_supcon_size, replace=False)
            batch_x = torch.stack([self.buffer_x[i] for i in idxs]).to(self.device)
            batch_y = torch.tensor([self.buffer_y[i] for i in idxs], device=self.device)

            z_batch = self.forward_proj(batch_x)
            loss_sc = self.supcon_loss(z_batch, batch_y)

        # ----- 3. Backward (θ + B) -----
        self.opt_theta.zero_grad()
        self.opt_b.zero_grad()

        total_loss = loss_mnl + lambda_sc * loss_sc
        total_loss.backward()

        self.opt_theta.step()
        self.opt_b.step()

        # ----- 4. V_inv 업데이트 (Sherman–Morrison) -----
        with torch.no_grad():
            z_s_det = z_s.detach()
            for j in range(z_s_det.shape[0]):
                z = z_s_det[j]
                v = self.V_inv @ z
                denom = 1.0 + z @ v
                self.V_inv -= torch.outer(v, v) / denom

        return float(loss_mnl.item()), float(loss_sc.item())


# ----------------- 공통 유틸: x_ctx + one-hot → X_S -----------------
def build_X_S_from_context(
    x_ctx: np.ndarray,   # (d_ctx,)
    S: List[int],        # 모델 index 리스트
    N_models: int,
    device: torch.device,
) -> torch.Tensor:
    """
    context x_ctx와 서버 one-hot을 붙여서
    Algorithm 1에서 쓰는 x_{tj} feature를 만든다. (Vectorized)
    """
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
    """
    하나의 컨텍스트에 대해
    argmax_k z(x, k)^T θ 를 반환한다.
    (Queue 환경, RouterBench 등 공통 inference 용도)
    """
    router.eval()
    with torch.no_grad():
        X_all = build_X_S_from_context(
            x_ctx, list(range(N_models)), N_models, device
        )
        logits = router.get_scores(X_all)  # (N_models,)
        k_hat = int(torch.argmax(logits).item())
    return k_hat
