#B_MNL_Bandit/llm_embedding.py
from __future__ import annotations

from typing import List
import numpy as np


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """
    안정적인 softmax (float32)
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if (not np.isfinite(s)) or (s <= 0.0):
        return np.ones_like(x, dtype=np.float32) / float(len(x))
    return ex / s


def select_anchors_by_model_upto_n(
    util_mat: np.ndarray,
    n_per_model: int,
    use_margin: bool,              # 시그니처 유지(미사용)
    seed: int,
    margin_mode: str = "abs",      # 시그니처 유지(미사용)
    tie_eps: float = 1e-9,
    mode: str = "sample",          # "sample" or "all"
) -> List[np.ndarray]:
    """
    앵커 선택 (단독 1등(strict) 제거)

    - 기본 pool: row별 최고값 집합(top-set)에 포함되는 모델이면 그 row를 해당 모델 pool에 포함
      (즉, 공동 1등 허용)

    - pool이 비는 모델이 있으면(= top-set에 한 번도 못 든 모델),
      fallback으로 "전체 row"에서 랜덤 샘플링한다.
      -> anchor pool empty를 구조적으로 방지한다.

    mode:
      - "sample": pool에서 n_per_model개를 랜덤 샘플
      - "all": pool 전체를 반환
    """
    U = np.asarray(util_mat, dtype=np.float32)
    if U.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={U.shape}")

    N, K = U.shape
    rng = np.random.RandomState(int(seed))

    if N == 0:
        return [np.zeros((0,), dtype=np.int64) for _ in range(K)]

    n_per_model = int(n_per_model)

    # top-set membership (동률 허용)
    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))  # (N,K) bool

    anchors_by_k: List[np.ndarray] = []
    all_rows = np.arange(N, dtype=np.int64)

    for k in range(K):
        pool = np.where(is_top[:, k])[0].astype(np.int64)

        # fallback: top-set이 비면 전체 row에서 뽑는다(랜덤)
        if pool.size == 0:
            raise ValueError(
                f"anchor pool empty for model k={k}. "
            )
        if str(mode).lower().strip() == "all":
            anchors_by_k.append(pool.astype(np.int64))
            continue

        if n_per_model <= 0:
            anchors_by_k.append(np.zeros((0,), dtype=np.int64))
            continue

        take = min(n_per_model, int(pool.size))
        anchors_by_k.append(rng.choice(pool, size=take, replace=False).astype(np.int64))

    return anchors_by_k


def compute_anchor_centroids(
    X_ctx: np.ndarray,
    anchors_by_k: List[np.ndarray],
    normalize: bool,
) -> np.ndarray:
    """
    X_ctx: (N,d)
    anchors_by_k[k]: row index list
    return xi: (K,d)
    """
    X = np.asarray(X_ctx, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_ctx must be 2D (N,d), got shape={X.shape}")

    K = len(anchors_by_k)
    d = int(X.shape[1])
    xi = np.zeros((K, d), dtype=np.float32)

    for k in range(K):
        idx = np.asarray(anchors_by_k[k], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"anchors_by_k[{k}] is empty. anchor_n_per_model/seed/tie_eps 점검이 필요하다.")
        idx = np.unique(idx)
        xi[k] = X[idx].mean(axis=0)

    if bool(normalize):
        denom = np.linalg.norm(xi, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        xi = xi / denom

    return xi


def compute_score_matrix(
    util_mat: np.ndarray,
    anchors_by_k: List[np.ndarray],
) -> np.ndarray:
    """
    U: (N,K)
    anchors_by_k[j]에 해당하는 row들의 평균 util을 S[:,j]로 둔다
    return S: (K,K)
    """
    U = np.asarray(util_mat, dtype=np.float32)
    if U.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={U.shape}")

    N, K = U.shape
    S = np.zeros((K, K), dtype=np.float32)

    for j in range(K):
        idx = np.asarray(anchors_by_k[j], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"anchors_by_k[{j}] is empty.")
        idx = np.unique(idx)
        S[:, j] = U[idx].mean(axis=0)

    return S


def build_a_table_from_xi_S(
    xi: np.ndarray,
    S: np.ndarray,
    weight_mode: str,
    topK: int,
    tau: float,
    normalize_a: bool,
) -> np.ndarray:
    """
    xi: (K,d) centroid
    S : (K,K) score matrix
    return a: (K,d)
    """
    xi = np.asarray(xi, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)

    if xi.ndim != 2:
        raise ValueError(f"xi must be 2D (K,d), got {xi.shape}")
    if S.ndim != 2:
        raise ValueError(f"S must be 2D (K,K), got {S.shape}")

    K, d = xi.shape
    if S.shape != (K, K):
        raise ValueError(f"S shape must be (K,K)={(K,K)}, got {S.shape}")

    mode = str(weight_mode).lower().strip()
    if mode not in ("topk_softmax", "softmax_all", "self_centroid"):
        raise ValueError(f"weight_mode must be one of topk_softmax/softmax_all/self_centroid, got {mode}")

    Ktop = min(int(topK), K)
    tau = max(float(tau), 1e-6)

    a = np.zeros((K, d), dtype=np.float32)

    for i in range(K):
        row = S[i, :]

        if mode == "self_centroid":
            a[i] = xi[i]
            continue

        if mode == "softmax_all":
            logits = row / tau
            w = _softmax_np(logits)
            a[i] = (w[:, None] * xi).sum(axis=0)
            continue

        # topk_softmax
        if Ktop >= K:
            top_idx = np.arange(K, dtype=np.int64)
        else:
            top_idx = np.argpartition(-row, Ktop - 1)[:Ktop]

        logits = row[top_idx] / tau
        w = _softmax_np(logits)
        a[i] = (w[:, None] * xi[top_idx]).sum(axis=0)

    if bool(normalize_a):
        denom = np.linalg.norm(a, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        a = a / denom

    return a
