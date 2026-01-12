# llm_embedding.py
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if (not np.isfinite(s)) or (s <= 0.0):
        return np.ones_like(x, dtype=np.float64) / float(len(x))
    return ex / s


def select_anchors_by_model(
    util_mat: np.ndarray,
    n_per_model: int,
    use_margin: bool,
    seed: int,
    margin_mode: str = "abs",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    각 모델 k에 대해 'winner(=argmax util)'인 샘플들 중 n개를 anchor로 뽑는다.

    util_mat: (N, K)
    return:
      anchors_by_k: list length K, each is (n,) idx array
      anchor_idx_all: (K*n,) 전체 anchor 인덱스
    """
    util_mat = np.asarray(util_mat)
    if util_mat.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={util_mat.shape}")

    N, K = util_mat.shape
    n = int(n_per_model)
    if n <= 0:
        raise ValueError(f"n_per_model must be > 0, got {n}")

    margin_mode = str(margin_mode).lower().strip()
    if margin_mode not in ("abs", "gap"):
        raise ValueError(f"margin_mode must be 'abs' or 'gap', got {margin_mode}")

    winners = np.argmax(util_mat, axis=1).astype(np.int64)
    counts = np.bincount(winners, minlength=K).astype(int)

    less = np.where(counts < n)[0].tolist()
    if len(less) > 0:
        raise ValueError(
            f"[Anchor] insufficient winners for n={n}. "
            f"counts_by_model={counts.tolist()}, less_models={less}. "
            f"Reduce anchor_n_per_model / change utility / increase dataset."
        )

    rng = np.random.RandomState(int(seed))
    anchors_by_k: List[np.ndarray] = []
    anchor_all: List[int] = []

    for k in range(K):
        idxs = np.where(winners == k)[0].astype(np.int64)

        if use_margin:
            if margin_mode == "abs":
                score = util_mat[idxs, k]
            else:
                u_sub = util_mat[idxs]  # (n_winner, K)
                tmp = u_sub.copy()
                tmp[:, k] = -np.inf
                max_other = np.max(tmp, axis=1)
                score = u_sub[:, k] - max_other

            order = np.argsort(-score)
            chosen = idxs[order[:n]]
        else:
            chosen = rng.choice(idxs, size=n, replace=False).astype(np.int64)

        anchors_by_k.append(chosen)
        anchor_all.extend([int(i) for i in chosen.tolist()])

    anchor_idx_all = np.array(anchor_all, dtype=np.int64)
    return anchors_by_k, anchor_idx_all


def compute_anchor_centroids(
    X_ctx: np.ndarray,
    anchors_by_k: List[np.ndarray],
    normalize: bool,
) -> np.ndarray:
    """
    각 모델 k의 anchor들의 평균을 centroid xi[k]로 만든다.

    X_ctx: (N, d)
    anchors_by_k: list length K
    return xi: (K, d)
    """
    X_ctx = np.asarray(X_ctx)
    if X_ctx.ndim != 2:
        raise ValueError(f"X_ctx must be 2D (N,d), got shape={X_ctx.shape}")

    N, d = X_ctx.shape
    K = len(anchors_by_k)
    if K <= 0:
        raise ValueError("anchors_by_k must be non-empty")

    xi = np.zeros((K, d), dtype=np.float64)
    for j in range(K):
        idx_j = np.asarray(anchors_by_k[j], dtype=np.int64)
        if idx_j.size == 0:
            raise ValueError(f"anchors_by_k[{j}] is empty")
        xi[j] = X_ctx[idx_j].mean(axis=0)

    if normalize:
        denom = np.linalg.norm(xi, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        xi = xi / denom

    return xi


def compute_score_matrix(
    util_mat: np.ndarray,
    anchors_by_k: List[np.ndarray],
) -> np.ndarray:
    """
    S[:, j] = model j의 anchor들에서 평균 util 벡터(모든 모델에 대한 mean util)

    util_mat: (N,K)
    return S: (K,K)
    """
    util_mat = np.asarray(util_mat)
    if util_mat.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={util_mat.shape}")

    N, K = util_mat.shape
    S = np.zeros((K, K), dtype=np.float64)

    for j in range(K):
        idx_j = np.asarray(anchors_by_k[j], dtype=np.int64)
        if idx_j.size == 0:
            S[:, j] = 0.0
        else:
            S[:, j] = util_mat[idx_j].mean(axis=0)

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
    a_i = sum_j w_{i,j} * xi_j  (score matrix S로 weight 생성)

    xi: (K,d)
    S:  (K,K)
    return a: (K,d)
    """
    xi = np.asarray(xi)
    S = np.asarray(S)

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

    a = np.zeros((K, d), dtype=np.float64)

    for i in range(K):
        row = S[i, :]  # (K,)

        if mode == "self_centroid":
            a[i] = xi[i]
            continue

        if mode == "softmax_all":
            logits = row / tau
            w = _softmax_np(logits)           # (K,)
            a[i] = (w[:, None] * xi).sum(axis=0)
            continue

        # mode == "topk_softmax"
        if Ktop >= K:
            top_idx = np.arange(K, dtype=np.int64)
        else:
            top_idx = np.argpartition(-row, Ktop - 1)[:Ktop]

        logits = row[top_idx] / tau
        w = _softmax_np(logits)              # (Ktop,)
        a[i] = (w[:, None] * xi[top_idx]).sum(axis=0)

    if normalize_a:
        denom = np.linalg.norm(a, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        a = a / denom

    return a
