# llm_embedding.py
from __future__ import annotations

from typing import List
import numpy as np


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if (not np.isfinite(s)) or (s <= 0.0):
        return np.ones_like(x, dtype=np.float64) / float(len(x))
    return ex / s

def select_anchors_by_model_upto_n(
    util_mat: np.ndarray,
    n_per_model: int,
    use_margin: bool,
    seed: int,
    margin_mode: str = "abs",
    tie_eps: float = 1e-9,
    mode: str = "sample",   # "sample" or "all"
) -> List[np.ndarray]:
    """
    strict winner(단독 1등) 기준으로 anchors를 만든다.
    - mode="all": 모델 k의 strict-winner pool 전부 사용
    - mode="sample": pool에서 n_per_model개 랜덤 샘플(부족하면 전부 사용)
    """
    U = np.asarray(util_mat, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={U.shape}")

    N, K = U.shape
    n = int(n_per_model)
    if n <= 0:
        n = 1

    rng = np.random.RandomState(int(seed))

    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))
    tie_size = is_top.sum(axis=1)
    strict_idx = np.where(tie_size == 1)[0].astype(np.int64)

    winners = np.argmax(U, axis=1).astype(np.int64)

    mode = str(mode).lower().strip()
    if mode not in ("sample", "all"):
        raise ValueError(f"mode must be 'sample' or 'all', got {mode}")

    anchors_by_k: List[np.ndarray] = []
    for k in range(K):
        pool = strict_idx[winners[strict_idx] == k]
        if pool.size == 0:
            anchors_by_k.append(np.zeros((0,), dtype=np.int64))
            continue

        if mode == "all":
            chosen = pool.astype(np.int64)
        else:
            # sample
            if pool.size <= n:
                chosen = pool.astype(np.int64)
            else:
                chosen = rng.choice(pool, size=n, replace=False).astype(np.int64)

        anchors_by_k.append(chosen)

    return anchors_by_k




def compute_anchor_centroids(
    X_ctx: np.ndarray,
    anchors_by_k: List[np.ndarray],
    normalize: bool,
) -> np.ndarray:
    X = np.asarray(X_ctx)
    if X.ndim != 2:
        raise ValueError(f"X_ctx must be 2D (N,d), got shape={X.shape}")

    K = len(anchors_by_k)
    d = X.shape[1]
    xi = np.zeros((K, d), dtype=np.float64)

    for k in range(K):
        idx = np.asarray(anchors_by_k[k], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"anchors_by_k[{k}] is empty. seed 구성/anchor_n_per_model 점검이 필요하다.")
        xi[k] = X[np.unique(idx)].mean(axis=0)

    if normalize:
        denom = np.linalg.norm(xi, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        xi = xi / denom

    return xi


def compute_score_matrix(
    util_mat: np.ndarray,
    anchors_by_k: List[np.ndarray],
) -> np.ndarray:
    U = np.asarray(util_mat, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"util_mat must be 2D (N,K), got shape={U.shape}")

    N, K = U.shape
    S = np.zeros((K, K), dtype=np.float64)

    for j in range(K):
        idx = np.asarray(anchors_by_k[j], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"anchors_by_k[{j}] is empty.")
        S[:, j] = U[np.unique(idx)].mean(axis=0)

    return S


def build_a_table_from_xi_S(
    xi: np.ndarray,
    S: np.ndarray,
    weight_mode: str,
    topK: int,
    tau: float,
    normalize_a: bool,
) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

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
        row = S[i, :]

        if mode == "self_centroid":
            a[i] = xi[i]
            continue

        if mode == "softmax_all":
            logits = row / tau
            w = _softmax_np(logits)
            a[i] = (w[:, None] * xi).sum(axis=0)
            continue

        if Ktop >= K:
            top_idx = np.arange(K, dtype=np.int64)
        else:
            top_idx = np.argpartition(-row, Ktop - 1)[:Ktop]

        logits = row[top_idx] / tau
        w = _softmax_np(logits)
        a[i] = (w[:, None] * xi[top_idx]).sum(axis=0)

    if normalize_a:
        denom = np.linalg.norm(a, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        a = a / denom

    return a
