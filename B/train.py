#train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from queue_config import QueueConfig
from mnl_router import MNLRouter
from queue_env import queue_env
from b_contrastive import offline_pretrain_B_supcon

from llm_embedding import (
    compute_anchor_centroids,
    compute_score_matrix,
    build_a_table_from_xi_S,
)


# ============================================================
# CLI: common args
# ============================================================
def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp1")
    ap.add_argument("--lam_list", type=float, nargs="+", default=None)
    ap.add_argument("--job_pool_size", type=int, default=None)
    ap.add_argument("--use_offline_stream", action="store_true")

    # optional overrides (필요한 것만 최소로)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embedder_model", type=str, default=None)

    return ap


# ============================================================
# Helpers (dataset-agnostic)
# ============================================================
def _apply_common_overrides(cfg: QueueConfig, args: argparse.Namespace):
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        cfg.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        cfg.embedder_model = str(args.embedder_model)


def _ensure_prompt_column(df: pd.DataFrame):
    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다. 데이터 로더에서 prompt 컬럼을 만들어서 넘겨야 한다.")


def _dropna_required(df, models, cost_map, use_cost):
    need = list(models)         
    if use_cost:
        need += [cost_map[m] for m in models if m in cost_map]
    need = list(dict.fromkeys(need))
    return df.dropna(subset=need).reset_index(drop=True)


def _load_or_lock_model_index(output_dir: Path, models: List[str]) -> List[str]:
    """
    output_dir/model_index.json이 존재하면 그 순서를 강제한다.
    없으면 현재 models로 생성한다.
    """
    model_index_path = output_dir / "model_index.json"
    if model_index_path.exists():
        obj = json.loads(model_index_path.read_text(encoding="utf-8"))
        fixed = obj.get("models", None)
        if not isinstance(fixed, list) or len(fixed) == 0:
            raise ValueError(f"model_index.json이 있으나 models 리스트가 비정상이다: {model_index_path}")
        fixed = [str(m) for m in fixed]
        missing = [m for m in fixed if m not in models]
        extra = [m for m in models if m not in fixed]
        if missing:
            raise ValueError(f"model_index.json 기준으로 df에 없는 모델이 있다: missing={missing}")
        if extra:
            # 실험 재현성 위해 extra도 에러로 막는 편이 안전하다
            raise ValueError(f"model_index.json에 없는 추가 모델이 df에 있다: extra={extra}")
        return fixed

    model2idx = {m: i for i, m in enumerate(models)}
    idx2model = {i: m for m, i in model2idx.items()}
    model_index_path.write_text(
        json.dumps({"models": models, "model2idx": model2idx, "idx2model": idx2model}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return models


def compute_acc_cost_util_all(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    use_cost: bool,
    lam_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    acc = df[models].astype(np.float64).to_numpy()
    if not use_cost:
        return acc, acc.copy()

    N, K = acc.shape
    cost = np.zeros((N, K), dtype=np.float64)
    for j, m in enumerate(models):
        col = cost_map.get(m, None)
        if col is not None and col in df.columns:
            cost[:, j] = df[col].astype(np.float64).to_numpy()
        else:
            cost[:, j] = 0.0   # 노트북 정책
    util = acc - float(lam_cost) * cost
    return acc, util


def _sample_winner_balanced(
    winners: np.ndarray,
    candidate_idx: np.ndarray,
    K: int,
    n_total: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if n_total <= 0:
        return np.zeros((0,), dtype=np.int64)

    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)

    base = n_total // K
    rem = n_total % K
    order = rng.permutation(K)
    need = np.full(K, base, dtype=int)
    need[order[:rem]] += 1

    chosen_list: List[np.ndarray] = []
    chosen_set = set()
    shortage = 0

    for k in range(K):
        pool = candidate_idx[winners[candidate_idx] == k]
        if pool.size >= need[k]:
            sel = rng.choice(pool, size=need[k], replace=False)
        else:
            sel = pool
            shortage += (need[k] - pool.size)

        if sel.size > 0:
            chosen_list.append(sel)
            for v in sel.tolist():
                chosen_set.add(int(v))

    chosen = np.concatenate(chosen_list).astype(np.int64) if chosen_list else np.zeros((0,), dtype=np.int64)

    if shortage > 0:
        mask = np.array([int(i) not in chosen_set for i in candidate_idx.tolist()], dtype=bool)
        rest = candidate_idx[mask]
        if rest.size >= shortage:
            extra = rng.choice(rest, size=shortage, replace=False)
        else:
            extra = rng.choice(candidate_idx, size=shortage, replace=True)
        chosen = np.concatenate([chosen, extra.astype(np.int64)])

    if chosen.size > n_total:
        chosen = chosen[:n_total]
    return chosen.astype(np.int64)


def build_offline_partition_7030(util_train: np.ndarray, cfg: QueueConfig):
    """
    cfg.offline_total_ratio 만큼 offline(TB)로 뽑고,
    TB 안에서 winner-balanced(tb_win_frac) + random(tb_rand_frac) 섞는다.
    """
    N, K = util_train.shape
    rng = np.random.RandomState(int(cfg.seed) + 999)
    winners = np.argmax(util_train, axis=1).astype(np.int64)

    offline_total = int(round(float(cfg.offline_total_ratio) * N))
    offline_total = max(offline_total, K)
    offline_total = min(offline_total, max(2, N - 2))

    win_frac = float(cfg.tb_win_frac)
    rand_frac = float(cfg.tb_rand_frac)
    if abs((win_frac + rand_frac) - 1.0) > 1e-6:
        s = win_frac + rand_frac
        win_frac /= s
        rand_frac /= s

    n_win = int(round(win_frac * offline_total))
    n_win = max(n_win, K)
    n_win = min(n_win, offline_total)
    n_rand = offline_total - n_win

    candidate = np.arange(N, dtype=np.int64)
    win_idx = _sample_winner_balanced(winners, candidate, K, n_win, rng)
    remain = np.setdiff1d(candidate, win_idx, assume_unique=False)

    if n_rand > 0:
        if remain.size < n_rand:
            raise RuntimeError(f"random sampling 부족: need={n_rand}, have={remain.size}")
        rand_idx = rng.choice(remain, size=n_rand, replace=False).astype(np.int64)
    else:
        rand_idx = np.zeros((0,), dtype=np.int64)

    offline_idx = np.concatenate([win_idx, rand_idx]).astype(np.int64)
    offline_idx = rng.permutation(offline_idx)
    online_idx = np.setdiff1d(candidate, offline_idx, assume_unique=False).astype(np.int64)
    return win_idx, rand_idx, offline_idx, online_idx


def _safe_anchors_by_model(
    util_mat: np.ndarray,
    n_per_model: int,
    use_margin: bool,
    margin_mode: str,
    seed: int,
) -> List[np.ndarray]:
    """
    winner가 0개인 모델이 있어도 죽지 않게 anchors를 만든다.
    """
    rng = np.random.RandomState(int(seed))
    N, K = util_mat.shape
    n = max(1, int(n_per_model))

    winners = np.argmax(util_mat, axis=1).astype(np.int64)
    anchors_by_k: List[np.ndarray] = []

    margin_mode = str(margin_mode).lower().strip()

    for k in range(K):
        idxs = np.where(winners == k)[0].astype(np.int64)
        if idxs.size == 0:
            idxs = np.arange(N, dtype=np.int64)

        if use_margin:
            if margin_mode == "abs":
                score = util_mat[idxs, k]
            else:
                u_sub = util_mat[idxs]
                tmp = u_sub.copy()
                tmp[:, k] = -np.inf
                max_other = np.max(tmp, axis=1)
                score = u_sub[:, k] - max_other

            order = np.argsort(-score)
            chosen = idxs[order[:n]]
            if chosen.size < n:
                extra = rng.choice(idxs, size=(n - chosen.size), replace=True)
                chosen = np.concatenate([chosen, extra])
        else:
            chosen = rng.choice(idxs, size=n, replace=(idxs.size < n))

        anchors_by_k.append(chosen.astype(np.int64))

    return anchors_by_k


class Embedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = str(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] '{model_name}' on {self.device}")

    def transform(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embs.astype(np.float32)


# ============================================================
# Public: run_pipeline
# ============================================================
def run_pipeline(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    args: argparse.Namespace,
    config: QueueConfig,
):
    """
    dataset별 스크립트(routerbench.py, mixinstruct.py 등)에서 호출하는 공통 파이프라인이다.
    - df: 최소 'prompt' + model score 컬럼(+ optional cost 컬럼)
    - models: score 컬럼 리스트(모델명)
    - cost_map: model -> cost column name
    """
    _apply_common_overrides(config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_prompt_column(df)

    # 모델 순서 고정(재현성)
    # models = _load_or_lock_model_index(output_dir, list(models))
    models = sorted(list(models))
    # 결측 제거
    df = _dropna_required(df, models, cost_map, use_cost=bool(config.use_cost))

    # train split
    df_train, _ = train_test_split(
        df,
        test_size=float(config.test_size),
        random_state=int(config.seed),
        shuffle=True,
    )
    df_train = df_train.reset_index(drop=True)

    if getattr(args, "job_pool_size", None) is not None:
        df_train = df_train.sample(n=int(args.job_pool_size), random_state=int(config.seed)).reset_index(drop=True)

    device = torch.device(str(config.device))

    # embedding
    embedder = Embedder(str(config.embedder_model), device)
    X_train = embedder.transform(df_train["prompt"].astype(str).tolist())
    d_ctx = int(X_train.shape[1])

    config.d_ctx = d_ctx
    config.n_models = int(len(models))

    # lambdas
    if args.lam_list is not None:
        lambdas = [float(x) for x in args.lam_list]
    else:
        lambdas = [float(config.lam_cost)] if bool(config.use_cost) else [0.0]

    # 실험 기록(옵션)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps({k: getattr(config, k) for k in dir(config) if not k.startswith("_") and isinstance(getattr(config, k), (int, float, str, bool))},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for lam in lambdas:
        lam = float(lam)

        acc_train, util_train = compute_acc_cost_util_all(
            df_train,
            models=models,
            cost_map=cost_map,
            use_cost=bool(config.use_cost),
            lam_cost=lam,
        )

        win_idx, rand_idx, offline_idx, online_idx = build_offline_partition_7030(util_train, config)
        print(f"[Partition] TB(off)={len(offline_idx)}  Train\\TB={len(online_idx)}")

        # router init
        d_proj_eff = d_ctx if str(config.b_type).lower() == "none" else int(config.d_proj)
        router = MNLRouter(
            d_ctx=d_ctx,
            n_models=int(len(models)),
            d_proj=d_proj_eff,
            lambda_0=float(config.lambda_0),
            supcon_temp=float(config.supcon_temp),
            device=str(config.device),
            b_type=str(config.b_type),
            b_hidden_mult=int(config.b_hidden_mult),
            theta_solver="lbfgs",
            lbfgs_max_iter=int(config.lbfgs_max_iter),
            lbfgs_history_size=int(config.lbfgs_history_size),
            lbfgs_line_search=str(config.lbfgs_line_search),
            hist_init_capacity=int(config.hist_init_capacity),
            lr_b=float(config.offline_lr_B),
        ).to(device)

        # offline pool
        X_off = X_train[offline_idx]
        U_off = util_train[offline_idx]

        # (1) B pretrain
        offline_pretrain_B_supcon(router, X_off, U_off, n_models=int(len(models)), cfg=config)

        # (2) a_table build
        router.eval()
        with torch.no_grad():
            Z_off = router.B(torch.from_numpy(X_off).float().to(router.dev)).detach().cpu().numpy().astype(np.float32)

        anchors_by_k = _safe_anchors_by_model(
            util_mat=U_off,
            n_per_model=int(config.anchor_n_per_model),
            use_margin=bool(config.anchor_use_margin),
            margin_mode=str(config.anchor_margin_mode),
            seed=int(config.seed) + int(config.anchor_seed_offset),
        )

        xi = compute_anchor_centroids(Z_off, anchors_by_k, normalize=bool(config.anchor_xi_normalize))
        S = compute_score_matrix(U_off, anchors_by_k)
        a_table = build_a_table_from_xi_S(
            xi=xi,
            S=S,
            weight_mode=str(config.embed_weight_mode),
            topK=int(config.embed_topK),
            tau=float(config.embed_tau),
            normalize_a=bool(config.embed_a_normalize),
        ).astype(np.float32)

        np.save(output_dir / f"a_table_lam_{lam:.4f}.npy", a_table)
        router.set_a_table(torch.from_numpy(a_table), normalize=False)

        # online
        router.reset_for_online()

        stream_idx = offline_idx if bool(args.use_offline_stream) else online_idx
        X_stream = X_train[stream_idx]
        acc_stream = acc_train[stream_idx]
        util_stream = util_train[stream_idx]


        router, avg_reg, Q_reg_T, reg_hist, Q_diff_hist, Q_r_hist, Q_o_hist, explore_rate = queue_env(
            X_ctx=X_stream,
            acc_mat=acc_stream,
            util_mat=util_stream,
            config=config,
            router=router,
            model_names=models,
        )
        pd.DataFrame({"cum_regret": reg_hist}).to_csv(
        output_dir / f"regret_history_lam_{lam:.2f}.csv",
        index=False,)
        # [수정됨] Q_diff, Q_router, Q_oracle 모두 저장
        pd.DataFrame({
            "Q_diff": Q_diff_hist,
            "Q_router": Q_r_hist,
            "Q_oracle": Q_o_hist
        }).to_csv(
            output_dir / f"Qregret_history_lam_{lam:.2f}.csv",
            index=False,
        )

        summary = {
            "lam_cost": lam,
            "avg_regret": float(avg_reg),
            "final_Q_gap": float(Q_reg_T),
            "n_train": int(df_train.shape[0]),
            "n_stream": int(X_stream.shape[0]),
            "use_offline_stream": bool(args.use_offline_stream),
            "offline_total": int(len(offline_idx)),
            "offline_rand": int(len(rand_idx)),
            "supcon_pos_strategy": str(config.supcon_pos_strategy),
            "ak_build_use_seed_only": bool(getattr(config, "ak_build_use_seed_only", True)),
            "explore_rate": float(explore_rate),
            "cnt_decision": int(getattr(router, "_T", 0)),
        }

        print(f"[Done] lam={lam:.4f} avg_reg={avg_reg:.6f} Q_gap={Q_reg_T:.3f}")