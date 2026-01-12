# train.py
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    select_anchors_by_model_upto_n,
    compute_anchor_centroids,
    compute_score_matrix,
    build_a_table_from_xi_S,
)


# ============================================================
# JSON save helpers
# ============================================================
def _try_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.device):
        return str(x)
    if isinstance(x, torch.dtype):
        return str(x)
    if torch.is_tensor(x):
        return {"torch_tensor": True, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in list(x)]
    return str(x)


def _cfg_to_dict(cfg: QueueConfig) -> Dict[str, Any]:
    if is_dataclass(cfg):
        raw = dict(asdict(cfg))
    else:
        raw = dict(getattr(cfg, "__dict__", {}))
        for k in dir(cfg):
            if k.startswith("_") or k in raw:
                continue
            try:
                v = getattr(cfg, k)
            except Exception:
                continue
            if callable(v):
                continue
            raw[k] = v

    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        if callable(v):
            continue
        out[str(k)] = _to_jsonable(v)
    return out


def _dump_json(path: Path, obj: Any):
    path.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_env_cfg(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    patterns = (
        "arrival", "queue", "assort",
        "explore", "c1", "alpha",
        "eps", "steps", "lam", "lambda", "cost",
        "odds", "r_", "seed",
        "offline", "tie", "anchor", "supcon", "b_type", "d_proj",
        "embed_", "weight_mode", "tau", "topk",
    )
    keep: Dict[str, Any] = {}
    for k, v in cfg_dict.items():
        lk = str(k).lower()
        if any(p in lk for p in patterns):
            keep[k] = v
    return keep


# ============================================================
# CLI
# ============================================================
def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp1")
    ap.add_argument("--lam_list", type=float, nargs="+", default=None)
    ap.add_argument("--job_pool_size", type=int, default=None)
    ap.add_argument("--use_offline_stream", action="store_true")

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embedder_model", type=str, default=None)
    return ap


def _apply_common_overrides(cfg: QueueConfig, args: argparse.Namespace):
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        cfg.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        cfg.embedder_model = str(args.embedder_model)


def _ensure_prompt_column(df: pd.DataFrame):
    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다.")


def _dropna_required(df, models, cost_map, use_cost):
    need = list(models)
    if use_cost:
        need += [cost_map[m] for m in models if m in cost_map]
    need = list(dict.fromkeys(need))
    return df.dropna(subset=need).reset_index(drop=True)


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
            cost[:, j] = 0.0
    util = acc - float(lam_cost) * cost
    return acc, util


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
# Offline split 
# ============================================================
def build_offline_partition_strict_only_per_model(
    util_train: np.ndarray,
    n_per_model: int,
    tie_eps: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    b_type='none'용:
    - offline = 모델별 strict winner pool에서 최대 n_per_model개까지(부족하면 있는 것만)
    - 중복 샘플링 금지
    - online = 나머지 전부
    """
    U = np.asarray(util_train, dtype=np.float64)
    N, K = U.shape
    rng = np.random.RandomState(int(seed))

    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))
    tie_size = is_top.sum(axis=1)
    strict_idx = np.where(tie_size == 1)[0].astype(np.int64)
    winners = np.argmax(U, axis=1).astype(np.int64)

    chosen_all = []
    for k in range(K):
        pool = strict_idx[winners[strict_idx] == k]
        if pool.size == 0:
            # strict winner가 아예 없으면 이 모델은 offline seed가 0개가 된다
            continue
        take = min(int(n_per_model), int(pool.size))
        chosen = rng.choice(pool, size=take, replace=False).astype(np.int64)
        chosen_all.append(chosen)

    offline_idx = np.unique(np.concatenate(chosen_all)) if chosen_all else np.zeros((0,), dtype=np.int64)
    offline_idx = offline_idx.astype(np.int64)

    seed_idx = offline_idx.copy()
    rand_idx = np.zeros((0,), dtype=np.int64)

    all_idx = np.arange(N, dtype=np.int64)
    online_idx = np.setdiff1d(all_idx, offline_idx, assume_unique=False).astype(np.int64)
    return seed_idx, rand_idx, offline_idx, online_idx

def build_offline_partition_mincover_then_random(
    util_train: np.ndarray,
    offline_total_ratio: float,
    min_per_model: int,
    tie_eps: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    b_type!='none'용 (strict-only min-cover):
    - offline_total = round(ratio*N)
    - 먼저 각 모델 k의 strict winner pool에서 min_per_model개까지(부족하면 있는 것만) 확보
    - 남은 offline 자리는 전체에서 랜덤으로 채운다
    """
    U = np.asarray(util_train, dtype=np.float64)
    N, K = U.shape
    rng = np.random.RandomState(int(seed) + 999)

    offline_total = int(round(float(offline_total_ratio) * N))
    offline_total = max(offline_total, 1)
    offline_total = min(offline_total, N)

    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))
    tie_size = is_top.sum(axis=1)
    strict_idx = np.where(tie_size == 1)[0].astype(np.int64)
    winners = np.argmax(U, axis=1).astype(np.int64)

    seed_list = []
    for k in range(K):
        pool = strict_idx[winners[strict_idx] == k]
        if pool.size == 0:
            continue
        take = min(int(min_per_model), int(pool.size))
        chosen = rng.choice(pool, size=take, replace=False).astype(np.int64)
        seed_list.append(chosen)

    seed_idx = np.unique(np.concatenate(seed_list)) if seed_list else np.zeros((0,), dtype=np.int64)
    seed_idx = seed_idx.astype(np.int64)

    all_idx = np.arange(N, dtype=np.int64)
    remain = np.setdiff1d(all_idx, seed_idx, assume_unique=False).astype(np.int64)

    n_rand = int(offline_total - seed_idx.size)
    if n_rand > 0:
        # remain이 부족하면 replace=True로 채우는 대신, 그냥 remain 전부만 쓰는 편이 더 “중복 없음” 정책에 맞다
        take = min(n_rand, int(remain.size))
        rand_idx = rng.choice(remain, size=take, replace=False).astype(np.int64)
    else:
        rand_idx = np.zeros((0,), dtype=np.int64)

    offline_idx = np.concatenate([seed_idx, rand_idx]).astype(np.int64)
    offline_idx = rng.permutation(offline_idx).astype(np.int64)

    online_idx = np.setdiff1d(all_idx, offline_idx, assume_unique=False).astype(np.int64)
    return seed_idx, rand_idx, offline_idx, online_idx



# ============================================================
# run_pipeline
# ============================================================
def run_pipeline(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    args: argparse.Namespace,
    config: QueueConfig,
):
    _apply_common_overrides(config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_prompt_column(df)

    models = sorted(list(models))
    df = _dropna_required(df, models, cost_map, use_cost=bool(config.use_cost))

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

    embedder = Embedder(str(config.embedder_model), device)
    X_train = embedder.transform(df_train["prompt"].astype(str).tolist())
    d_ctx = int(X_train.shape[1])

    config.d_ctx = d_ctx
    config.n_models = int(len(models))

    lambdas = [float(x) for x in args.lam_list] if args.lam_list is not None else ([float(config.lam_cost)] if bool(config.use_cost) else [0.0])

    cfg_full = _cfg_to_dict(config)
    env_cfg = _filter_env_cfg(cfg_full)

    _dump_json(output_dir / "run_args.json", vars(args))
    _dump_json(output_dir / "config_full.json", cfg_full)
    _dump_json(output_dir / "env_cfg.json", env_cfg)
    _dump_json(output_dir / "models.json", list(models))
    _dump_json(output_dir / "cost_map.json", {k: str(v) for k, v in cost_map.items()})
    _dump_json(
        output_dir / "run_meta.json",
        {
            "time_local": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None),
            "git_commit": _try_git_commit(),
            "n_after_dropna": int(df.shape[0]),
            "n_train": int(df_train.shape[0]),
            "d_ctx": int(d_ctx),
            "n_models": int(len(models)),
            "use_cost": bool(config.use_cost),
            "lambdas": list(map(float, lambdas)),
        },
    )

    btype = str(config.b_type).lower().strip()

    # b_type==none이면 무조건 all
    if btype == "none":
        anchor_mode = "all"
    else:
        anchor_mode = str(getattr(config, "ak_anchor_mode", "sample")).lower().strip()


    for lam in lambdas:
        lam = float(lam)

        acc_train, util_train = compute_acc_cost_util_all(
            df_train,
            models=models,
            cost_map=cost_map,
            use_cost=bool(config.use_cost),
            lam_cost=lam,
        )

        # -------- offline split --------
        if btype == "none":
            seed_idx, rand_idx, offline_idx, online_idx = build_offline_partition_strict_only_per_model(
                util_train=util_train,
                n_per_model=int(config.offline_seed_min_per_model),
                tie_eps=float(config.offline_tie_eps),
                seed=int(config.seed) + 999,
            )
        else:
            seed_idx, rand_idx, offline_idx, online_idx = build_offline_partition_mincover_then_random(
                util_train=util_train,
                offline_total_ratio=float(config.offline_total_ratio),
                min_per_model=int(config.offline_seed_min_per_model),
                tie_eps=float(config.offline_tie_eps),
                seed=int(config.seed),
            )

        print(f"[Partition] offline={len(offline_idx)} (seed={len(seed_idx)}, rand={len(rand_idx)}) online={len(online_idx)}")

        # tie stats (전체 train 기준)
        U = np.asarray(util_train, dtype=np.float64)
        mx = U.max(axis=1, keepdims=True)
        is_top = (U >= (mx - float(config.offline_tie_eps)))
        tie_size = is_top.sum(axis=1)
        print(f"[Tie] strict_ratio(tie==1)={float((tie_size==1).mean()):.4f} tie_mean={float(tie_size.mean()):.3f} tie_max={int(tie_size.max())}")

        stream_idx = offline_idx if bool(args.use_offline_stream) else online_idx

        # stats dump
        np.savez(
            output_dir / f"partition_idx_lam_{lam:.4f}.npz",
            seed_idx=np.asarray(seed_idx, dtype=np.int64),
            rand_idx=np.asarray(rand_idx, dtype=np.int64),
            offline_idx=np.asarray(offline_idx, dtype=np.int64),
            online_idx=np.asarray(online_idx, dtype=np.int64),
            stream_idx=np.asarray(stream_idx, dtype=np.int64),
        )
        _dump_json(
            output_dir / f"stats_lam_{lam:.4f}.json",
            {
                "lam_cost": float(lam),
                "offline_total": int(len(np.asarray(offline_idx))),
                "offline_seed": int(len(np.asarray(seed_idx))),
                "offline_rand": int(len(np.asarray(rand_idx))),
                "online": int(len(np.asarray(online_idx))),
                "stream": int(len(np.asarray(stream_idx))),
                "b_type": btype,
                "supcon_pos_strategy": str(config.supcon_pos_strategy),
                "strict_ratio_tie_eq_1": float((tie_size == 1).mean()),
                "tie_mean": float(tie_size.mean()),
                "tie_max": int(tie_size.max()),
            },
        )

        # -------- Router init --------
        d_proj_eff = d_ctx if btype == "none" else int(config.d_proj)
        router = MNLRouter(
            d_ctx=d_ctx,
            n_models=int(len(models)),
            d_proj=d_proj_eff,
            combine_mode=str(config.combine_mode),
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

        X_off = X_train[np.asarray(offline_idx, dtype=np.int64)]
        U_off = util_train[np.asarray(offline_idx, dtype=np.int64)]

        # -------- (1) B pretrain --------
        if btype == "none":
            print("[Offline] skip B pretrain (b_type=none).")
        else:
            print(f"[Offline] B pretrain uses OFFLINE full (pos_strategy={str(config.supcon_pos_strategy)}).")
            offline_pretrain_B_supcon(router, X_off, U_off, n_models=int(len(models)), cfg=config)

        # -------- (2) a_table build  --------
        router.eval()
        with torch.no_grad():
            Z_off = (
                router.B(torch.from_numpy(X_off).float().to(router.dev))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        anchors_by_k = select_anchors_by_model_upto_n(
            util_mat=U_off,                          
            n_per_model=int(config.anchor_n_per_model),
            use_margin=False,
            seed=int(config.seed) + int(config.anchor_seed_offset),
            margin_mode="abs",
            tie_eps=float(config.offline_tie_eps),
            mode=anchor_mode,
        )


        empty = [k for k, idx in enumerate(anchors_by_k) if len(idx) == 0]
        if len(empty) > 0:
            raise ValueError(
                f"anchor pool empty models={empty}. "
                f"(offline_total_ratio/lam_cost/offline_tie_eps/anchor_n_per_model) 조정이 필요하다."
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

        # -------- Online --------
        router.reset_for_online()

        X_stream = X_train[np.asarray(stream_idx, dtype=np.int64)]
        acc_stream = acc_train[np.asarray(stream_idx, dtype=np.int64)]
        util_stream = util_train[np.asarray(stream_idx, dtype=np.int64)]

        router, avg_reg, Q_reg_T, reg_hist, Q_diff_hist, Q_r_hist, Q_o_hist, explore_rate = queue_env(
            X_ctx=X_stream,
            acc_mat=acc_stream,
            util_mat=util_stream,
            config=config,
            router=router,
            model_names=models,
        )

        pd.DataFrame({"cum_regret": reg_hist}).to_csv(output_dir / f"regret_history_lam_{lam:.2f}.csv", index=False)
        pd.DataFrame({"Q_diff": Q_diff_hist, "Q_router": Q_r_hist, "Q_oracle": Q_o_hist}).to_csv(
            output_dir / f"Qregret_history_lam_{lam:.2f}.csv", index=False
        )

        summary = {
            "lam_cost": float(lam),
            "avg_regret": float(avg_reg),
            "final_Q_gap": float(Q_reg_T),
            "n_train": int(df_train.shape[0]),
            "n_stream": int(len(np.asarray(stream_idx))),
            "use_offline_stream": bool(args.use_offline_stream),
            "offline_total": int(len(np.asarray(offline_idx))),
            "offline_seed": int(len(np.asarray(seed_idx))),
            "offline_rand": int(len(np.asarray(rand_idx))),
            "b_type": str(config.b_type),
            "supcon_pos_strategy": str(config.supcon_pos_strategy),
            "explore_rate": float(explore_rate),
            "cnt_decision": int(getattr(router, "_T", 0)),
        }

        (output_dir / f"summary_lam_{lam:.4f}.json").write_text(
            json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[Done] lam={lam:.4f} avg_reg={avg_reg:.6f} Q_gap={Q_reg_T:.3f}")
