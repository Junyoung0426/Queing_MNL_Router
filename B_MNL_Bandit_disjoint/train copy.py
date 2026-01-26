# B_MNL_Bandit_disjoint/train.py
from __future__ import annotations

import argparse
import json
import os
import random
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from queue_config import QueueConfig
from mnl_router import MNLRouter
from queue_env import queue_env
from b_contrastive import offline_pretrain_B_supcon


def set_full_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if bool(getattr(torch, "use_deterministic_algorithms", None)):
        torch.use_deterministic_algorithms(True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return {"torch_tensor": True, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _safe_mean(x) -> float:
    arr = np.asarray(x, dtype=np.float32)
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _cfg_to_dict(cfg: QueueConfig) -> Dict[str, Any]:
    out = {}
    for k, v in vars(cfg).items():
        if str(k).startswith("_"):
            continue
        if callable(v):
            continue
        out[str(k)] = _to_jsonable(v)

    for k in ["c1", "mean_explore_rate", "cqb_tau"]:
        try:
            if hasattr(cfg, k):
                out[k] = _to_jsonable(getattr(cfg, k))
        except Exception:
            pass

    return out


def _dump_json(path: Path, obj: Any):
    path.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp_auto")
    ap.add_argument("--lam_list", type=float, nargs="+", default=None)
    ap.add_argument("--job_pool_size", type=int, default=None)
    ap.add_argument("--use_offline_stream", action="store_true")
    ap.add_argument("--assort_K", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embedder_model", type=str, default=None)
    ap.add_argument("--b_type", type=str, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--target_explore_rate", type=float, default=None)
    ap.add_argument("--alpha_coef", type=float, default=None)
    ap.add_argument("--arrival_rate", type=float, default=None)
    return ap


def _apply_common_overrides(cfg: QueueConfig, args: argparse.Namespace):
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        cfg.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        cfg.embedder_model = str(args.embedder_model)
    if getattr(args, "b_type", None) is not None:
        cfg.b_type = str(args.b_type)
    if getattr(args, "assort_K", None) is not None:
        cfg.assort_K = int(args.assort_K)
    if getattr(args, "target_explore_rate", None) is not None:
        cfg.target_explore_rate = float(args.target_explore_rate)
    if getattr(args, "alpha_coef", None) is not None:
        cfg.alpha_coef = float(args.alpha_coef)
    if getattr(args, "arrival_rate", None) is not None:
        cfg.arrival_rate = float(args.arrival_rate)


def _ensure_prompt_column(df: pd.DataFrame):
    if "prompt" not in df.columns:
        raise ValueError("df needs prompt column")


def _dropna_required(df: pd.DataFrame, models: List[str], cost_map: Dict[str, str], use_cost: bool) -> pd.DataFrame:
    df2 = df.copy()
    if "orig_row" not in df2.columns:
        df2["orig_row"] = df2.index.to_numpy()

    need = list(models)
    if use_cost:
        need += [cost_map[m] for m in models if (m in cost_map and cost_map[m] in df2.columns)]
    need = list(dict.fromkeys(need))
    df2 = df2.dropna(subset=need).reset_index(drop=True)
    return df2


def compute_acc_cost_util_all(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    use_cost: bool,
    lam_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    acc = df[models].astype(np.float32).to_numpy()
    if not use_cost:
        return acc, acc.copy()

    N, K = acc.shape
    cost = np.zeros((N, K), dtype=np.float32)
    for j, m in enumerate(models):
        col = cost_map.get(m, None)
        if col is not None and col in df.columns:
            cost[:, j] = df[col].astype(np.float32).to_numpy()
        else:
            cost[:, j] = 0.0
    util = acc - float(lam_cost) * cost
    return acc, util


def build_offline_partition_per_model_unique(
    util_train: np.ndarray,
    per_model: int,
    seed: int,
    tie_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    U = np.asarray(util_train, dtype=np.float32)
    N, K = U.shape
    rng = np.random.RandomState(int(seed) + 999)

    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))

    chosen = set()
    offline = []

    model_order = rng.permutation(K)
    for m in model_order:
        cand = np.where(is_top[:, m])[0]
        if cand.size == 0:
            continue
        rng.shuffle(cand)
        take = 0
        for idx in cand:
            if idx in chosen:
                continue
            offline.append(int(idx))
            chosen.add(int(idx))
            take += 1
            if take >= int(per_model):
                break

    offline_idx = np.asarray(offline, dtype=np.int64)
    online_idx = np.setdiff1d(np.arange(N, dtype=np.int64), offline_idx, assume_unique=False).astype(np.int64)
    return offline_idx, online_idx


def _build_router(config: QueueConfig, d_ctx: int, n_models: int, d_proj_eff: int) -> MNLRouter:
    sig = inspect.signature(MNLRouter.__init__)

    kwargs = dict(
        d_ctx=int(d_ctx),
        n_models=int(n_models),
        d_proj=int(d_proj_eff),
        lambda_0=float(getattr(config, "lambda_0", 1.0)),
        supcon_temp=float(getattr(config, "supcon_temp", 0.07)),
        device=str(getattr(config, "device", "cpu")),
        b_type=str(getattr(config, "b_type", "linear")),
        b_hidden_mult=int(getattr(config, "b_hidden_mult", 2)),
        lbfgs_max_iter=int(getattr(config, "lbfgs_max_iter", 80)),
        lbfgs_history_size=int(getattr(config, "lbfgs_history_size", 50)),
        lbfgs_line_search=str(getattr(config, "lbfgs_line_search", "strong_wolfe")),
        hist_init_capacity=int(getattr(config, "hist_init_capacity", 2048)),
        lr_b=float(getattr(config, "offline_lr_B", 1e-3)),
    )

    if "combine_mode" in sig.parameters:
        kwargs["combine_mode"] = "mul"
    if "normalize_z" in sig.parameters:
        kwargs["normalize_z"] = bool(getattr(config, "normalize_z", True))

    return MNLRouter(**kwargs)


def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _paper_axes(ax):
    ax.set_facecolor("white")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(direction="out", width=1.0, length=4.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)


def _read_series(reg_path: Path, q_path: Path, max_steps: Optional[int], qgap_abs: bool):
    df_reg = pd.read_csv(reg_path)
    std_reg = df_reg["cum_regret"].to_numpy() if "cum_regret" in df_reg.columns else df_reg.iloc[:, 0].to_numpy()

    df_q = pd.read_csv(q_path)
    q_diff = df_q["Q_diff"].to_numpy() if "Q_diff" in df_q.columns else df_q.iloc[:, 0].to_numpy()

    L = min(len(std_reg), len(q_diff))
    if max_steps is not None:
        L = min(L, int(max_steps))

    std_reg = std_reg[:L]
    q_diff = q_diff[:L]
    if qgap_abs:
        q_diff = np.abs(q_diff)

    t = np.arange(1, L + 1)
    q_cum = np.cumsum(q_diff)
    return t, std_reg, q_diff, q_cum


def run_plotting(output_dir: Path, target_lambdas: List[float], max_steps: Optional[int] = None):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not target_lambdas:
        return

    target_lambdas = sorted(target_lambdas)
    set_paper_style()

    series_lam = {}
    for lam in target_lambdas:
        lam_tag = f"{lam:.2f}"
        reg_path = output_dir / f"regret_history_lam_{lam_tag}.csv"
        q_path = output_dir / f"Qregret_history_lam_{lam_tag}.csv"
        if not reg_path.exists() or not q_path.exists():
            continue
        t, std_reg, q_diff, q_cum = _read_series(reg_path, q_path, max_steps, qgap_abs=False)
        series_lam[lam] = {"t": t, "std_reg": std_reg, "q_diff": q_diff, "q_cum": q_cum}

    if not series_lam:
        return

    filename_suffix = f"_{max_steps}" if max_steps is not None else ""
    fig_w, fig_h = 5.0, 4.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.gca()
    for lam in sorted(series_lam.keys()):
        d = series_lam[lam]
        ax.plot(d["t"], d["std_reg"], label=rf"$\lambda={lam:.2f}$")
    ax.set_xlabel("t (time)")
    ax.set_ylabel("Cumulative Regret (lower is better)")
    _paper_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / f"standard_regret_all_lams{filename_suffix}.png")
    plt.close(fig)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.gca()
    for lam in sorted(series_lam.keys()):
        d = series_lam[lam]
        ax.plot(d["t"], d["q_diff"], label=rf"$\lambda={lam:.2f}$")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t (time)")
    ax.set_ylabel(r"$Q_r(t)-Q_o(t)$")
    _paper_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / f"queue_gap_all_lams{filename_suffix}.png")
    plt.close(fig)


def run_pipeline(df: pd.DataFrame, models: List[str], cost_map: Dict[str, str], args: argparse.Namespace, config: QueueConfig):
    _apply_common_overrides(config, args)
    config.explore_enabled = True
    b_type = str(getattr(config, "b_type", "linear")).lower().strip()
    config.b_type = b_type

    if bool(getattr(args, "deterministic", False)):
        set_full_determinism(int(getattr(config, "seed", 0)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_prompt_column(df)

    models = sorted(list(models))
    df2 = _dropna_required(df, models, cost_map, bool(getattr(config, "use_cost", True)))

    df_train, _ = train_test_split(
        df2,
        test_size=float(getattr(config, "test_size", 0.2)),
        random_state=int(getattr(config, "seed", 0)),
        shuffle=True,
    )
    df_train = df_train.reset_index(drop=True)

    device = torch.device(str(getattr(config, "device", "cpu")))

    embedder = SentenceTransformer(
        str(getattr(config, "embedder_model", "sentence-transformers/all-MiniLM-L6-v2")),
        device=str(device),
    )
    X_train = embedder.encode(
        df_train["prompt"].astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    config.d_ctx = int(X_train.shape[1])
    config.n_models = int(len(models))

    if getattr(args, "lam_list", None) is not None:
        lambdas = [float(x) for x in args.lam_list]
    else:
        if bool(getattr(config, "use_cost", True)):
            lambdas = [float(getattr(config, "lam_cost", 0.0))]
        else:
            lambdas = [0.0]

    _dump_json(output_dir / "config_full.json", _cfg_to_dict(config))
    _dump_json(output_dir / "models.json", list(models))

    for lam in lambdas:
        lam = float(lam)
        start_time = time.time()

        acc_train, util_train = compute_acc_cost_util_all(
            df_train,
            models,
            cost_map,
            use_cost=bool(getattr(config, "use_cost", True)),
            lam_cost=lam,
        )

        mode = str(getattr(config, "offline_partition_mode", "util")).lower().strip()
        N = int(len(df_train))

        if mode == "random":
            rng = np.random.RandomState(int(getattr(config, "seed", 0)) + 999)
            ratio = float(getattr(config, "offline_total_ratio", 0.10))
            n_off = int(round(ratio * N))
            n_off = max(1, min(N, n_off))
            perm = rng.permutation(N).astype(np.int64)
            offline_idx = perm[:n_off]
            online_idx = perm[n_off:]
        else:
            per_model = int(getattr(config, "offline_per_model", 5))
            tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
            if per_model <= 0:
                offline_idx = np.zeros((0,), dtype=np.int64)
                online_idx = np.arange(N, dtype=np.int64)
            else:
                offline_idx, online_idx = build_offline_partition_per_model_unique(
                    util_train=util_train,
                    per_model=per_model,
                    seed=int(getattr(config, "seed", 0)),
                    tie_eps=tie_eps,
                )

        job_pool_size = getattr(args, "job_pool_size", None)
        if job_pool_size is not None:
            rng = np.random.RandomState(int(getattr(config, "seed", 0)) + 2027)
            pool = np.asarray(online_idx, dtype=np.int64)
            n_take = min(int(job_pool_size), int(pool.size))
            stream_idx_base = rng.choice(pool, size=n_take, replace=False).astype(np.int64)
        else:
            stream_idx_base = np.asarray(online_idx, dtype=np.int64)

        if bool(getattr(args, "use_offline_stream", False)) and len(offline_idx) > 0:
            stream_idx = np.asarray(offline_idx, dtype=np.int64)
        else:
            stream_idx = np.asarray(stream_idx_base, dtype=np.int64)

        np.savez(
            output_dir / f"partition_idx_lam_{lam:.4f}.npz",
            seed_idx=np.zeros((0,), dtype=np.int64),
            rand_idx=np.zeros((0,), dtype=np.int64),
            offline_idx=np.asarray(offline_idx, dtype=np.int64),
            online_idx=np.asarray(online_idx, dtype=np.int64),
            stream_idx=np.asarray(stream_idx, dtype=np.int64),
        )

        d_ctx = int(getattr(config, "d_ctx", X_train.shape[1]))
        n_models = int(getattr(config, "n_models", len(models)))
        d_proj_eff = d_ctx if b_type == "none" else int(getattr(config, "d_proj", d_ctx))

        router = _build_router(config=config, d_ctx=d_ctx, n_models=n_models, d_proj_eff=d_proj_eff).to(device)

        if b_type != "none" and len(offline_idx) > 0:
            X_off = X_train[np.asarray(offline_idx, dtype=np.int64)]
            U_off = util_train[np.asarray(offline_idx, dtype=np.int64)]
            offline_pretrain_B_supcon(router, X_off, U_off, n_models=n_models, cfg=config)

        router.reset_for_online()

        row_ids = df_train["orig_row"].to_numpy(dtype=np.int64)[stream_idx] if "orig_row" in df_train.columns else stream_idx.copy()
        sample_ids = df_train["sample_id"].to_numpy()[stream_idx] if "sample_id" in df_train.columns else None

        router, avg_reg, Q_gap, reg_hist, Q_diff, Q_r, Q_o, expl_rate = queue_env(
            X_ctx=X_train[stream_idx],
            acc_mat=acc_train[stream_idx],
            util_mat=util_train[stream_idx],
            config=config,
            router=router,
            model_names=models,
            row_ids=row_ids,
            sample_ids=sample_ids,
        )

        elapsed_sec = time.time() - start_time

        pd.DataFrame({"cum_regret": reg_hist}).to_csv(output_dir / f"regret_history_lam_{lam:.2f}.csv", index=False)
        pd.DataFrame({"Q_diff": Q_diff, "Q_router": Q_r, "Q_oracle": Q_o}).to_csv(output_dir / f"Qregret_history_lam_{lam:.2f}.csv", index=False)

        logs = getattr(router, "_queue_env_logs", None)
        if isinstance(logs, dict):
            np.savez(output_dir / f"departure_logs_lam_{lam:.4f}.npz", **{k: _to_jsonable(v) for k, v in logs.items()})

        dep_router_mean = float("nan")
        if isinstance(logs, dict):
            dep_router_mean = _safe_mean(logs.get("dep_prob_router_hist", []))

        summary = {
            "seed": int(getattr(config, "seed", 0)),
            "lam_cost": float(lam),
            "avg_regret": float(avg_reg),
            "final_Q_gap": float(Q_gap),
            "target_explore_rate": float(getattr(config, "target_explore_rate", float("nan"))),
            "alpha_coef": float(getattr(config, "alpha_coef", float("nan"))),
            "c1": float(getattr(config, "c1", float("nan"))),
            "mean_explore_rate": float(getattr(config, "mean_explore_rate", float("nan"))),
            "cqb_tau": int(getattr(config, "cqb_tau", -1)) if hasattr(config, "cqb_tau") else -1,
            "explore_rate": float(expl_rate),
            "arrival rate": float(config.arrival_rate),
            "dep_router_mean": float(dep_router_mean),
            "n_stream": int(np.asarray(stream_idx).size),
            "offline_total": int(len(offline_idx)),
            "online_total": int(len(online_idx)),
            "use_offline_stream": bool(getattr(args, "use_offline_stream", False)),
            "b_type": str(b_type),
            "elapsed_seconds": float(elapsed_sec),
            "elapsed_minutes": float(elapsed_sec / 60.0),
        }

        _dump_json(output_dir / f"summary_lam_{lam:.4f}.json", summary)

    try:
        run_plotting(output_dir, lambdas)
    except Exception:
        pass
