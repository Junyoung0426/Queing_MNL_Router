# B_MNL_Bandit_disjoint/train.py
from __future__ import annotations

import argparse
import json
import os
import random
import inspect
import re  # Added for potential regex needs, though train.py usually knows its lambdas
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# --- [Plotting Libs] ---
import matplotlib
matplotlib.use("Agg") # Must be before importing pyplot for server environments
import matplotlib.pyplot as plt
# -----------------------

from queue_config import QueueConfig
from mnl_router import MNLRouter
from queue_env import queue_env
from b_contrastive import offline_pretrain_B_supcon


# ----------------------------
# Determinism (optional)
# ----------------------------
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


# ----------------------------
# JSON helpers
# ----------------------------
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


# ----------------------------
# CLI
# ----------------------------
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



# B_MNL_Bandit_disjoint/train.py

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



# ----------------------------
# Data helpers
# ----------------------------
def _ensure_prompt_column(df: pd.DataFrame):
    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다")


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


# ----------------------------
# Partitioning
# ----------------------------
def build_offline_partition_strict_only_per_model(
    util_train: np.ndarray, n_per_model: int, tie_eps: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U = np.asarray(util_train, dtype=np.float32)
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
            continue
        take = min(int(n_per_model), int(pool.size))
        chosen_all.append(rng.choice(pool, size=take, replace=False).astype(np.int64))

    seed_idx = np.unique(np.concatenate(chosen_all)) if chosen_all else np.zeros((0,), dtype=np.int64)
    rand_idx = np.zeros((0,), dtype=np.int64)
    offline_idx = seed_idx.copy()
    online_idx = np.setdiff1d(np.arange(N, dtype=np.int64), offline_idx, assume_unique=False).astype(np.int64)
    return seed_idx, rand_idx, offline_idx, online_idx


def build_offline_partition_mincover_then_random(
    util_train: np.ndarray, ratio: float, min_per: int, tie_eps: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U = np.asarray(util_train, dtype=np.float32)
    N, K = U.shape
    rng = np.random.RandomState(int(seed) + 999)

    n_off = int(round(float(ratio) * N))
    n_off = max(1, n_off)
    n_off = min(N, n_off)

    mx = U.max(axis=1, keepdims=True)
    is_top = (U >= (mx - float(tie_eps)))
    tie_size = is_top.sum(axis=1)
    strict_idx = np.where(tie_size == 1)[0].astype(np.int64)
    winners = np.argmax(U, axis=1).astype(np.int64)

    seed_list = []
    if int(min_per) > 0:
        for k in range(K):
            pool = strict_idx[winners[strict_idx] == k]
            if pool.size == 0:
                continue
            take = min(int(min_per), int(pool.size))
            seed_list.append(rng.choice(pool, size=take, replace=False).astype(np.int64))

    seed_idx = np.unique(np.concatenate(seed_list)) if seed_list else np.zeros((0,), dtype=np.int64)

    all_idx = np.arange(N, dtype=np.int64)
    remain = np.setdiff1d(all_idx, seed_idx, assume_unique=False).astype(np.int64)

    n_rand = int(n_off - seed_idx.size)
    if n_rand > 0 and remain.size > 0:
        take = min(n_rand, int(remain.size))
        rand_idx = rng.choice(remain, size=take, replace=False).astype(np.int64)
    else:
        rand_idx = np.zeros((0,), dtype=np.int64)

    offline_idx = np.concatenate([seed_idx, rand_idx]).astype(np.int64)
    offline_idx = rng.permutation(offline_idx).astype(np.int64)

    online_idx = np.setdiff1d(all_idx, offline_idx, assume_unique=False).astype(np.int64)
    return seed_idx, rand_idx, offline_idx, online_idx


# ----------------------------
# Router builder
# ----------------------------
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


# ----------------------------
# PLOTTING FUNCTION (Integrated with Paper Style)
# ----------------------------
def set_paper_style():
    """Sets matplotlib params for paper-quality plots."""
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
    """Applies specific spine and grid styling."""
    ax.set_facecolor("white")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(direction="out", width=1.0, length=4.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)


def _read_series(reg_path: Path, q_path: Path, max_steps: Optional[int], qgap_abs: bool):
    """Reads and truncates regret and queue data series."""
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
    """
    Plots Standard Regret and Queue Gap for the given lambdas using paper style.
    """
    print(f"\n[Plotting] Generating plots in {output_dir}...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not target_lambdas:
        print("[Warn] No lambdas to plot.")
        return

    target_lambdas = sorted(target_lambdas)
    print(f"[Plotting] Target Lambdas: {target_lambdas}")
    
    set_paper_style()
    
    series_lam = {}

    # 1. Load Data
    for lam in target_lambdas:
        lam_tag = f"{lam:.2f}"
        reg_path = output_dir / f"regret_history_lam_{lam_tag}.csv"
        q_path = output_dir / f"Qregret_history_lam_{lam_tag}.csv"

        if not reg_path.exists() or not q_path.exists():
            print(f"[Warn] Missing files for lambda={lam}, skipping plotting.")
            continue
        
        # Read data
        t, std_reg, q_diff, q_cum = _read_series(reg_path, q_path, max_steps, qgap_abs=False)
        series_lam[lam] = {"t": t, "std_reg": std_reg, "q_diff": q_diff, "q_cum": q_cum}

    if not series_lam:
        print("[Error] No valid data loaded for plotting.")
        return

    filename_suffix = ""
    if max_steps is not None:
        filename_suffix = f"_{max_steps}"

    # Default figure size from provided script
    fig_w, fig_h = 5.0, 4.0

    # 2. Standard Regret Plot
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
    
    out_std = plots_dir / f"standard_regret_all_lams{filename_suffix}.png"
    fig.savefig(out_std)
    plt.close(fig)
    print(f"[Saved] {out_std}")

    # 3. Queue Gap Plot (Q_r - Q_o)
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
    
    out_q = plots_dir / f"queue_gap_all_lams{filename_suffix}.png"
    fig.savefig(out_q)
    plt.close(fig)
    print(f"[Saved] {out_q}")

    print("[Plotting] All plots generated successfully.\n")


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(df: pd.DataFrame, models: List[str], cost_map: Dict[str, str], args: argparse.Namespace, config: QueueConfig):
    _apply_common_overrides(config, args)
    config.explore_enabled = True
    b_type = str(getattr(config, "b_type", "linear")).lower().strip()
    config.b_type = b_type

    if bool(getattr(args, "deterministic", False)):
        set_full_determinism(int(getattr(config, "seed", 0)))

    if b_type == "none":
        print("\n[Auto-Config] b_type='none' -> Offline split off(Pure Online), B pretrain skip")
        config.offline_total_ratio = 0.0
        config.offline_seed_min_per_model = 0
    else:
        print(f"\n[Auto-Config] b_type='{b_type}' -> Offline split on (ratio={getattr(config, 'offline_total_ratio', None)})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_prompt_column(df)

    models = sorted(list(models))

    df2 = _dropna_required(df, models, cost_map, bool(getattr(config, "use_cost", True)))

    df_train, _ = train_test_split(df2, test_size=float(getattr(config, "test_size", 0.2)), random_state=int(getattr(config, "seed", 0)), shuffle=True)
    df_train = df_train.reset_index(drop=True)

    if getattr(args, "job_pool_size", None) is not None:
        df_train = df_train.sample(n=int(args.job_pool_size), random_state=int(getattr(config, "seed", 0))).reset_index(drop=True)

    device = torch.device(str(getattr(config, "device", "cpu")))

    # embedder
    embedder = SentenceTransformer(str(getattr(config, "embedder_model", "sentence-transformers/all-MiniLM-L6-v2")), device=str(device))
    X_train = embedder.encode(
        df_train["prompt"].astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    config.d_ctx = int(X_train.shape[1])
    config.n_models = int(len(models))

    # lambda list
    if getattr(args, "lam_list", None) is not None:
        lambdas = [float(x) for x in args.lam_list]
    else:
        if bool(getattr(config, "use_cost", True)):
            lambdas = [float(getattr(config, "lam_cost", 0.0))]
        else:
            lambdas = [0.0]

    # save meta
    _dump_json(output_dir / "config_full.json", _cfg_to_dict(config))
    _dump_json(output_dir / "models.json", list(models))

    for lam in lambdas:
        lam = float(lam)
        print(f"\n>>> Running lambda={lam:.4f} <<<")
        start_time = time.time()
        acc_train, util_train = compute_acc_cost_util_all(
            df_train, models, cost_map,
            use_cost=bool(getattr(config, "use_cost", True)),
            lam_cost=lam
        )

        # Partitioning
        ratio = float(getattr(config, "offline_total_ratio", 0.0))
        if ratio <= 0.0:
            seed_idx = np.zeros((0,), dtype=np.int64)
            rand_idx = np.zeros((0,), dtype=np.int64)
            offline_idx = np.zeros((0,), dtype=np.int64)
            online_idx = np.arange(len(df_train), dtype=np.int64)
        else:
            seed_idx, rand_idx, offline_idx, online_idx = build_offline_partition_mincover_then_random(
                util_train=util_train,
                ratio=ratio,
                min_per=int(getattr(config, "offline_seed_min_per_model", 0)),
                tie_eps=float(getattr(config, "offline_tie_eps", 0.0)),
                seed=int(getattr(config, "seed", 0)),
            )

        print(f"[Partition] offline={len(offline_idx)} online={len(online_idx)}")

        # stream
        if bool(getattr(args, "use_offline_stream", False)) and len(offline_idx) > 0:
            stream_idx = offline_idx
        else:
            stream_idx = online_idx

        stream_idx = np.asarray(stream_idx, dtype=np.int64)

        # Save indices
        np.savez(
            output_dir / f"partition_idx_lam_{lam:.4f}.npz",
            seed_idx=seed_idx,
            rand_idx=rand_idx,
            offline_idx=offline_idx,
            online_idx=online_idx,
            stream_idx=stream_idx,
        )

        # Router init
        d_ctx = int(getattr(config, "d_ctx", X_train.shape[1]))
        n_models = int(getattr(config, "n_models", len(models)))
        d_proj_eff = d_ctx if b_type == "none" else int(getattr(config, "d_proj", d_ctx))

        router = _build_router(config=config, d_ctx=d_ctx, n_models=n_models, d_proj_eff=d_proj_eff).to(device)

        # Offline Pretrain (B only)
        if b_type != "none" and len(offline_idx) > 0:
            X_off = X_train[np.asarray(offline_idx, dtype=np.int64)]
            U_off = util_train[np.asarray(offline_idx, dtype=np.int64)]
            print(f"[Offline] Pretraining B (n={len(offline_idx)})")
            offline_pretrain_B_supcon(router, X_off, U_off, n_models=n_models, cfg=config)
        else:
            print("[Offline] skip B pretrain")

        # Online Bandit
        print(f"[Online] Bandit stream size={int(stream_idx.size)}")
        router.reset_for_online()

        # row/sample id mapping
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
        end_time = time.time()
        elapsed_sec = end_time - start_time

        # Logging
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
            "n_stream": int(stream_idx.size),
            "offline_total": int(len(offline_idx)),
            "online_total": int(len(online_idx)),
            "use_offline_stream": bool(getattr(args, "use_offline_stream", False)),
            "b_type": str(b_type),
            "elapsed_seconds": float(elapsed_sec),
            "elapsed_minutes": float(elapsed_sec / 60.0),
        }

        _dump_json(output_dir / f"summary_lam_{lam:.4f}.json", summary)

        print(f"[Done] lam={lam:.4f} avg_reg={avg_reg:.6f} Q_gap={Q_gap:.3f} dep_mean={dep_router_mean:.6f} time={elapsed_sec:.1f}s")

    # --- [Run Plotting] ---
    try:
        run_plotting(output_dir, lambdas)
    except Exception as e:
        print(f"[Error] Failed to generate plots: {e}")
    # ----------------------