from __future__ import annotations

import sys
import subprocess
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
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--cache_build", action="store_true")

    ap.add_argument("--supcon_uc_tau_neg", type=float, default=None)
    ap.add_argument("--supcon_uc_tau_pos", type=float, default=None)
    ap.add_argument("--supcon_uc_neg_cap", type=int, default=None)
    ap.add_argument("--supcon_temp", type=float, default=None)
    ap.add_argument("--offline_epochs", type=int, default=None)

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

    if getattr(args, "supcon_uc_tau_neg", None) is not None:
        cfg.supcon_uc_tau_neg = float(args.supcon_uc_tau_neg)
    if getattr(args, "supcon_uc_tau_pos", None) is not None:
        cfg.supcon_uc_tau_pos = float(args.supcon_uc_tau_pos)
    if getattr(args, "supcon_uc_neg_cap", None) is not None:
        cfg.supcon_uc_neg_cap = int(args.supcon_uc_neg_cap)
    if getattr(args, "supcon_temp", None) is not None:
        cfg.supcon_temp = float(args.supcon_temp)
    if getattr(args, "offline_epochs", None) is not None:
        cfg.offline_epochs = int(args.offline_epochs)



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


def _safe_npy_load(path: Path, allow_pickle: bool, mmap: bool):
    if mmap:
        try:
            return np.load(path, allow_pickle=allow_pickle, mmap_mode="r")
        except ValueError:
            return np.load(path, allow_pickle=allow_pickle)
    return np.load(path, allow_pickle=allow_pickle)


def run_pipeline(df: pd.DataFrame, models: List[str], cost_map: Dict[str, str], args: argparse.Namespace, config: QueueConfig):
    _apply_common_overrides(config, args)
    config.explore_enabled = True
    b_type = str(getattr(config, "b_type", "linear")).lower().strip()
    config.b_type = b_type

    if bool(getattr(args, "deterministic", False)):
        set_full_determinism(int(getattr(config, "seed", 0)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cost = bool(getattr(config, "use_cost", True))
    device = torch.device(str(getattr(config, "device", "cpu")))

    cache_dir = getattr(args, "cache_dir", None)
    cache_build = bool(getattr(args, "cache_build", False))

    def _cache_ready(cd: Path) -> bool:
        need = [cd / "X.npy", cd / "acc.npy", cd / "orig_row.npy", cd / "sample_id.npy", cd / "models.json", cd / "meta.json"]
        return all(p.exists() for p in need)

    def _maybe_build_cache(cd: Path, data_path: Optional[str]):
        if _cache_ready(cd):
            return
        if not cache_build:
            raise RuntimeError(f"cache missing: {cd}")
        if data_path is None:
            raise RuntimeError("cache_build requires --data in caller script")
        script = (Path(__file__).resolve().parent.parent / "tools" / "build_dataset_cache.py").resolve()
        if not script.exists():
            raise RuntimeError(f"cache builder not found: {script}")
        cmd = [
            sys.executable,
            str(script),
            "--data",
            str(Path(data_path).resolve()),
            "--cache_dir",
            str(cd),
            "--embedder_model",
            str(getattr(config, "embedder_model", "sentence-transformers/all-MiniLM-L6-v2")),
            "--device",
            str(device),
        ]
        if use_cost:
            cmd.append("--use_cost")
        subprocess.run(cmd, check=True)

    data_path = getattr(args, "data", None) if hasattr(args, "data") else None

    if cache_dir is not None:
        cd = Path(cache_dir).expanduser().resolve()
        cd.mkdir(parents=True, exist_ok=True)
        _maybe_build_cache(cd, data_path)

        X_all = _safe_npy_load(cd / "X.npy", allow_pickle=False, mmap=True)
        acc_all = _safe_npy_load(cd / "acc.npy", allow_pickle=False, mmap=True)
        orig_row_all = _safe_npy_load(cd / "orig_row.npy", allow_pickle=False, mmap=True)
        sample_id_all = _safe_npy_load(cd / "sample_id.npy", allow_pickle=True, mmap=True)
        models_cached = json.loads((cd / "models.json").read_text(encoding="utf-8"))

        models = sorted(list(models_cached))
        config.d_ctx = int(X_all.shape[1])
        config.n_models = int(len(models))

        if use_cost and (cd / "cost.npy").exists():
            cost_all = _safe_npy_load(cd / "cost.npy", allow_pickle=False, mmap=True)
        else:
            cost_all = None

        N_all = int(acc_all.shape[0])
        rng_split = int(getattr(config, "seed", 0))
        all_idx = np.arange(N_all, dtype=np.int64)

        tr_idx, _ = train_test_split(
            all_idx,
            test_size=float(getattr(config, "test_size", 0.2)),
            random_state=rng_split,
            shuffle=True,
        )
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        N = int(tr_idx.size)

        if getattr(args, "lam_list", None) is not None:
            lambdas = [float(x) for x in args.lam_list]
        else:
            lambdas = [float(getattr(config, "lam_cost", 0.0))] if use_cost else [0.0]

        _dump_json(output_dir / "config_full.json", _cfg_to_dict(config))
        _dump_json(output_dir / "models.json", list(models))

        for lam in lambdas:
            lam = float(lam)
            print(f"\n>>> Running lambda={lam:.4f} <<<")
            start_time = time.time()

            acc_train_full = np.asarray(acc_all[tr_idx], dtype=np.float32)
            if use_cost and (cost_all is not None):
                cost_train_full = np.asarray(cost_all[tr_idx], dtype=np.float32)
            else:
                cost_train_full = np.zeros_like(acc_train_full, dtype=np.float32)

            util_train_full = acc_train_full - float(lam) * cost_train_full

            mode = str(getattr(config, "offline_partition_mode", "util")).lower().strip()
            all_local = np.arange(N, dtype=np.int64)

            job_pool_size = getattr(args, "job_pool_size", None)
            if job_pool_size is not None:
                seed0 = int(getattr(config, "seed", 0))
                n_take = min(int(job_pool_size), N)
                df_tmp = pd.DataFrame(index=np.arange(N, dtype=np.int64))
                df_pool = df_tmp.sample(n=n_take, random_state=seed0)
                online_pool_local = df_pool.index.to_numpy(dtype=np.int64)
            else:
                online_pool_local = all_local.copy()

            remain_local = np.setdiff1d(all_local, online_pool_local, assume_unique=False).astype(np.int64)

            seed_idx = np.zeros((0,), dtype=np.int64)
            rand_idx = np.zeros((0,), dtype=np.int64)

            if remain_local.size == 0:
                offline_local = np.zeros((0,), dtype=np.int64)
            else:
                if mode == "random":
                    rng = np.random.RandomState(int(getattr(config, "seed", 0)) + 999)
                    ratio = float(getattr(config, "offline_total_ratio", 0.10))
                    n_off = int(round(ratio * int(remain_local.size)))
                    n_off = max(1, min(int(remain_local.size), n_off))
                    perm = rng.permutation(int(remain_local.size)).astype(np.int64)
                    offline_local = remain_local[perm[:n_off]].astype(np.int64)
                elif mode == "strict":
                    n_per_model = int(getattr(config, "offline_per_model", 5))
                    tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                    s_l, r_l, off_l, _ = build_offline_partition_strict_only_per_model(
                        util_train=util_train_full[remain_local],
                        n_per_model=n_per_model,
                        tie_eps=tie_eps,
                        seed=int(getattr(config, "seed", 0)),
                    )
                    seed_idx = remain_local[np.asarray(s_l, dtype=np.int64)] if int(len(s_l)) > 0 else np.zeros((0,), dtype=np.int64)
                    rand_idx = remain_local[np.asarray(r_l, dtype=np.int64)] if int(len(r_l)) > 0 else np.zeros((0,), dtype=np.int64)
                    offline_local = remain_local[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)
                elif mode == "mincover":
                    ratio = float(getattr(config, "offline_total_ratio", 0.10))
                    min_per = int(getattr(config, "offline_seed_min_per_model", 0))
                    tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                    s_l, r_l, off_l, _ = build_offline_partition_mincover_then_random(
                        util_train=util_train_full[remain_local],
                        ratio=ratio,
                        min_per=min_per,
                        tie_eps=tie_eps,
                        seed=int(getattr(config, "seed", 0)),
                    )
                    seed_idx = remain_local[np.asarray(s_l, dtype=np.int64)] if int(len(s_l)) > 0 else np.zeros((0,), dtype=np.int64)
                    rand_idx = remain_local[np.asarray(r_l, dtype=np.int64)] if int(len(r_l)) > 0 else np.zeros((0,), dtype=np.int64)
                    offline_local = remain_local[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)
                else:
                    per_model = int(getattr(config, "offline_per_model", 5))
                    tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                    if per_model <= 0:
                        offline_local = np.zeros((0,), dtype=np.int64)
                    else:
                        off_l, _ = build_offline_partition_per_model_unique(
                            util_train=util_train_full[remain_local],
                            per_model=per_model,
                            seed=int(getattr(config, "seed", 0)),
                            tie_eps=tie_eps,
                        )
                        offline_local = remain_local[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)

            online_local = np.asarray(online_pool_local, dtype=np.int64)

            if bool(getattr(args, "use_offline_stream", False)) and int(len(offline_local)) > 0:
                stream_local = np.asarray(offline_local, dtype=np.int64)
            else:
                stream_local = np.asarray(online_local, dtype=np.int64)

            print(f"[Split] N_train={N} job_pool={int(len(online_local))} remain={int(len(remain_local))} offline={int(len(offline_local))} stream={int(len(stream_local))} mode={mode} b_type={b_type}")

            offline_global = tr_idx[np.asarray(offline_local, dtype=np.int64)] if int(len(offline_local)) > 0 else np.zeros((0,), dtype=np.int64)
            online_global = tr_idx[np.asarray(online_local, dtype=np.int64)]
            stream_global = tr_idx[np.asarray(stream_local, dtype=np.int64)]

            np.savez(
                output_dir / f"partition_idx_lam_{lam:.4f}.npz",
                seed_idx=np.asarray(seed_idx, dtype=np.int64),
                rand_idx=np.asarray(rand_idx, dtype=np.int64),
                offline_idx=np.asarray(offline_local, dtype=np.int64),
                online_idx=np.asarray(online_local, dtype=np.int64),
                stream_idx=np.asarray(stream_local, dtype=np.int64),
                offline_idx_global=np.asarray(offline_global, dtype=np.int64),
                online_idx_global=np.asarray(online_global, dtype=np.int64),
                stream_idx_global=np.asarray(stream_global, dtype=np.int64),
                train_idx_global=np.asarray(tr_idx, dtype=np.int64),
            )

            d_ctx = int(getattr(config, "d_ctx", int(X_all.shape[1])))
            n_models = int(getattr(config, "n_models", len(models)))
            d_proj_eff = d_ctx if b_type == "none" else int(getattr(config, "d_proj", d_ctx))

            router = _build_router(config=config, d_ctx=d_ctx, n_models=n_models, d_proj_eff=d_proj_eff).to(device)

            if b_type != "none" and int(len(offline_local)) > 0:
                X_off = np.asarray(X_all[offline_global], dtype=np.float32)
                U_off = np.asarray(util_train_full[offline_local], dtype=np.float32)
                print(f"[Offline] Pretraining B (n={int(len(offline_local))})")
                offline_pretrain_B_supcon(router, X_off, U_off, n_models=n_models, cfg=config)
            else:
                print("[Offline] skip B pretrain")

            print(f"[Online] Bandit stream size={int(stream_local.size)}")
            router.reset_for_online()

            row_ids = np.asarray(orig_row_all[stream_global], dtype=np.int64)
            sample_ids = np.asarray(sample_id_all[stream_global])

            router, avg_reg, Q_gap, reg_hist, Q_diff, Q_r, Q_o, expl_rate = queue_env(
                X_ctx=np.asarray(X_all[stream_global], dtype=np.float32),
                acc_mat=np.asarray(acc_train_full[stream_local], dtype=np.float32),
                util_mat=np.asarray(util_train_full[stream_local], dtype=np.float32),
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
                "n_stream": int(stream_local.size),
                "offline_total": int(len(offline_local)),
                "online_total": int(len(online_local)),
                "use_offline_stream": bool(getattr(args, "use_offline_stream", False)),
                "b_type": str(b_type),
                "offline_partition_mode": str(mode),
                "cache_dir": str(cd),
                "elapsed_seconds": float(elapsed_sec),
                "elapsed_minutes": float(elapsed_sec / 60.0),
            }

            _dump_json(output_dir / f"summary_lam_{lam:.4f}.json", summary)

            print(f"[Done] lam={lam:.4f} avg_reg={avg_reg:.6f} Q_gap={Q_gap:.3f} dep_mean={dep_router_mean:.6f} time={elapsed_sec:.1f}s")

        try:
            run_plotting(output_dir, lambdas)
        except Exception as e:
            print(f"[Error] Failed to generate plots: {e}")
        return

    _ensure_prompt_column(df)

    models = sorted(list(models))
    df2 = _dropna_required(df, models, cost_map, use_cost)

    df_train, _ = train_test_split(
        df2,
        test_size=float(getattr(config, "test_size", 0.2)),
        random_state=int(getattr(config, "seed", 0)),
        shuffle=True,
    )
    df_train = df_train.reset_index(drop=True)

    embedder = SentenceTransformer(str(getattr(config, "embedder_model", "sentence-transformers/all-MiniLM-L6-v2")), device=str(device))
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
        lambdas = [float(getattr(config, "lam_cost", 0.0))] if use_cost else [0.0]

    _dump_json(output_dir / "config_full.json", _cfg_to_dict(config))
    _dump_json(output_dir / "models.json", list(models))

    for lam in lambdas:
        lam = float(lam)
        print(f"\n>>> Running lambda={lam:.4f} <<<")
        start_time = time.time()

        acc_train, util_train = compute_acc_cost_util_all(
            df_train,
            models,
            cost_map,
            use_cost=use_cost,
            lam_cost=lam,
        )

        mode = str(getattr(config, "offline_partition_mode", "util")).lower().strip()
        N = int(len(df_train))
        all_idx = np.arange(N, dtype=np.int64)

        job_pool_size = getattr(args, "job_pool_size", None)
        if job_pool_size is not None:
            seed0 = int(getattr(config, "seed", 0))
            n_take = min(int(job_pool_size), N)
            df_pool = df_train.sample(n=n_take, random_state=seed0)
            online_pool_idx = df_pool.index.to_numpy(dtype=np.int64)
        else:
            online_pool_idx = all_idx.copy()

        remain_idx = np.setdiff1d(all_idx, online_pool_idx, assume_unique=False).astype(np.int64)

        seed_idx = np.zeros((0,), dtype=np.int64)
        rand_idx = np.zeros((0,), dtype=np.int64)

        if remain_idx.size == 0:
            offline_idx = np.zeros((0,), dtype=np.int64)
        else:
            if mode == "random":
                rng = np.random.RandomState(int(getattr(config, "seed", 0)) + 999)
                ratio = float(getattr(config, "offline_total_ratio", 0.10))
                n_off = int(round(ratio * int(remain_idx.size)))
                n_off = max(1, min(int(remain_idx.size), n_off))
                perm = rng.permutation(int(remain_idx.size)).astype(np.int64)
                offline_idx = remain_idx[perm[:n_off]].astype(np.int64)
            elif mode == "strict":
                n_per_model = int(getattr(config, "offline_per_model", 5))
                tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                s_l, r_l, off_l, _ = build_offline_partition_strict_only_per_model(
                    util_train=util_train[remain_idx],
                    n_per_model=n_per_model,
                    tie_eps=tie_eps,
                    seed=int(getattr(config, "seed", 0)),
                )
                seed_idx = remain_idx[np.asarray(s_l, dtype=np.int64)] if int(len(s_l)) > 0 else np.zeros((0,), dtype=np.int64)
                rand_idx = remain_idx[np.asarray(r_l, dtype=np.int64)] if int(len(r_l)) > 0 else np.zeros((0,), dtype=np.int64)
                offline_idx = remain_idx[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)
            elif mode == "mincover":
                ratio = float(getattr(config, "offline_total_ratio", 0.10))
                min_per = int(getattr(config, "offline_seed_min_per_model", 0))
                tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                s_l, r_l, off_l, _ = build_offline_partition_mincover_then_random(
                    util_train=util_train[remain_idx],
                    ratio=ratio,
                    min_per=min_per,
                    tie_eps=tie_eps,
                    seed=int(getattr(config, "seed", 0)),
                )
                seed_idx = remain_idx[np.asarray(s_l, dtype=np.int64)] if int(len(s_l)) > 0 else np.zeros((0,), dtype=np.int64)
                rand_idx = remain_idx[np.asarray(r_l, dtype=np.int64)] if int(len(r_l)) > 0 else np.zeros((0,), dtype=np.int64)
                offline_idx = remain_idx[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)
            else:
                per_model = int(getattr(config, "offline_per_model", 5))
                tie_eps = float(getattr(config, "offline_tie_eps", 1e-9))
                if per_model <= 0:
                    offline_idx = np.zeros((0,), dtype=np.int64)
                else:
                    off_l, _ = build_offline_partition_per_model_unique(
                        util_train=util_train[remain_idx],
                        per_model=per_model,
                        seed=int(getattr(config, "seed", 0)),
                        tie_eps=tie_eps,
                    )
                    offline_idx = remain_idx[np.asarray(off_l, dtype=np.int64)] if int(len(off_l)) > 0 else np.zeros((0,), dtype=np.int64)

        online_idx = np.asarray(online_pool_idx, dtype=np.int64)

        if bool(getattr(args, "use_offline_stream", False)) and int(len(offline_idx)) > 0:
            stream_idx = np.asarray(offline_idx, dtype=np.int64)
        else:
            stream_idx = np.asarray(online_idx, dtype=np.int64)

        np.savez(
            output_dir / f"partition_idx_lam_{lam:.4f}.npz",
            seed_idx=np.asarray(seed_idx, dtype=np.int64),
            rand_idx=np.asarray(rand_idx, dtype=np.int64),
            offline_idx=np.asarray(offline_idx, dtype=np.int64),
            online_idx=np.asarray(online_idx, dtype=np.int64),
            stream_idx=np.asarray(stream_idx, dtype=np.int64),
        )

        d_ctx = int(getattr(config, "d_ctx", X_train.shape[1]))
        n_models = int(getattr(config, "n_models", len(models)))
        d_proj_eff = d_ctx if b_type == "none" else int(getattr(config, "d_proj", d_ctx))

        router = _build_router(config=config, d_ctx=d_ctx, n_models=n_models, d_proj_eff=d_proj_eff).to(device)

        if b_type != "none" and int(len(offline_idx)) > 0:
            X_off = X_train[np.asarray(offline_idx, dtype=np.int64)]
            U_off = util_train[np.asarray(offline_idx, dtype=np.int64)]
            print(f"[Offline] Pretraining B (n={int(len(offline_idx))})")
            offline_pretrain_B_supcon(router, X_off, U_off, n_models=n_models, cfg=config)
        else:
            print("[Offline] skip B pretrain")

        print(f"[Online] Bandit stream size={int(stream_idx.size)}")
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
            "n_stream": int(stream_idx.size),
            "offline_total": int(len(offline_idx)),
            "online_total": int(len(online_idx)),
            "use_offline_stream": bool(getattr(args, "use_offline_stream", False)),
            "b_type": str(b_type),
            "offline_partition_mode": str(mode),
            "elapsed_seconds": float(elapsed_sec),
            "elapsed_minutes": float(elapsed_sec / 60.0),
        }

        _dump_json(output_dir / f"summary_lam_{lam:.4f}.json", summary)

        print(f"[Done] lam={lam:.4f} avg_reg={avg_reg:.6f} Q_gap={Q_gap:.3f} dep_mean={dep_router_mean:.6f} time={elapsed_sec:.1f}s")

    try:
        run_plotting(output_dir, lambdas)
    except Exception as e:
        print(f"[Error] Failed to generate plots: {e}")
