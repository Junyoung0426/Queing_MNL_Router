import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import subprocess

LAM_RE = re.compile(r"regret_history_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")
QLAM_RE = re.compile(r"Qregret_history_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")


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


def parse_args():
    ap = argparse.ArgumentParser(description="Overlay lambdas in a single directory")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory containing regret_history_lam_*.csv and Qregret_history_lam_*.csv")
    ap.add_argument("--lambdas", type=float, nargs="*", default=None, help="Optional list of lambdas to plot")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional maximum number of steps to plot")
    ap.add_argument("--qgap_abs", action="store_true", help="Use absolute Q gap")
    ap.add_argument("--fig_w", type=float, default=5.0)
    ap.add_argument("--fig_h", type=float, default=4.0)
    return ap.parse_args()


def _parse_lambda_from_name(name: str, is_q: bool):
    m = (QLAM_RE.match(name) if is_q else LAM_RE.match(name))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _pick_latest_by_mtime(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def collect_pairs(output_dir: Path):
    reg_files = list(output_dir.rglob("regret_history_lam_*.csv"))
    q_files = list(output_dir.rglob("Qregret_history_lam_*.csv"))

    reg_map = {}
    q_map = {}

    for f in reg_files:
        lam = _parse_lambda_from_name(f.name, is_q=False)
        if lam is None:
            continue
        reg_map.setdefault(lam, []).append(f)

    for f in q_files:
        lam = _parse_lambda_from_name(f.name, is_q=True)
        if lam is None:
            continue
        q_map.setdefault(lam, []).append(f)

    common_lams = sorted(set(reg_map.keys()) & set(q_map.keys()))
    pairs = {}
    for lam in common_lams:
        pairs[lam] = {
            "reg_path": _pick_latest_by_mtime(reg_map[lam]),
            "q_path": _pick_latest_by_mtime(q_map[lam]),
        }

    return pairs


def read_series(reg_path: Path, q_path: Path, max_steps, qgap_abs: bool):
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


def main():
    args = parse_args()
    set_paper_style()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(output_dir)
    if not pairs:
        print(f"[Error] no valid regret/Qregret pairs found under: {output_dir}")
        return

    if args.lambdas is not None and len(args.lambdas) > 0:
        target_lambdas = sorted([float(x) for x in args.lambdas if float(x) in pairs])
    else:
        target_lambdas = sorted(pairs.keys())

    if not target_lambdas:
        print("[Error] no lambdas selected/found")
        return

    filename_suffix = ""
    if args.max_steps is not None:
        filename_suffix = f"_{int(args.max_steps)}"
    if args.qgap_abs:
        filename_suffix += "_absQ"

    series_lam = {}
    for lam in target_lambdas:
        reg_path = pairs[lam]["reg_path"]
        q_path = pairs[lam]["q_path"]
        t, std_reg, q_diff, q_cum = read_series(reg_path, q_path, args.max_steps, bool(args.qgap_abs))
        series_lam[lam] = {"t": t, "std_reg": std_reg, "q_diff": q_diff, "q_cum": q_cum}

    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    ax = fig.gca()
    for lam in target_lambdas:
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

    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    ax = fig.gca()
    for lam in target_lambdas:
        d = series_lam[lam]
        ax.plot(d["t"], d["q_diff"], label=rf"$\lambda={lam:.2f}$")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t (time)")
    ax.set_ylabel(r"$Q_r(t)-Q_o(t)$" if args.qgap_abs else r"$Q_r(t)-Q_o(t)$")
    _paper_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_q = plots_dir / f"queue_gap_all_lams{filename_suffix}.png"
    fig.savefig(out_q)
    plt.close(fig)
    print(f"[Saved] {out_q}")

    print("\n[Done] All plots generated.")


if __name__ == "__main__":
    main()
