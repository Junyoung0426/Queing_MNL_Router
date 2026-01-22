import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAM_RE = re.compile(r"regret_history_lam_([0-9]+\.[0-9]+)\.csv$")
QLAM_RE = re.compile(r"Qregret_history_lam_([0-9]+\.[0-9]+)\.csv$")


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

        "lines.linewidth": 2.0,

        "axes.grid": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_args():
    ap = argparse.ArgumentParser(description="Plot regrets for multiple algorithms under a root directory")
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--lambdas", type=float, nargs="*", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--include", type=str, nargs="*", default=None)
    ap.add_argument("--exclude", type=str, nargs="*", default=["plots"])
    ap.add_argument("--qgap_abs", action="store_true")
    ap.add_argument("--fig_w", type=float, default=6.0)
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


def collect_runs(root_dir: Path, include, exclude):
    alg_dirs = []
    excl = set(exclude or [])
    incl = set(include) if include is not None else None

    for p in root_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name in excl:
            continue
        if incl is not None and p.name not in incl:
            continue
        alg_dirs.append(p)

    if not alg_dirs:
        raise RuntimeError(f"no algorithm dirs found under: {root_dir}")

    alg_data = {}
    for alg_dir in sorted(alg_dirs, key=lambda x: x.name):
        reg_files = list(alg_dir.rglob("regret_history_lam_*.csv"))
        q_files = list(alg_dir.rglob("Qregret_history_lam_*.csv"))

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
        if not common_lams:
            continue

        alg_runs = {}
        for lam in common_lams:
            reg_path = _pick_latest_by_mtime(reg_map[lam])
            q_path = _pick_latest_by_mtime(q_map[lam])
            alg_runs[lam] = {"reg_path": reg_path, "q_path": q_path}

        alg_data[alg_dir.name] = alg_runs

    if not alg_data:
        raise RuntimeError(f"no valid regret/Qregret csv pairs found under: {root_dir}")

    return alg_data


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

    rounds = np.arange(1, L + 1)
    q_cum = np.cumsum(q_diff)
    return rounds, std_reg, q_diff, q_cum


def _paper_axes(ax):
    ax.set_facecolor("white")

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(direction="out", width=1.0, length=4.0)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)


def _rename_for_legend(name: str) -> str:
    s = str(name)

    if s == "AQCB-CL":
        return "AQCB-CL(ours)"

    # use function replacement to avoid re replacement backslash parsing
    s = re.sub(r"(?i)_eps\b", lambda _: r"-$\epsilon$", s)
    s = re.sub(r"(?i)\beps\b", lambda _: r"$\epsilon$", s)

    return s


def _rename_for_filename(name: str) -> str:
    s = str(name)
    s = re.sub(r"(?i)_eps\b", lambda _: "ε", s)
    s = re.sub(r"(?i)\beps\b", lambda _: "ε", s)
    return s



def _bold_legend_label(ax, target_label: str):
    leg = ax.get_legend()
    if leg is None:
        return
    for txt in leg.get_texts():
        if txt.get_text() == target_label:
            txt.set_fontweight("bold")


def main():
    args = parse_args()
    set_paper_style()

    root_dir = Path(args.root_dir)
    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    alg_data = collect_runs(root_dir=root_dir, include=args.include, exclude=args.exclude)

    if args.lambdas is not None and len(args.lambdas) > 0:
        target_lambdas = sorted([float(x) for x in args.lambdas])
    else:
        lam_set = set()
        for _, runs in alg_data.items():
            lam_set |= set(runs.keys())
        target_lambdas = sorted(lam_set)

    if not target_lambdas:
        raise RuntimeError("no lambdas found")

    filename_suffix = ""
    if args.max_steps is not None:
        filename_suffix = f"_{int(args.max_steps)}"
    if args.qgap_abs:
        filename_suffix += "_absQ"

    for lam in target_lambdas:
        series = {}
        for alg, runs in alg_data.items():
            if lam not in runs:
                continue
            reg_path = runs[lam]["reg_path"]
            q_path = runs[lam]["q_path"]
            rounds, std_reg, q_diff, q_cum = read_series(
                reg_path=reg_path,
                q_path=q_path,
                max_steps=args.max_steps,
                qgap_abs=bool(args.qgap_abs),
            )
            series[alg] = {"rounds": rounds, "std_reg": std_reg, "q_diff": q_diff, "q_cum": q_cum}

        if not series:
            print(f"[Warn] lambda={lam} has no data across algorithms, skipping")
            continue

        alg_names = sorted(series.keys())
        lam_tag = f"{lam:.2f}"

        if 0 < len(alg_names) <= 3:
            prefix_str = "_vs_".join(_rename_for_filename(a) for a in alg_names)
        else:
            prefix_str = "ALLALG"

        # 1) Standard regret
        fig = plt.figure(figsize=(args.fig_w, args.fig_h))
        ax = fig.gca()
        for alg in alg_names:
            d = series[alg]
            label = _rename_for_legend(alg)
            ax.plot(d["rounds"], d["std_reg"], label=label)
        ax.set_xlabel("t (time)")
        ax.set_ylabel("Cumulative Regret (lower is better)")
        _paper_axes(ax)
        ax.legend(loc="upper left")
        _bold_legend_label(ax, "AQCB-CL(ours)")
        fig.tight_layout()
        out_std = plots_dir / f"{prefix_str}_standard_regret_lam_{lam_tag}{filename_suffix}.png"
        fig.savefig(out_std)
        plt.close(fig)
        print(f"[Saved] {out_std}")

        # 2) Instantaneous Q-gap
        fig = plt.figure(figsize=(args.fig_w, args.fig_h))
        ax = fig.gca()
        for alg in alg_names:
            d = series[alg]
            label = _rename_for_legend(alg)
            ax.plot(d["rounds"], d["q_diff"], label=label)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("t (time)")
        ax.set_ylabel(r"$|Q_r(t)-Q_o(t)|$" if args.qgap_abs else r"$Q_r(t)-Q_o(t)$")
        _paper_axes(ax)
        ax.legend(loc="upper left")
        _bold_legend_label(ax, "AQCB-CL(ours)")
        fig.tight_layout()
        out_q = plots_dir / f"{prefix_str}_queue_gap_lam_{lam_tag}{filename_suffix}.png"
        fig.savefig(out_q)
        plt.close(fig)
        print(f"[Saved] {out_q}")

        # 3) Cumulative Q-gap
        fig = plt.figure(figsize=(args.fig_w, args.fig_h))
        ax = fig.gca()
        for alg in alg_names:
            d = series[alg]
            label = _rename_for_legend(alg)
            ax.plot(d["rounds"], d["q_cum"], label=label)
        ax.set_xlabel("t (time)")
        ax.set_ylabel(r"Cumulative $|Q_r(s)-Q_o(s)|$" if args.qgap_abs else r"Cumulative $(Q_r(s)-Q_o(s))$")
        _paper_axes(ax)
        ax.legend(loc="upper left")
        _bold_legend_label(ax, "AQCB-CL(ours)")
        fig.tight_layout()
        out_qc = plots_dir / f"{prefix_str}_queue_cum_lam_{lam_tag}{filename_suffix}.png"
        fig.savefig(out_qc)
        plt.close(fig)
        print(f"[Saved] {out_qc}")

    print("\n[Done] All plots generated.")


if __name__ == "__main__":
    main()
