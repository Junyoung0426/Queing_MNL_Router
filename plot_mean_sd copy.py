import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REG_AVG_RE = re.compile(r"regret_mean_sd_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")
Q_AVG_RE = re.compile(r"qgap_mean_sd_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")


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


def parse_args():
    ap = argparse.ArgumentParser(description="Plot mean+sd (from avg folder)")
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--lambdas", type=float, nargs="*", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--include", type=str, nargs="*", default=None)
    ap.add_argument("--exclude", type=str, nargs="*", default=["plots"])
    ap.add_argument("--fig_w", type=float, default=5.0)
    ap.add_argument("--fig_h", type=float, default=4.0)
    ap.add_argument("--sd_alpha", type=float, default=0.2)
    ap.add_argument("--sd_mult", type=float, default=1.0)
    ap.add_argument("--no_sd", action="store_true")
    ap.add_argument(
        "--mode",
        type=str,
        default="per_lam",
        choices=["per_lam", "per_alg"],
    )
    return ap.parse_args()


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
    if s == "ACQB":
        return "ACQB(ours)"
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


def _annotate_lambda(ax, lam: float):
    ax.text(
        0.98, 0.98,
        rf"$\lambda={lam:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.8, alpha=1.0),
    )


def _parse_lambda_from_name(name: str, is_q: bool):
    m = (Q_AVG_RE.match(name) if is_q else REG_AVG_RE.match(name))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def collect_avg(root_dir: Path, include, exclude):
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
        reg_files = list(alg_dir.glob("regret_mean_sd_lam_*.csv"))
        q_files = list(alg_dir.glob("qgap_mean_sd_lam_*.csv"))

        reg_map = {}
        q_map = {}

        for f in reg_files:
            lam = _parse_lambda_from_name(f.name, is_q=False)
            if lam is None:
                continue
            reg_map[lam] = f

        for f in q_files:
            lam = _parse_lambda_from_name(f.name, is_q=True)
            if lam is None:
                continue
            q_map[lam] = f

        common_lams = sorted(set(reg_map.keys()) & set(q_map.keys()))
        if not common_lams:
            continue

        runs = {}
        for lam in common_lams:
            runs[lam] = {"reg_path": reg_map[lam], "q_path": q_map[lam]}
        alg_data[alg_dir.name] = runs

    if not alg_data:
        raise RuntimeError(f"no valid avg csv pairs found under: {root_dir}")

    return alg_data


def read_avg_series(reg_path: Path, q_path: Path, max_steps: int | None):
    df_reg = pd.read_csv(reg_path)
    t = df_reg["t"].to_numpy(dtype=np.int64)
    mu_reg = df_reg["cum_regret_mean"].to_numpy(dtype=np.float64)
    sd_reg = df_reg["cum_regret_sd"].to_numpy(dtype=np.float64)

    df_q = pd.read_csv(q_path)
    tq = df_q["t"].to_numpy(dtype=np.int64)
    if "Q_diff_mean" in df_q.columns:
        mu_q = df_q["Q_diff_mean"].to_numpy(dtype=np.float64)
    else:
        mu_q = df_q.iloc[:, 1].to_numpy(dtype=np.float64)
    if "Q_diff_sd" in df_q.columns:
        sd_q = df_q["Q_diff_sd"].to_numpy(dtype=np.float64)
    else:
        sd_q = np.zeros_like(mu_q)

    L = min(len(t), len(tq), len(mu_reg), len(mu_q))
    if max_steps is not None:
        L = min(L, int(max_steps))

    t = t[:L]
    mu_reg = mu_reg[:L]
    sd_reg = sd_reg[:L]
    mu_q = mu_q[:L]
    sd_q = sd_q[:L]
    return t, mu_reg, sd_reg, mu_q, sd_q


def main():
    args = parse_args()
    set_paper_style()

    root_dir = Path(args.root_dir).expanduser().resolve()
    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    alg_data = collect_avg(root_dir=root_dir, include=args.include, exclude=args.exclude)

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

    m = float(args.sd_mult)

    if args.mode == "per_lam":
        for lam in target_lambdas:
            series = {}
            for alg, runs in alg_data.items():
                if lam not in runs:
                    continue
                reg_path = runs[lam]["reg_path"]
                q_path = runs[lam]["q_path"]
                t, mu_reg, sd_reg, mu_q, sd_q = read_avg_series(reg_path, q_path, args.max_steps)
                series[alg] = {"t": t, "mu_reg": mu_reg, "sd_reg": sd_reg, "mu_q": mu_q, "sd_q": sd_q}

            if not series:
                continue

            alg_names = sorted(series.keys())
            lam_tag = f"{lam:.2f}"

            if 0 < len(alg_names) <= 3:
                prefix_str = "_vs_".join(_rename_for_filename(a) for a in alg_names)
            else:
                prefix_str = "ALLALG"

            fig = plt.figure(figsize=(args.fig_w, args.fig_h))
            ax = fig.gca()
            for alg in alg_names:
                d = series[alg]
                ax.plot(d["t"], d["mu_reg"], label=_rename_for_legend(alg))
                if not args.no_sd:
                    ax.fill_between(
                        d["t"],
                        d["mu_reg"] - m * d["sd_reg"],
                        d["mu_reg"] + m * d["sd_reg"],
                        alpha=float(args.sd_alpha),
                    )
            ax.set_xlabel("t (time)")
            ax.set_ylabel("Cumulative regret")
            _paper_axes(ax)
            ax.legend(loc="upper left")
            _bold_legend_label(ax, "ACQB(ours)")
            _annotate_lambda(ax, lam)
            fig.tight_layout()
            out_std = plots_dir / f"{prefix_str}_mean_regret_lam_{lam_tag}{filename_suffix}.png"
            fig.savefig(out_std)
            plt.close(fig)

            fig = plt.figure(figsize=(args.fig_w, args.fig_h))
            ax = fig.gca()
            for alg in alg_names:
                d = series[alg]
                ax.plot(d["t"], d["mu_q"], label=_rename_for_legend(alg))
                if not args.no_sd:
                    ax.fill_between(
                        d["t"],
                        d["mu_q"] - m * d["sd_q"],
                        d["mu_q"] + m * d["sd_q"],
                        alpha=float(args.sd_alpha),
                    )
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("t (time)")
            ax.set_ylabel(r"$Q(t)-Q_*(t)$")
            _paper_axes(ax)
            ax.legend(loc="upper left")
            _bold_legend_label(ax, "ACQB(ours)")
            _annotate_lambda(ax, lam)
            fig.tight_layout()
            out_q = plots_dir / f"{prefix_str}_mean_qgap_lam_{lam_tag}{filename_suffix}.png"
            fig.savefig(out_q)
            plt.close(fig)

        print("[Done] plots saved to:", str(plots_dir))
        return

    any_plotted = False
    for alg, runs in sorted(alg_data.items(), key=lambda kv: kv[0]):
        lams_here = sorted(set(runs.keys()) & set(target_lambdas))
        if not lams_here:
            continue

        any_plotted = True
        alg_label = _rename_for_legend(alg)
        alg_file = _rename_for_filename(alg)

        series_lam = {}
        for lam in lams_here:
            reg_path = runs[lam]["reg_path"]
            q_path = runs[lam]["q_path"]
            t, mu_reg, sd_reg, mu_q, sd_q = read_avg_series(reg_path, q_path, args.max_steps)
            series_lam[lam] = {"t": t, "mu_reg": mu_reg, "sd_reg": sd_reg, "mu_q": mu_q, "sd_q": sd_q}

        fig = plt.figure(figsize=(args.fig_w, args.fig_h))
        ax = fig.gca()
        for lam in lams_here:
            d = series_lam[lam]
            ax.plot(d["t"], d["mu_reg"], label=rf"$\lambda={lam:.2f}$")
            if not args.no_sd:
                ax.fill_between(
                    d["t"],
                    d["mu_reg"] - m * d["sd_reg"],
                    d["mu_reg"] + m * d["sd_reg"],
                    alpha=float(args.sd_alpha),
                )
        ax.set_xlabel("t (time)")
        ax.set_ylabel("Cumulative regret")
        _paper_axes(ax)
        ax.legend(loc="upper left")
        fig.tight_layout()
        out_std = plots_dir / f"{alg_file}_mean_regret_all_lams{filename_suffix}.png"
        fig.savefig(out_std)
        plt.close(fig)

        fig = plt.figure(figsize=(args.fig_w, args.fig_h))
        ax = fig.gca()
        for lam in lams_here:
            d = series_lam[lam]
            ax.plot(d["t"], d["mu_q"], label=rf"$\lambda={lam:.2f}$")
            if not args.no_sd:
                ax.fill_between(
                    d["t"],
                    d["mu_q"] - m * d["sd_q"],
                    d["mu_q"] + m * d["sd_q"],
                    alpha=float(args.sd_alpha),
                )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("t (time)")
        ax.set_ylabel(r"$Q(t)-Q_*(t)$")
        _paper_axes(ax)
        ax.legend(loc="upper left")
        fig.tight_layout()
        out_q = plots_dir / f"{alg_file}_mean_qgap_all_lams{filename_suffix}.png"
        fig.savefig(out_q)
        plt.close(fig)

        print(f"[Saved] {alg_label} -> {out_std.name}, {out_q.name}")

    if not any_plotted:
        print("[Warn] no plots generated")

    print("[Done] plots saved to:", str(plots_dir))


if __name__ == "__main__":
    main()
