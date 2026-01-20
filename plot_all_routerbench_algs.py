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


def parse_args():
    ap = argparse.ArgumentParser(description="Plot regrets for multiple algorithms under a root directory")
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="알고리즘별 폴더들이 들어있는 루트 디렉토리 (예: online_full/routerbench)",
    )
    ap.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=None,
        help="플롯할 lambda 리스트. 지정 안 하면 root 아래에서 자동 검색.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="플롯 X축(step) 최대 길이 제한 (None이면 전체 사용)",
    )
    ap.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="포함할 알고리즘 폴더명 리스트 (예: 2random_policy 3qucb). 지정 안 하면 자동 탐색.",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["plots"],
        help="제외할 폴더명 리스트",
    )
    ap.add_argument(
        "--qgap_abs",
        action="store_true",
        help="Q-gap을 abs로 그린다. (기본은 signed Q_r - Q_o)",
    )
    return ap.parse_args()


def _parse_lambda_from_name(name: str, is_q: bool) -> float | None:
    m = (QLAM_RE.match(name) if is_q else LAM_RE.match(name))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _pick_latest_by_mtime(paths: list[Path]) -> Path:
    # 같은 lambda 파일이 여러 개 있으면 최신 수정본을 고른다
    return max(paths, key=lambda p: p.stat().st_mtime)


def collect_runs(root_dir: Path, include: list[str] | None, exclude: list[str]):
    """
    return:
      alg_data[alg_name][lam] = dict(reg_path=..., q_path=...)
    """
    alg_dirs = []
    for p in root_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name in set(exclude):
            continue
        if include is not None and p.name not in set(include):
            continue
        alg_dirs.append(p)

    if not alg_dirs:
        raise RuntimeError(f"no algorithm dirs found under: {root_dir}")

    alg_data = {}

    for alg_dir in sorted(alg_dirs, key=lambda x: x.name):
        # 재귀적으로 찾는다 (output_dir 내부에 run 폴더를 만들었을 가능성 대비)
        reg_files = list(alg_dir.rglob("regret_history_lam_*.csv"))
        q_files = list(alg_dir.rglob("Qregret_history_lam_*.csv"))

        reg_map: dict[float, list[Path]] = {}
        q_map: dict[float, list[Path]] = {}

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
            # 이 알고리즘 폴더에는 유효 데이터가 없다
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


def read_series(reg_path: Path, q_path: Path, max_steps: int | None, qgap_abs: bool):
    df_reg = pd.read_csv(reg_path)
    if "cum_regret" in df_reg.columns:
        std_reg = df_reg["cum_regret"].to_numpy()
    else:
        std_reg = df_reg.iloc[:, 0].to_numpy()

    df_q = pd.read_csv(q_path)
    if "Q_diff" in df_q.columns:
        q_diff = df_q["Q_diff"].to_numpy()
    else:
        q_diff = df_q.iloc[:, 0].to_numpy()

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


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    alg_data = collect_runs(
        root_dir=root_dir,
        include=args.include,
        exclude=args.exclude or [],
    )

    # lambda 자동 수집
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

    plt.style.use("seaborn-v0_8-whitegrid")

    # lambda별로 “알고리즘 전부” 한 장에 그린다
    for lam in target_lambdas:
        # 데이터 로드
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
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(alg_names)))

        lam_tag = f"{lam:.2f}"

        # 1) Standard regret
        plt.figure(figsize=(10, 6))
        for i, alg in enumerate(alg_names):
            d = series[alg]
            plt.plot(d["rounds"], d["std_reg"], label=alg, color=colors[i], linewidth=2)
        plt.xlabel("Time Step t", fontsize=12)
        plt.ylabel("Standard Cumulative Regret", fontsize=12)
        plt.title(f"Standard Regret (lambda={lam_tag})", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out_std = plots_dir / f"ALLALG_standard_regret_lam_{lam_tag}{filename_suffix}.png"
        plt.savefig(out_std, dpi=300)
        plt.close()
        print(f"[Saved] {out_std}")

        # 2) Instantaneous Q-gap
        plt.figure(figsize=(10, 6))
        for i, alg in enumerate(alg_names):
            d = series[alg]
            plt.plot(d["rounds"], d["q_diff"], label=alg, color=colors[i], alpha=0.8, linewidth=1.5)
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Time Step t", fontsize=12)
        ylabel = r"$|Q_r(t)-Q_o(t)|$" if args.qgap_abs else r"$Q_r(t)-Q_o(t)$"
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"Instantaneous Queue Gap (lambda={lam_tag})", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out_q = plots_dir / f"ALLALG_queue_gap_lam_{lam_tag}{filename_suffix}.png"
        plt.savefig(out_q, dpi=300)
        plt.close()
        print(f"[Saved] {out_q}")

        # 3) Cumulative Q-gap
        plt.figure(figsize=(10, 6))
        for i, alg in enumerate(alg_names):
            d = series[alg]
            plt.plot(d["rounds"], d["q_cum"], label=alg, color=colors[i], linewidth=2)
        plt.xlabel("Time Step t", fontsize=12)
        ylabel = r"Cumulative $|Q_r(s)-Q_o(s)|$" if args.qgap_abs else r"Cumulative $(Q_r(s)-Q_o(s))$"
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"Cumulative Queue Gap (lambda={lam_tag})", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out_qc = plots_dir / f"ALLALG_queue_cum_lam_{lam_tag}{filename_suffix}.png"
        plt.savefig(out_qc, dpi=300)
        plt.close()
        print(f"[Saved] {out_qc}")

    print("\n[Done] All plots generated.")


if __name__ == "__main__":
    main()