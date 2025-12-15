#plot_regret.py

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # X 서버 없는 환경 지원
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot combined regrets for multiple lambdas")
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="CSV 파일들이 저장된 디렉토리 (예: runs_mnl/exp_nonB)",
    )
    ap.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=None,
        help="플롯할 lambda 리스트. (예: 1.0 10.0). 지정 안 하면 폴더 내 자동 검색.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="플롯 X축(step) 최대 길이 제한 (None이면 전체 사용)",
    )
    return ap.parse_args()


def get_lambdas_from_dir(output_dir: Path):
    """
    폴더 내의 csv 파일들을 스캔하여 사용 가능한 lambda 리스트를 추출한다.
    파일명 패턴: regret_history_lam_{LAMBDA}.csv
    """
    files = list(output_dir.glob("regret_history_lam_*.csv"))
    if not files:
        print(f"[Warn] No 'regret_history_lam_*.csv' files found in {output_dir}")
        return []

    lambdas = []
    for f in files:
        try:
            # 파일명 예시: regret_history_lam_1.00.csv
            stem = f.stem  # regret_history_lam_1.00
            parts = stem.split("_")
            if "lam" in parts:
                lam_idx = parts.index("lam") + 1
                lam_val = float(parts[lam_idx])
                lambdas.append(lam_val)
        except (ValueError, IndexError):
            continue

    return sorted(list(set(lambdas)))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lambda 리스트 확정
    if args.lambdas:
        target_lambdas = sorted(args.lambdas)
    else:
        target_lambdas = get_lambdas_from_dir(output_dir)

    if not target_lambdas:
        print("[Error] No lambdas found. Check directory or filenames.")
        return

    print(f"[Info] Target Lambdas: {target_lambdas}")

    # 데이터 로딩
    data_store = {}

    for lam in target_lambdas:
        lam_tag = f"{lam:.2f}"

        reg_path = output_dir / f"regret_history_lam_{lam_tag}.csv"
        q_path   = output_dir / f"Qregret_history_lam_{lam_tag}.csv"

        if not reg_path.exists() or not q_path.exists():
            print(f"[Warn] Missing files for lambda={lam} (expected {reg_path.name}, {q_path.name}), skipping.")
            continue

        # 1) Load Standard Cumulative Regret
        df_reg = pd.read_csv(reg_path)

        if "cum_regret" in df_reg.columns:
            std_reg = df_reg["cum_regret"].to_numpy()
        else:
            # 예전 포맷 대비 방어적 처리: 첫 컬럼 사용
            std_reg = df_reg.iloc[:, 0].to_numpy()

        rounds = np.arange(1, len(std_reg) + 1)  # 1, 2, ..., T

        # 2) Load Queue Gap (Instantaneous Q(t) - Q*(t))
        df_q = pd.read_csv(q_path)
        if "Q_diff" in df_q.columns:
            q_diff = df_q["Q_diff"].to_numpy()
        else:
            q_diff = df_q.iloc[:, 0].to_numpy()

        # 3) 길이 맞추기 + max_steps 적용
        L = min(len(rounds), len(q_diff))
        if args.max_steps is not None:
            L = min(L, args.max_steps)

        rounds = rounds[:L]
        std_reg = std_reg[:L]
        q_diff = q_diff[:L]                 # 여기서 q_diff는 이미 |Q - Q*|
        q_cum = np.cumsum(q_diff)           # 누적 queue gap

        data_store[lam] = {
            "rounds": rounds,
            "std_reg": std_reg,
            "q_diff": q_diff,
            "q_cum": q_cum,
        }

    if not data_store:
        print("[Error] No valid data loaded.")
        return

    # 공통 스타일 설정
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data_store)))

    # -------------------------------------------------------
    # Plot 1: Standard Cumulative Regret
    # (모델이 얼마나 잘 선택되고 있는가?)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 10))
    for i, (lam, data) in enumerate(sorted(data_store.items(), key=lambda x: x[0])):
        plt.plot(
            data["rounds"],
            data["std_reg"],
            label=f"$\\lambda={lam}$",
            color=colors[i],
            linewidth=2,
        )

    plt.xlabel("Time Step $t$", fontsize=12)
    plt.ylabel("Standard Cumulative Regret", fontsize=12)
    plt.title("Standard Regret (Learning Performance)", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    out_std = plots_dir / "plot_standard_regret.png"
    plt.savefig(out_std, dpi=300)
    plt.close()
    print(f"[Saved] {out_std}")



    # -------------------------------------------------------
    # Plot 3: Cumulative Queue Gap
    # \sum_{s=1}^t |Q(s) - Q^*(s)|
    # -------------------------------------------------------
    plt.figure(figsize=(10, 10))
    for i, (lam, data) in enumerate(sorted(data_store.items(), key=lambda x: x[0])):
        plt.plot(
            data["rounds"],
            data["q_cum"],
            label=f"$\\lambda={lam}$",
            color=colors[i],
            linewidth=2,
        )

    plt.xlabel("Time Step $t$", fontsize=12)
    plt.ylabel(r"Cumulative $Q(s) - Q^*(s)$", fontsize=12)
    plt.title("Cumulative Queue Gap ", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    out_q_cum = plots_dir / "plot_queue_cumulative.png"
    plt.savefig(out_q_cum, dpi=300)
    plt.close()
    print(f"[Saved] {out_q_cum}")

    print("\n[Done] All plots generated successfully.")


if __name__ == "__main__":
    main()
