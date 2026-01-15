import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    files = list(output_dir.glob("regret_history_lam_*.csv"))
    if not files:
        print(f"[Warn] No 'regret_history_lam_*.csv' files found in {output_dir}")
        return []

    lambdas = []
    for f in files:
        try:
            stem = f.stem
            parts = stem.split("_")
            if "lam" in parts:
                lam_idx = parts.index("lam") + 1
                lam_val = float(parts[lam_idx])
                lambdas.append(lam_val)
        except (RuntimeError, IndexError):
            continue

    return sorted(list(set(lambdas)))

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.lambdas:
        target_lambdas = sorted(args.lambdas)
    else:
        target_lambdas = get_lambdas_from_dir(output_dir)

    if not target_lambdas:
        print("[Error] No lambdas found. Check directory or filenames.")
        return

    print(f"[Info] Target Lambdas: {target_lambdas}")
    
    # max_steps 설정 여부에 따라 파일명 접미사 생성
    filename_suffix = ""
    if args.max_steps is not None:
        print(f"[Info] Plotting restricted to first {args.max_steps} steps.")
        filename_suffix = f"_{args.max_steps}"

    data_store = {}

    for lam in target_lambdas:
        lam_tag = f"{lam:.2f}"

        reg_path = output_dir / f"regret_history_lam_{lam_tag}.csv"
        q_path   = output_dir / f"Qregret_history_lam_{lam_tag}.csv"

        if not reg_path.exists() or not q_path.exists():
            print(f"[Warn] Missing files for lambda={lam} (expected {reg_path.name}, {q_path.name}), skipping.")
            continue

        df_reg = pd.read_csv(reg_path)

        if "cum_regret" in df_reg.columns:
            std_reg = df_reg["cum_regret"].to_numpy()
        else:
            std_reg = df_reg.iloc[:, 0].to_numpy()

        rounds = np.arange(1, len(std_reg) + 1)

        df_q = pd.read_csv(q_path)
        if "Q_diff" in df_q.columns:
            q_diff = df_q["Q_diff"].to_numpy()
        else:
            q_diff = df_q.iloc[:, 0].to_numpy()

        # 데이터 길이 자르기 로직
        L = min(len(rounds), len(q_diff))
        if args.max_steps is not None:
            L = min(L, args.max_steps)

        rounds = rounds[:L]
        std_reg = std_reg[:L]
        q_diff = q_diff[:L]
        q_cum = np.cumsum(q_diff)

        data_store[lam] = {
            "rounds": rounds,
            "std_reg": std_reg,
            "q_diff": q_diff,
            "q_cum": q_cum,
        }

    if not data_store:
        print("[Error] No valid data loaded.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data_store)))

    # 1. Standard Regret Plot
    plt.figure(figsize=(10,6))
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
    plt.title(f"Standard Regret (First {args.max_steps if args.max_steps else 'All'} Steps)", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 파일명에 suffix 추가
    out_std = plots_dir / f"plot_standard_regret{filename_suffix}.png"
    plt.savefig(out_std, dpi=300)
    plt.close()
    print(f"[Saved] {out_std}")

    # 2. Queue Stability Plot
    plt.figure(figsize=(10, 6))
    for i, (lam, data) in enumerate(sorted(data_store.items(), key=lambda x: x[0])):
        plt.plot(
            data["rounds"],
            data["q_diff"],
            label=f"$\\lambda={lam}$",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Time Step $t$", fontsize=12)
    plt.ylabel(r"$|Q(t) - Q^*(t)|$", fontsize=12)
    plt.title("Instantaneous Queue Gap (Absolute)", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 파일명에 suffix 추가
    out_q_diff = plots_dir / f"plot_queue_stability{filename_suffix}.png"
    plt.savefig(out_q_diff, dpi=300)
    plt.close()
    print(f"[Saved] {out_q_diff}")

    # 3. Cumulative Queue Gap Plot
    plt.figure(figsize=(10, 6))
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

    # 파일명에 suffix 추가
    out_q_cum = plots_dir / f"plot_queue_cumulative{filename_suffix}.png"
    plt.savefig(out_q_cum, dpi=300)
    plt.close()
    print(f"[Saved] {out_q_cum}")

    print("\n[Done] All plots generated successfully.")

if __name__ == "__main__":
    main()