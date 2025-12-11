# routerbench_train_queue_mnl.py
# [Usage]
#   python3 routerbench_train_queue_mnl.py \
#       --data "routerbench_0shot.pkl" \
#       --output_dir "runs_mnl/exp1" \
#       --use_cost \
#       --lam_list "0,1,10,50,100,1000"
#

# python3 routerbench_train_queue_mnl.py \
#     --data "routerbench_0shot.pkl" \
#     --output_dir "runs_mnl/exp1" \
#     --use_cost \
    # --lam_cost 50.0 \
    # --lam_list ""

# Pipeline:
# 1) Load RouterBench, split train/test (stratified).
# 2) Save test_set.pkl.
# 3) SentenceTransformer 로 train prompt 임베딩 학습/적용.
# 4) Save embedder (joblib).
# 5) Train-only 데이터를 queue 환경의 job으로 사용.
# 6) λ (cost weight)에 대해:
#    - acc_mat, util_mat (acc - λ*cost 또는 acc) 구성
#    - QueueConfig 세팅
#    - queue_env(...) 로 MNLRouter joint 학습
#    - 모델 가중치, regret history (standard / Q-length) 저장
# 7) 최종 lambda별 avg regret summary 저장.

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
# from mnl_router_nonB import MNLRouter

from mnl_router import MNLRouter
from queue_env import queue_env
from queue_config import QueueConfig


# ----------------------------
# Data Loading & Utilities
# ----------------------------
def load_routerbench(path: str):
    df = pd.read_pickle(path)
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return df, models, cost_map


def compute_utilities(
    row: pd.Series,
    models: List[str],
    cost_map: Dict[str, str],
    use_cost: bool,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    acc, cost, utility(acc - lam*cost 혹은 acc-only) 반환.
    """
    acc = np.array([float(row[m]) for m in models], dtype=np.float64)
    costs = np.array(
        [float(row.get(cost_map.get(m), 0.0)) for m in models],
        dtype=np.float64,
    )
    if use_cost:
        u = acc - lam * costs
    else:
        u = acc.copy()
    return acc, costs, u


# ----------------------------
# SentenceTransformer Embedder
# ----------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] SentenceTransformer '{model_name}' on {self.device}")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        print(f"[Embedder] Encoding {len(texts)} train prompts...")
        return self.transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=self.device,
        )
        return embs.astype(np.float32)


# ----------------------------
# Train one queue-MNL router at a given lambda
# ----------------------------
def train_single_queue_router(
    args,
    lam_cost_for_training: float,
    X_train: np.ndarray,
    df_train: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
) -> Tuple[MNLRouter, float, float, List[float], List[float]]:
    """
    λ 하나에 대해 queue 환경에서 MNLRouter 학습.

    return:
      - router
      - avg_regret         (Regret_T / T)
      - Q_regret_T         (최종 queue-length regret Q(T) - Q*(T))
      - regret_history     (step별 cum_regret)
      - Q_regret_history   (step별 Q(t) - Q*(t))
    """
    N_train = len(df_train)
    K_models = len(models)

    # acc_mat, util_mat 구성
    acc_mat = np.zeros((N_train, K_models), dtype=np.float64)
    util_mat = np.zeros((N_train, K_models), dtype=np.float64)

    print(f"[Lambda={lam_cost_for_training}] Building acc_mat/util_mat...")
    for i, (_, row) in enumerate(df_train.iterrows()):
        acc, costs, u = compute_utilities(
            row,
            models=models,
            cost_map=cost_map,
            use_cost=args.use_cost,
            lam=lam_cost_for_training,
        )
        acc_mat[i] = acc
        util_mat[i] = u

    # QueueConfig 세팅 (Unknown horizon 알고리즘 기준)
    config = QueueConfig(
        d_proj=args.d_proj,
        reg_lambda=args.reg_lambda,
        supcon_temp=args.supcon_temp,
        supcon_bs=args.supcon_bs,
        supcon_weight=args.supcon_weight,
        assort_K=args.assort_K,
        kappa=1.0,
        lambda0=None,  # None이면 reg_lambda 사용
        c0=args.c0,
        arrival_rate=args.arrival_rate,
        max_steps=args.max_steps,
        log_every=args.log_every,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # queue 환경에서 학습
    print(f"[Lambda={lam_cost_for_training}] Training in queue env...")
    router, avg_reg, Q_reg_T, reg_hist, Q_reg_hist = queue_env(
        X_ctx=X_train,
        acc_mat=acc_mat,
        util_mat=util_mat,
        config=config,
    )

    return router, avg_reg, Q_reg_T, reg_hist, Q_reg_hist


# ----------------------------
# Main Training Script
# ----------------------------
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Saving results to: {output_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 0) RouterBench 로드
    df, models, cost_map = load_routerbench(args.data)
    initial_count = len(df)
    df.dropna(subset=models, inplace=True)
    df = df.reset_index(drop=True)
    print(f"[Info] Loaded {initial_count} rows, {len(df)} remain after drop-NaN.")
    K_models = len(models)

    # 1) Train/Test split
    print(
        f"[Info] Split: Train {100*(1-args.test_size):.0f}% / "
        f"Test {100*args.test_size:.0f}% (stratified by eval_name)"
    )
    stratify_col = df["eval_name"] if "eval_name" in df.columns else None
    if stratify_col is not None:
        stratify_col = stratify_col.fillna("unknown")
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_col,
    )

    # [SAVE 1] Test set 저장
    test_set_path = output_dir / "test_set.pkl"
    df_test.to_pickle(test_set_path)
    print(f"[Info] Test set saved to: {test_set_path}")

    # 2) SentenceTransformer 임베딩 (train)
    prompts_train = df_train["prompt"].astype(str).tolist()
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedder_model,
        device=device,
    )
    X_train = embedder.fit_transform(prompts_train)
    d_query = X_train.shape[1]
    print(f"[Info] d_query (embedding dim) = {d_query}")

    # [SAVE 2] embedder 저장 (joblib로 래핑)
    embedder_path = output_dir / "embedder.joblib"
    joblib.dump(embedder, embedder_path)
    print(f"[Info] Embedder saved to: {embedder_path}")

    # [SAVE 3] meta 정보 + args 저장
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    meta = {
        "d_query": d_query,
        "K_models": K_models,
        "models": models,
        "cost_map": cost_map,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    # 3) λ 리스트 구성
    final_summary = []  # (lambda, avg_reg, Q_reg_T)

    if args.use_cost:
        # lam_list 문자열 파싱 (기본값 "0,1,10,50,100,1000")
        if args.lam_list is not None and args.lam_list.strip() != "":
            try:
                lambda_points_to_train = [
                    float(x) for x in args.lam_list.split(",") if x.strip() != ""
                ]
            except ValueError:
                raise ValueError(f"Invalid --lam_list format: {args.lam_list}")
            print(f"[Info] Using cost-aware utility with λ list = {lambda_points_to_train}.")
        else:
            lambda_points_to_train = [args.lam_cost]
            print(f"[Info] Using cost-aware utility with single λ = {args.lam_cost}.")
    else:
        print("!! WARNING: --use_cost 안 켜짐. lambda=0.0 하나만 학습 (acc-only).")
        lambda_points_to_train = [0.0]

    # 4) λ별 학습 루프
    for lam_train in lambda_points_to_train:
        print(
            f"\n{'='*60}\n"
            f"[Lambda={lam_train}] Training start (queue_env)\n"
            f"{'='*60}"
        )

        router, avg_reg, Q_reg_T, reg_hist, Q_reg_hist = train_single_queue_router(
            args,
            lam_cost_for_training=lam_train,
            X_train=X_train,
            df_train=df_train,
            models=models,
            cost_map=cost_map,
        )

        final_summary.append((lam_train, avg_reg, Q_reg_T))

        # [SAVE 모델] state_dict 저장
        model_path = models_dir / f"mnl_router_lam_{lam_train:.2f}.pth"
        torch.save(router.state_dict(), model_path)
        print(f"[Lambda={lam_train}] Router weights saved to: {model_path}")

        # [SAVE regret history] standard regret
        rounds = np.arange(1, len(reg_hist) + 1, dtype=int)
        reg_df = pd.DataFrame(
            {
                "round": rounds,
                "cum_regret": np.array(reg_hist, dtype=float),
                "avg_regret": np.array(reg_hist, dtype=float) / rounds,
            }
        )
        reg_path = output_dir / f"regret_history_lam_{lam_train:.2f}.csv"
        reg_df.to_csv(reg_path, index=False)
        print(f"[Lambda={lam_train}] Standard regret history saved to: {reg_path}")

        # [SAVE queue-length gap history] Q(t) - Q*(t)
        q_rounds = np.arange(1, len(Q_reg_hist) + 1, dtype=int)
        q_df = pd.DataFrame(
            {
                "step": q_rounds,
                "Q_diff": np.array(Q_reg_hist, dtype=float),  # = Q(t) - Q*(t)
            }
        )
        q_path = output_dir / f"Qregret_history_lam_{lam_train:.2f}.csv"
        q_df.to_csv(q_path, index=False)
        print(f"[Lambda={lam_train}] Queue-length gap history saved to: {q_path}")

    # 5) 최종 summary 저장
    summary_df = pd.DataFrame(
        final_summary,
        columns=["lambda", "final_avg_regret", "final_Q_regret_T"],
    )
    summary_path = output_dir / "final_regret_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Final] Summary saved to: {summary_path}")

    print("\n[Summary] Final Regret vs Lambda")
    print("-------------------------------------------")
    print(f"{'Lambda':<10} | {'AvgStdReg':<15} | {'Qreg_T':<15}")
    print("-------------------------------------------")
    for lam, r, qr in final_summary:
        print(f"{lam:<10.2f} | {r:<15.6f} | {qr:<15.6f}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train MNLRouter + Queue env on RouterBench (Unknown horizon)"
    )
    ap.add_argument("--data", type=str, required=True, help="Path to RouterBench .pkl file")
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp1", help="Directory to save results")

    # Data split
    ap.add_argument("--test_size", type=float, default=0.3, help="Fraction for test set (0.3 = 30%)")

    # Utility & cost
    ap.add_argument("--use_cost", action="store_true", help="Use utility = acc - lam * cost")
    ap.add_argument("--lam_cost", type=float, default=50.0, help="Cost penalty λ for utility (fallback)")
    ap.add_argument(
        "--lam_list",
        type=str,
        default="0,1,10,50,100,1000",
        help="Comma-separated list of λ values (e.g. '0,1,10,50,100,1000'). "
             "If empty string, falls back to lam_cost.",
    )

    # MNLRouter / queue hyperparams
    ap.add_argument("--d_proj", type=int, default=128, help="Latent dim d_proj")
    ap.add_argument("--reg_lambda", type=float, default=1.0, help="Ridge λ for V_inv")
    ap.add_argument("--supcon_temp", type=float, default=0.07, help="SupCon temperature")
    ap.add_argument("--supcon_bs", type=int, default=64, help="SupCon batch size")
    ap.add_argument("--supcon_weight", type=float, default=1.0, help="SupCon loss weight λ_sc")

    ap.add_argument("--assort_K", type=int, default=2, help="Assortment size |S_t|")

    ap.add_argument("--arrival_rate", type=float, default=0.7, help="Job arrival probability")
    ap.add_argument("--max_steps", type=int, default=100000, help="Max queue simulation steps")
    ap.add_argument("--log_every", type=int, default=1000, help="Logging interval")

    # Unknown-horizon exploration parameter
    ap.add_argument("--c0", type=float, default=1.0, help="Exploration coefficient c0 in η(t)")

    # Embedder / logging / seed
    ap.add_argument("--embedder_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    return ap.parse_args()


if __name__ == "__main__":
    main()
