#!/usr/bin/env python3
# routerbench_train_queue_mnl_nonB.py
#
# Usage 예시:
#   python3 NONB/routerbench_train_queue_mnl_nonB.py \
#       --data routerbench_0shot.pkl \
#       --output_dir runs_mnl/exp_nonB \
#       --lam_list 50.0
#
# job pool을 100개로 고정하고 싶으면:
#   python3 NONB/routerbench_train_queue_mnl_nonB.py \
#       --data routerbench_0shot.pkl \
#       --output_dir runs_mnl/exp_nonB_pool100 \
#       --lam_list 50.0 \
#       --job_pool_size 100

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from queue_env_nonB import queue_env
from queue_config import QueueConfig


# ----------------------------
# Data Loading & Utilities
# ----------------------------
def load_routerbench(path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_pickle(path)

    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}

    subset_cols = models + list(cost_map.values())
    df = df.dropna(subset=subset_cols).reset_index(drop=True)

    return df, models, cost_map


def compute_utilities(
    row: pd.Series,
    models: List[str],
    cost_map: Dict[str, str],
    use_cost: bool,
    lam_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    acc, cost, utility(acc - lam*cost 혹은 acc-only) 반환한다.
    """
    acc = np.array([float(row[m]) for m in models], dtype=np.float64)
    costs = np.array(
        [float(row.get(cost_map.get(m), 0.0)) for m in models],
        dtype=np.float64,
    )
    if use_cost:
        u = acc - lam_cost * costs
    else:
        u = acc.copy()
    return acc, costs, u


# ----------------------------
# SentenceTransformer Embedder
# ----------------------------
class Embedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = str(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] SentenceTransformer '{model_name}' on {self.device}")

    def transform(self, texts: List[str]) -> np.ndarray:
        print(f"[Embedder] Encoding {len(texts)} prompts...")
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embs.astype(np.float32)


# -------------------------------------------------------
# Main Process (NonB: offline SupCon 없이 바로 queue_env)
# -------------------------------------------------------
def main():
    args = parse_args()

    # 0. Config 로딩
    config = QueueConfig()
    device = torch.device(config.device)

    print("========== QueueConfig ==========")
    print(config)
    print("=================================")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 1. RouterBench 데이터 로딩
    print(f"[Info] Loading Data: {args.data} ...")
    df, models, cost_map = load_routerbench(args.data)
    df.dropna(subset=models, inplace=True)
    df = df.reset_index(drop=True)
    K_models = len(models)
    print(f"[Info] Total rows after drop-NaN: {len(df)}, #Models={K_models}")

    # 2. Train/Test Split + (옵션) job pool 고정 샘플링 + 임베딩
    embedder = Embedder(config.embedder_model, device)

    df_train, df_test = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.seed,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(
        f"[Split] Total={len(df)} | Train={len(df_train)} | Test={len(df_test)}"
    )

    # ---- job pool 고정 옵션 ----
    if args.job_pool_size is not None:
        n_pool = int(args.job_pool_size)
        if not (1 <= n_pool <= len(df_train)):
            raise ValueError(f"job_pool_size must be in [1, {len(df_train)}], got {n_pool}")

        df_train = (
            df_train.sample(n=n_pool, random_state=config.seed)
            .reset_index(drop=True)
        )
        print(f"[Pool] Fixed job pool size = {n_pool} (sampled from train set)")
    else:
        print("[Pool] Using full train set as job pool")

    # train job pool에 대해서만 임베딩 생성
    X_train = embedder.transform(df_train["prompt"].astype(str).tolist())
    d_ctx = X_train.shape[1]
    print(f"[Info] d_ctx (embedding dim) = {d_ctx}")

    # 3. λ 리스트 구성
    if config.use_cost:
        if args.lam_list is not None:
            lambdas = args.lam_list
            print(f"[Lambda] Using CLI lam_list: {lambdas}")
        else:
            lambdas = [config.lam_cost]
            print(f"[Lambda] Using QueueConfig.lam_cost = {config.lam_cost}")
    else:
        lambdas = [0.0]
        print("[Lambda] use_cost=False → λ=0.0 (acc-only)")

    # 4. Lambda Loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Results will be saved under: {output_dir}")

    for lam in lambdas:
        print(f"\n>> Experiment Lambda = {lam}")

        # acc / util matrices for train job pool
        acc_train = np.zeros((len(df_train), K_models), dtype=np.float64)
        util_train = np.zeros((len(df_train), K_models), dtype=np.float64)
        for i, (_, row) in enumerate(df_train.iterrows()):
            acc, _, u = compute_utilities(
                row,
                models=models,
                cost_map=cost_map,
                use_cost=config.use_cost,
                lam_cost=lam,
            )
            acc_train[i] = acc
            util_train[i] = u

        # Queue 환경에서 NonB Router 학습
        print("[Online Phase] Starting Queue Bandit Simulation (NonB)...")

        router, avg_reg, Q_reg_T, reg_hist, Q_hist = queue_env(
            X_ctx=X_train,
            acc_mat=acc_train,
            util_mat=util_train,
            config=config,
        )

        # 결과 저장
        pd.DataFrame({"cum_regret": reg_hist}).to_csv(
            output_dir / f"regret_history_lam_{lam:.2f}.csv",
            index=False,
        )
        pd.DataFrame({"Q_diff": Q_hist}).to_csv(
            output_dir / f"Qregret_history_lam_{lam:.2f}.csv",
            index=False,
        )
        print(
            f"[Done] Lambda {lam} finished. "
            f"Avg Regret={avg_reg:.4f}, Q_reg_T={Q_reg_T:.4f}"
        )


# ----------------------------
# CLI Argument Parser
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Train NonB MNLRouter + Queue env (unknown horizon) on RouterBench"
    )
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to RouterBench .pkl file",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./runs_mnl/exp_nonB",
        help="Directory to save results",
    )
    ap.add_argument(
        "--lam_list",
        type=float,
        nargs="+",
        default=None,
        help=(
            "여러 lambda 값들을 공백으로 구분해서 입력한다 "
            "(예: --lam_list 10 30 50). "
            "지정하지 않으면 QueueConfig.lam_cost를 사용한다."
        ),
    )
    ap.add_argument(
        "--job_pool_size",
        type=int,
        default=None,
        help=(
            "None이면 train 전체를 job pool로 사용한다. "
            "정수 n을 주면 train에서 n개 문제를 랜덤으로 고정 샘플링한 뒤 "
            "그 고정 pool 안에서 arrival를 중복 허용 랜덤 샘플링한다. "
            "(예: --job_pool_size 100)"
        ),
    )
    return ap.parse_args()


if __name__ == "__main__":
    main()
