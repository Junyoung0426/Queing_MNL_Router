#!/usr/bin/env python3
# routerbench_train_queue_mnl_B.py
#
# 실행 예:
#   python3 routerbench_train_queue_mnl_B.py \
#     --data routerbench_0shot.pkl \
#     --output_dir runs_mnl/exp_B \
#     --lam_list 50.0
#
# pool 고정:
#   python3 routerbench_train_queue_mnl_B.py \
#     --data routerbench_0shot.pkl \
#     --output_dir runs_mnl/exp_B_pool100 \
#     --lam_list 50.0 \
#     --job_pool_size 100

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from queue_config import QueueConfig
from mnl_router import MNLRouter
from queue_env import queue_env


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
    acc = np.array([float(row[m]) for m in models], dtype=np.float64)
    costs = np.array([float(row.get(cost_map.get(m), 0.0)) for m in models], dtype=np.float64)
    u = acc - lam_cost * costs if use_cost else acc.copy()
    return acc, costs, u


class Embedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = str(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] SentenceTransformer '{model_name}' on {self.device}")

    def transform(self, texts: List[str]) -> np.ndarray:
        print(f"[Embedder] Encoding {len(texts)} prompts...")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embs.astype(np.float32)


def offline_pretrain_B_supcon(
    router: MNLRouter,
    X_ctx_train: np.ndarray,       # (N, d_ctx)
    winner_idx: np.ndarray,        # (N,)
    config: QueueConfig,
):
    """
    offline_ratio/epochs/lr_B/supcon_bs 그대로 써서 B만 SupCon으로 학습한다.
    스킵하더라도 freeze/reset은 항상 수행한다.
    """
    N = X_ctx_train.shape[0]

    # 항상 offline 시작 상태로 만든다
    router.unfreeze_B(lr_b=config.offline_lr_B)

    if config.offline_ratio <= 0.0 or config.offline_epochs <= 0:
        print("[Offline] skipped (offline_ratio<=0 or offline_epochs<=0)")
        router.freeze_B()
        router.reset_for_online()
        return

    n_off = max(2, int(N * config.offline_ratio))
    rng = np.random.RandomState(config.seed)
    off_idx = rng.choice(N, size=n_off, replace=False)

    X_off = torch.from_numpy(X_ctx_train[off_idx]).float().to(router.device)
    y_off = torch.from_numpy(winner_idx[off_idx]).long().to(router.device)

    bs = int(config.supcon_bs)
    replace = (n_off < bs)

    print(
        f"[Offline] SupCon pretrain: N_off={n_off}, bs={bs}, "
        f"epochs={config.offline_epochs}, b_type={config.b_type}"
    )

    for ep in range(1, int(config.offline_epochs) + 1):
        b_idx_np = rng.choice(n_off, size=bs, replace=replace)
        b_idx = torch.as_tensor(b_idx_np, device=router.device, dtype=torch.long)

        loss_sc = router.supcon_step(X_off[b_idx], y_off[b_idx], lambda_sc=1.0)

        if (ep % 50 == 0) or (ep == 1) or (ep == config.offline_epochs):
            print(f"[Offline] epoch={ep:4d} | supcon_loss={loss_sc:.4f}")

    router.freeze_B()
    router.reset_for_online()


def parse_args():
    ap = argparse.ArgumentParser(description="B version: offline SupCon(B-only) + online queue(theta-only)")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp_B")
    ap.add_argument("--lam_list", type=float, nargs="+", default=None)
    ap.add_argument("--job_pool_size", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()
    device = torch.device(config.device)

    print("========== QueueConfig ==========")
    print(config)
    print("=================================")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"[Info] Loading Data: {args.data} ...")
    df, models, cost_map = load_routerbench(args.data)
    K_models = len(models)
    print(f"[Info] Total rows after drop-NaN: {len(df)}, #Models={K_models}")

    # split
    df_train, df_test = train_test_split(df, test_size=config.test_size, random_state=config.seed)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"[Split] Total={len(df)} | Train={len(df_train)} | Test={len(df_test)}")

    # job_pool_size 적용 (pool support만 줄이고, arrival는 pool에서 with-replacement)
    if args.job_pool_size is not None:
        n_pool = int(args.job_pool_size)
        if not (1 <= n_pool <= len(df_train)):
            raise ValueError(f"job_pool_size must be in [1,{len(df_train)}], got {n_pool}")
        df_train = df_train.sample(n=n_pool, random_state=config.seed).reset_index(drop=True)
        print(f"[Pool] Fixed job pool size = {n_pool}")
    else:
        print("[Pool] Using full train set as job pool")

    # embed
    embedder = Embedder(config.embedder_model, device)
    X_train = embedder.transform(df_train["prompt"].astype(str).tolist())
    d_ctx = X_train.shape[1]
    print(f"[Info] d_ctx = {d_ctx}")

    # config 채우기
    config.d_ctx = d_ctx
    config.n_models = K_models

    # lambda list
    if config.use_cost:
        lambdas = args.lam_list if args.lam_list is not None else [config.lam_cost]
    else:
        lambdas = [0.0]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for lam in lambdas:
        print(f"\n>> Experiment Lambda = {lam}")

        acc_train = np.zeros((len(df_train), K_models), dtype=np.float64)
        util_train = np.zeros((len(df_train), K_models), dtype=np.float64)

        for i, (_, row) in enumerate(df_train.iterrows()):
            acc, _, u = compute_utilities(row, models, cost_map, config.use_cost, lam)
            acc_train[i] = acc
            util_train[i] = u

        # offline winner label (util 기준)
        winner_idx = np.argmax(util_train, axis=1).astype(np.int64)

        # router 생성 + offline pretrain
        lambda_0 = config.lambda0 if config.lambda0 is not None else config.reg_lambda
        router = MNLRouter(
            d_ctx=d_ctx,
            n_models=K_models,
            d_proj=config.d_proj,
            lambda_0=lambda_0,
            supcon_temp=config.supcon_temp,
            device=config.device,
            b_type=config.b_type,
            b_hidden_mult=config.b_hidden_mult,
            d_model_emb=config.d_model_emb,
            lr_b=config.offline_lr_B,
        ).to(device)

        offline_pretrain_B_supcon(router, X_train, winner_idx, config)

        # online queue
        print("[Online] Starting Queue Bandit Simulation (B)...")
        router, avg_reg, Q_reg_T, reg_hist, Q_hist = queue_env(
            X_ctx=X_train,
            acc_mat=acc_train,
            util_mat=util_train,
            config=config,
            router=router,
        )

        pd.DataFrame({"cum_regret": reg_hist}).to_csv(outdir / f"regret_history_lam_{lam:.2f}.csv", index=False)
        pd.DataFrame({"Q_diff": Q_hist}).to_csv(outdir / f"Qregret_history_lam_{lam:.2f}.csv", index=False)

        print(f"[Done] Lambda {lam} finished. AvgRegret={avg_reg:.6f}, Q_reg_T={Q_reg_T:.3f}")


if __name__ == "__main__":
    main()
