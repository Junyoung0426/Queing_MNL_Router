#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from queue_config import QueueConfig
from mnl_router import MNLRouter
from queue_env import queue_env

from llm_embedding import (
    compute_anchor_centroids,
    build_a_table_from_xi_S,
)

def infer_models_and_cost_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return models, cost_map

def load_routerbench(
    path: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_pickle(path)

    models_auto, cost_map = infer_models_and_cost_map(df)

    if models_fixed is None:
        models = sorted(models_auto)
    else:
        missing = [m for m in models_fixed if m not in models_auto]
        extra = [m for m in models_auto if m not in models_fixed]
        if missing:
            pass
        if extra:
            pass
        models = list(models_fixed)

    subset_cols = list(models)
    if use_cost:
        subset_cols += [cost_map[m] for m in models if m in cost_map]

    df = df.dropna(subset=subset_cols).reset_index(drop=True)
    return df, models, cost_map

def compute_acc_cost_util_all(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    use_cost: bool,
    lam_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(df)
    K = len(models)
    acc = np.zeros((N, K), dtype=np.float64)
    util = np.zeros((N, K), dtype=np.float64)

    for i, (_, row) in enumerate(df.iterrows()):
        a = np.array([float(row[m]) for m in models], dtype=np.float64)
        c = np.array([float(row.get(cost_map.get(m), 0.0)) for m in models], dtype=np.float64)
        u = a - lam_cost * c if use_cost else a.copy()
        acc[i] = a
        util[i] = u

    return acc, util

class Embedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = str(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] SentenceTransformer '{model_name}' on {self.device}")

    def transform(self, texts: List[str]) -> np.ndarray:
        print(f"[Embedder] Encoding {len(texts)} prompts...")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True,normalize_embeddings=True)
        return embs.astype(np.float32)

def _sample_winner_balanced(
    winners: np.ndarray,
    candidate_idx: np.ndarray,
    K: int,
    n_total: int,
    rng: np.random.RandomState,
) -> np.ndarray:

    if n_total <= 0:
        return np.zeros((0,), dtype=np.int64)

    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)

    base = n_total // K
    rem = n_total % K
    order = rng.permutation(K)
    need = np.full(K, base, dtype=int)
    need[order[:rem]] += 1

    chosen_list: List[np.ndarray] = []
    chosen_set = set()
    shortage = 0

    for k in range(K):
        pool = candidate_idx[winners[candidate_idx] == k]
        if pool.size >= need[k]:
            sel = rng.choice(pool, size=need[k], replace=False)
        else:
            sel = pool
            shortage += (need[k] - pool.size)

        if sel.size > 0:
            chosen_list.append(sel)
            for v in sel.tolist():
                chosen_set.add(int(v))

    chosen = np.concatenate(chosen_list).astype(np.int64) if chosen_list else np.zeros((0,), dtype=np.int64)

    if shortage > 0:
        mask = np.array([int(i) not in chosen_set for i in candidate_idx.tolist()], dtype=bool)
        rest = candidate_idx[mask]
        if rest.size >= shortage:
            extra = rng.choice(rest, size=shortage, replace=False)
        else:
            extra = rng.choice(candidate_idx, size=shortage, replace=True)
        chosen = np.concatenate([chosen, extra.astype(np.int64)])

    if chosen.size > n_total:
        chosen = chosen[:n_total]
    return chosen.astype(np.int64)

def build_offline_partition_7030(
    util_train: np.ndarray,
    config: QueueConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    N, K = util_train.shape
    rng = np.random.RandomState(int(config.seed) + 999)

    winners = np.argmax(util_train, axis=1).astype(np.int64)
    counts = np.bincount(winners, minlength=K).astype(int)
    if np.any(counts <= 0):
        bad = np.where(counts <= 0)[0].tolist()
        pass

    offline_total = int(round(float(config.offline_total_ratio) * N))
    offline_total = max(offline_total, K)
    offline_total = min(offline_total, N - 2)

    win_frac = float(config.tb_win_frac)
    rand_frac = float(config.tb_rand_frac)
    if abs((win_frac + rand_frac) - 1.0) > 1e-6:
        s = win_frac + rand_frac
        win_frac /= s
        rand_frac /= s

    n_win = int(round(win_frac * offline_total))
    n_win = max(n_win, K)
    n_win = min(n_win, offline_total)
    n_rand = offline_total - n_win

    candidate = np.arange(N, dtype=np.int64)

    win_idx = _sample_winner_balanced(
        winners=winners,
        candidate_idx=candidate,
        K=K,
        n_total=n_win,
        rng=rng,
    )

    remain = np.setdiff1d(candidate, win_idx, assume_unique=False)
    if n_rand > 0:
        if remain.size < n_rand:
            pass
        rand_idx = rng.choice(remain, size=n_rand, replace=False).astype(np.int64)
    else:
        rand_idx = np.zeros((0,), dtype=np.int64)

    offline_idx = np.concatenate([win_idx, rand_idx]).astype(np.int64)
    offline_idx = rng.permutation(offline_idx)

    online_idx = np.setdiff1d(candidate, offline_idx, assume_unique=False).astype(np.int64)
    if online_idx.size < 2:
        pass

    return win_idx, rand_idx, offline_idx, online_idx

def offline_pretrain_B_supcon(
    router: MNLRouter,
    X_ctx_off: np.ndarray,
    util_off: np.ndarray,
    config: QueueConfig,
):

    N = X_ctx_off.shape[0]
    if int(config.offline_epochs) <= 0 or N < 2:
        print("[Offline] skipped (offline_epochs<=0 or N_off<2)")
        return

    router.unfreeze_B(lr_b=float(config.offline_lr_B))

    rng = np.random.RandomState(int(config.seed) + 2024)

    X_t = torch.from_numpy(X_ctx_off).float().to(router.device)
    U_t = torch.from_numpy(util_off).float().to(router.device)

    bs = int(config.supcon_bs)
    replace = (N < bs)

    pos_strategy = str(config.supcon_pos_strategy).lower().strip()
    print(
        f"[Offline] SupCon pretrain: N_off={N}, bs={bs}, epochs={int(config.offline_epochs)}, "
        f"pos_strategy={pos_strategy}"
    )

    supcon_weighted = bool(getattr(config, "supcon_weighted", True))
    supcon_top1_boost = float(getattr(config, "supcon_top1_boost", 0.0))

    for ep in range(1, int(config.offline_epochs) + 1):
        b_idx_np = rng.choice(N, size=bs, replace=replace)
        b_idx = torch.as_tensor(b_idx_np, device=router.device, dtype=torch.long)

        xb = X_t[b_idx]
        ub = U_t[b_idx]

        router.train()
        if router.opt_b is None:
            pass

        router.opt_b.zero_grad(set_to_none=True)

        z = router.forward_ctx_supcon(xb)

        if pos_strategy == "top1":
            y = ub.argmax(dim=1)
            pos_mask = (y.unsqueeze(1) == y.unsqueeze(0))
            pos_mask.fill_diagonal_(False)

        elif pos_strategy == "topr_mass":
            pos_mask = router.build_pos_mask_adaptive_topk(
                util_batch=ub,
                max_k=int(config.supcon_topk_max_k),
                mode="mass",
                beta=float(config.supcon_topk_beta),
                q=float(config.supcon_topk_q),
            )

        elif pos_strategy == "topr_margin":
            pos_mask = router.build_pos_mask_adaptive_topk(
                util_batch=ub,
                max_k=int(config.supcon_topk_max_k),
                mode="margin",
                delta=float(config.supcon_topk_delta),
            )
        else:
            pass

        loss = router.supcon_loss_posmask(z, pos_mask)
        loss.backward()
        router.opt_b.step()

        if (ep % 50 == 0) or (ep == 1) or (ep == int(config.offline_epochs)):
            print(f"[Offline] epoch={ep:4d} | supcon_loss={float(loss.item()):.4f}")

    router.freeze_B()
    router.eval()

def parse_args():
    ap = argparse.ArgumentParser(
        description="B version: offline(20%)=winner-balanced(30%)+random(70%) + SupCon(B) + a_table from offline + online queue on holdout"
    )
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./runs_mnl/exp_B_off7030")
    ap.add_argument("--lam_list", type=float, nargs="+", default=None)
    ap.add_argument("--job_pool_size", type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    config = QueueConfig()
    device = torch.device(config.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_index_path = output_dir / "model_index.json"
    models_fixed: Optional[List[str]] = None

    if model_index_path.exists():
        obj = json.loads(model_index_path.read_text(encoding="utf-8"))
        models_fixed = obj.get("models", None)
        if not isinstance(models_fixed, list) or len(models_fixed) == 0:
            pass
        print(f"[ModelIndex] Loaded fixed model order from: {model_index_path}")
    else:
        print("[ModelIndex] No model_index.json. Will create one in this run.")

    df, models, cost_map = load_routerbench(
        args.data,
        use_cost=bool(config.use_cost),
        models_fixed=models_fixed,
    )
    K_models = len(models)

    if not model_index_path.exists():
        model2idx = {m: i for i, m in enumerate(models)}
        idx2model = {i: m for m, i in model2idx.items()}
        model_index_path.write_text(
            json.dumps({"models": models, "model2idx": model2idx, "idx2model": idx2model}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ModelIndex] Saved model mapping to: {model_index_path}")

    df_train, df_test = train_test_split(
        df,
        test_size=float(config.test_size),
        random_state=int(config.seed),
        shuffle=True,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"[Split] Total={len(df)} | Train={len(df_train)} | Test={len(df_test)}")

    if args.job_pool_size is not None:
        n_pool = int(args.job_pool_size)
        df_train = df_train.sample(n=n_pool, random_state=int(config.seed)).reset_index(drop=True)
        print(f"[Pool] Fixed train job pool size = {n_pool}")

    embedder = Embedder(config.embedder_model, device)
    X_train = embedder.transform(df_train["prompt"].astype(str).tolist())
    d_ctx = int(X_train.shape[1])

    config.d_ctx = d_ctx
    config.n_models = K_models

    if config.use_cost:
        lambdas = args.lam_list if args.lam_list is not None else [float(config.lam_cost)]
    else:
        lambdas = [0.0]

    for lam in lambdas:
        lam = float(lam)
        print(f"\n>> Experiment Lambda = {lam}")

        acc_train, util_train = compute_acc_cost_util_all(
            df=df_train,
            models=models,
            cost_map=cost_map,
            use_cost=bool(config.use_cost),
            lam_cost=lam,
        )

        win_idx, rand_idx, offline_idx, online_idx = build_offline_partition_7030(util_train, config)

        print(f"[Partition] |offline|={offline_idx.size} (ratio={config.offline_total_ratio})")
        print(f"[Partition]   |A_win|={win_idx.size} (winner-balanced ~{int(config.tb_win_frac*100)}%)")
        print(f"[Partition]   |T_rand|={rand_idx.size} (random ~{int(config.tb_rand_frac*100)}%)")
        print(f"[Partition] |E|={online_idx.size} (online stream)")

        (output_dir / f"idx_offline_lam_{lam:.2f}.json").write_text(
            json.dumps([int(i) for i in offline_idx.tolist()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"idx_win_lam_{lam:.2f}.json").write_text(
            json.dumps([int(i) for i in win_idx.tolist()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"idx_rand_lam_{lam:.2f}.json").write_text(
            json.dumps([int(i) for i in rand_idx.tolist()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"idx_online_lam_{lam:.2f}.json").write_text(
            json.dumps([int(i) for i in online_idx.tolist()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        router = MNLRouter(
            d_ctx=d_ctx,
            n_models=K_models,
            d_proj=int(config.d_proj),
            lambda_0=float(config.lambda_0),
            supcon_temp=float(config.supcon_temp),
            device=config.device,
            b_type=str(config.b_type),
            b_hidden_mult=int(config.b_hidden_mult),
            theta_solver="lbfgs",
            lbfgs_max_iter=int(config.lbfgs_max_iter),
            lbfgs_history_size=int(config.lbfgs_history_size),
            lbfgs_line_search=str(config.lbfgs_line_search),
            hist_init_capacity=int(config.hist_init_capacity),
            lr_b=float(config.offline_lr_B),
        ).to(device)

        X_off = X_train[offline_idx]
        U_off = util_train[offline_idx]
        offline_pretrain_B_supcon(router, X_off, U_off, config)

        router.eval()
        with torch.no_grad():
            X_off_t = torch.from_numpy(X_off).float().to(router.device)
            Z_off_t = router.B((X_off_t))
            Z_off = Z_off_t.detach().cpu().numpy().astype(np.float32)

        winners_off = np.argmax(U_off, axis=1).astype(np.int64)

        territories = [np.where(winners_off == k)[0].astype(np.int64) for k in range(K_models)]
        for k in range(K_models):
            if territories[k].size == 0:
                pass

        xi = compute_anchor_centroids(
            X_ctx=Z_off,
            anchors_by_k=territories,
            normalize=bool(config.anchor_xi_normalize),
        )

        S = np.zeros((K_models, K_models), dtype=np.float64)
        for j in range(K_models):
            idx_j = territories[j]
            S[:, j] = U_off[idx_j].mean(axis=0)

        a_table = build_a_table_from_xi_S(
            xi=xi,
            S=S,
            weight_mode=str(config.embed_weight_mode),
            topK=int(config.embed_topK),
            tau=float(config.embed_tau),
            normalize_a=bool(config.embed_a_normalize),
        ).astype(np.float32)

        np.save(output_dir / f"a_table_lam_{lam:.2f}.npy", a_table)
        router.set_a_table(torch.from_numpy(a_table), normalize=False)
        print(f"[Offline] a_table built from OFFLINE pool: shape={a_table.shape}")

        router.reset_for_online()

        X_online = X_train[online_idx]
        acc_online = acc_train[online_idx]
        util_online = util_train[online_idx]

        print(f"[Online] Starting Queue Bandit Simulation: N={len(online_idx)} ...")
        router, avg_reg, Q_reg_T, reg_hist, Q_hist = queue_env(
            X_ctx=X_online,
            acc_mat=acc_online,
            util_mat=util_online,
            config=config,
            router=router,
            model_names=models,
        )

        pd.DataFrame({"cum_regret": reg_hist}).to_csv(output_dir / f"regret_history_lam_{lam:.2f}.csv", index=False)
        pd.DataFrame({"Q_diff": Q_hist}).to_csv(output_dir / f"Qregret_history_lam_{lam:.2f}.csv", index=False)

        print(f"[Done] Lambda {lam} finished. AvgRegret={avg_reg:.6f}, Q_reg_T={Q_reg_T:.3f}")

if __name__ == "__main__":
    main()
