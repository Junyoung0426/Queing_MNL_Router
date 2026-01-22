#B_MNL_Bandit_disjoint/train_by_model/embedllm/embedllm_train.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
current_dir = os.path.dirname(os.path.abspath(__file__))          # .../train_by_model/embedllm
parent_dir = os.path.dirname(current_dir)                         # .../train_by_model
root_dir = os.path.dirname(parent_dir)                            # .../B_MNL_Bandit_disjoint

for p in (current_dir, root_dir):
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, current_dir)  
sys.path.insert(1, root_dir)      

sys.modules.pop("queue_config", None)
sys.modules.pop("train", None)


from train import add_common_args, run_pipeline, set_full_determinism
from queue_config import QueueConfig

# 확인
import queue_config as qc
print("[queue_config loaded from]", qc.__file__)


def _parse_size_b(model_name: str, default: float = 7.0) -> float:
    import re
    s = str(model_name).lower()
    m = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1))
    return float(default)


def load_embedllm_hf_csv_pivot(
    dataset_id: str,
    split: str,
    use_cost: bool,
    # max_prompts 인자 제거됨
    seed: int = 0,
    chunksize: int = 200_000,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    from huggingface_hub import hf_hub_download  # pip install huggingface_hub

    print(f"🔄 Loading FULL dataset from {dataset_id} ({split})...")

    split2file = {"train": "train.csv", "validation": "val.csv", "val": "val.csv", "test": "test.csv"}
    filename = split2file.get(split, split)
    csv_path = hf_hub_download(repo_id=dataset_id, filename=filename, repo_type="dataset")

    # model_order.csv로 모델 목록을 고정한다
    model_order_path = hf_hub_download(repo_id=dataset_id, filename="model_order.csv", repo_type="dataset")
    model_df = pd.read_csv(model_order_path)
    if "model_name" not in model_df.columns:
        raise ValueError("[EmbedLLM] model_order.csv missing 'model_name'")
    models = model_df["model_name"].astype(str).tolist()
    K = len(models)
    model_set = set(models)

    # cost proxy: 모델 규모(B)
    model_cost_proxy = {m: _parse_size_b(m) for m in models}

    # 전체 데이터를 담을 딕셔너리
    store: Dict[int, Dict[str, Any]] = {}
    completed: List[int] = []

    # 청크 단위로 읽지만, 중단 없이 끝까지 읽음
    reader = pd.read_csv(csv_path, chunksize=int(chunksize))
    
    total_rows_processed = 0
    
    for i, chunk in enumerate(reader):
        if "prompt_id" not in chunk.columns or "model_name" not in chunk.columns or "label" not in chunk.columns or "prompt" not in chunk.columns:
            raise ValueError(f"[EmbedLLM] required columns missing in {filename}")

        for r in chunk.itertuples(index=False):
            pid = int(getattr(r, "prompt_id"))
            mname = str(getattr(r, "model_name"))
            
            # 정의된 모델 목록에 없는 모델은 무시
            if mname not in model_set:
                continue

            rec = store.get(pid)
            if rec is None:
                rec = {
                    "prompt": str(getattr(r, "prompt")),
                    "labels": {},
                    "cnt": 0,
                }
                store[pid] = rec

            labels = rec["labels"]
            if mname not in labels:
                labels[mname] = float(getattr(r, "label"))
                rec["cnt"] += 1
                
                # 모든 모델(K개)의 점수가 다 모였으면 완료 리스트에 추가
                if rec["cnt"] == K:
                    completed.append(pid)
        
        total_rows_processed += len(chunk)
        if i % 5 == 0:
             print(f"   ...processed {total_rows_processed} raw rows, found {len(completed)} complete prompts so far")

    print(f"✅ Raw processing done. Total complete prompts found: {len(completed)}")

    # DataFrame 생성
    rows: List[Dict[str, Any]] = []
    for pid in completed:
        rec = store.get(pid)
        if rec is None or rec["cnt"] != K:
            continue
        labels = rec["labels"]
        
        # 안전장치: 혹시라도 모델이 누락되었는지 재확인
        if any(m not in labels for m in models):
            continue

        row: Dict[str, Any] = {
            "sample_id": pid,
            "prompt": rec["prompt"],
            "eval_name": "embedllm",
        }
        for m in models:
            row[m] = float(labels[m])
            if use_cost:
                row[f"{m}|total_cost"] = float(model_cost_proxy[m])
        rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("[EmbedLLM] 0 complete prompts collected. 데이터셋 확인 필요.")

    print(f"✅ Final DataFrame created: {len(df)} samples.")
    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    return df, models, cost_map


def parse_args():
    ap = argparse.ArgumentParser(description="Train on EmbedLLM(HF) with common pipeline (FULL DATA)")
    ap.add_argument("--data", type=str, default="RZ412/EmbedLLM")
    ap.add_argument("--hf_split", type=str, default="train")
    # --max_prompts 인자 제거됨
    ap.add_argument("--chunksize", type=int, default=200_000)
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()
    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        config.embedder_model = str(args.embedder_model)

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_embedllm_hf_csv_pivot(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        use_cost=bool(config.use_cost),
        seed=int(config.seed),
        chunksize=int(args.chunksize),
    )
    
    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()