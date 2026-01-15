#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from queue_config import QueueConfig
from train import add_common_args, run_pipeline



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
    max_prompts: int,
    seed: int = 0,
    chunksize: int = 200_000,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    from huggingface_hub import hf_hub_download  # pip install huggingface_hub

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

    rng = np.random.RandomState(int(seed))
    store: Dict[int, Dict[str, Any]] = {}
    completed: List[int] = []

    reader = pd.read_csv(csv_path, chunksize=int(chunksize))
    for chunk in reader:
        if "prompt_id" not in chunk.columns or "model_name" not in chunk.columns or "label" not in chunk.columns or "prompt" not in chunk.columns:
            raise ValueError(f"[EmbedLLM] required columns missing in {filename}")

        for r in chunk.itertuples(index=False):
            if len(completed) >= int(max_prompts):
                break

            pid = int(getattr(r, "prompt_id"))
            mname = str(getattr(r, "model_name"))
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
                if rec["cnt"] == K:
                    completed.append(pid)

        if len(completed) >= int(max_prompts):
            break

    rows: List[Dict[str, Any]] = []
    for pid in completed:
        rec = store.get(pid)
        if rec is None or rec["cnt"] != K:
            continue
        labels = rec["labels"]
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
        raise ValueError("[EmbedLLM] 0 complete prompts collected. max_prompts를 늘리거나 split을 바꿔야 한다")

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    return df, models, cost_map


def parse_args():
    ap = argparse.ArgumentParser(description="Train on EmbedLLM(HF) with common pipeline")
    ap.add_argument("--data", type=str, default="RZ412/EmbedLLM")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--max_prompts", type=int, default=5000,
                    help="완전한(prompt×all models) row를 이 개수만 수집하고 중단한다")
    ap.add_argument("--chunksize", type=int, default=200_000)
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()

    df, models, cost_map = load_embedllm_hf_csv_pivot(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        use_cost=bool(config.use_cost),
        max_prompts=int(args.max_prompts),
        seed=int(config.seed),
        chunksize=int(args.chunksize),
    )
    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()
