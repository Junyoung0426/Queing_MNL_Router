#train_by_model/routerbench_train.py

import argparse
from typing import Dict, List, Tuple

import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from queue_config import QueueConfig
from train import add_common_args, run_pipeline

from typing import Tuple, List, Dict, Optional
import pandas as pd

def infer_models_and_cost_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return models, cost_map


def load_routerbench_pkl(
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
            print(f"[Warn] models_fixed에 있지만 데이터에 없는 모델: {missing}")
        if extra:
            print(f"[Warn] 데이터에 있지만 models_fixed에 없는 모델: {extra}")
        models = list(models_fixed)

    subset_cols = ["prompt"] + list(models)

    if use_cost:
        for m in models:
            col = cost_map.get(m, None)
            if col is not None and col in df.columns:
                subset_cols.append(col)
            elif m not in cost_map:
                print(f"[Warn] Model '{m}' has no cost column. Assuming cost=0.0")

    df = df.dropna(subset=subset_cols).reset_index(drop=True)
    return df, models, cost_map


def parse_args():
    ap = argparse.ArgumentParser(description="Train on RouterBench(pkl) with common pipeline")
    ap.add_argument("--data", type=str, required=True, help="path to routerbench .pkl")
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()

    df, models, cost_map = load_routerbench_pkl(args.data, use_cost=bool(config.use_cost))
    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()
