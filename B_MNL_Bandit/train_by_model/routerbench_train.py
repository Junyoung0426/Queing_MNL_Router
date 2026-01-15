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
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "oracle_model"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return models, cost_map


def load_routerbench_pkl(
    path: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_pickle(path)

    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다")

    models_all, cost_map_all = infer_models_and_cost_map(df)

    if models_fixed is None:
        models = models_all
    else:
        models = list(models_fixed)
        missing = [m for m in models if m not in models_all]
        if missing:
            raise ValueError(f"models_fixed 중 total_cost prefix에 없는 모델이 있다: {missing}")

    cost_map = {m: cost_map_all[m] for m in models}

    keep = []
    for c in ["sample_id", "prompt", "eval_name"]:
        if c in df.columns:
            keep.append(c)

    keep += models
    if use_cost:
        keep += [cost_map[m] for m in models]

    # 중복 제거
    seen = set()
    keep = [c for c in keep if not (c in seen or seen.add(c))]

    df = df[keep].copy()
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
