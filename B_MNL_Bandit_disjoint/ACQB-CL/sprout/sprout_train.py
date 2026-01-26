import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

import sys
import argparse
import importlib.util
import pandas as pd
from typing import Tuple, List, Dict, Optional


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
project_dir = os.path.dirname(root_dir)
data_dir = os.path.join(project_dir, "Data")
DEFAULT_DATA = os.path.join(data_dir, "sprout_dataset.csv")


for p in (current_dir, root_dir):
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, current_dir)
sys.path.insert(1, root_dir)


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec: {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


queue_config_path = os.path.join(project_dir, "B_MNL_Bandit_disjoint", "ACQB", "sprout",  "queue_config.py")
train_path = os.path.join(root_dir, "train.py")

if not os.path.exists(queue_config_path):
    raise FileNotFoundError(queue_config_path)
if not os.path.exists(train_path):
    raise FileNotFoundError(train_path)

_load_module_from_path("queue_config", queue_config_path)
_load_module_from_path("train", train_path)

from train import add_common_args, run_pipeline, set_full_determinism
from queue_config import QueueConfig


def infer_models_and_cost_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    base = {"orig_row", "sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "oracle_model"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if str(c).endswith("|total_cost")}
    return models, cost_map



def load_routerbench_like_csv(
    path: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_csv(path)
    if "orig_row" not in df.columns:
        df["orig_row"] = np.arange(len(df), dtype=np.int64)

    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다")

    models_all, cost_map_all = infer_models_and_cost_map(df)

    if models_fixed is None:
        models = models_all
    else:
        models = list(models_fixed)
        missing = [m for m in models if m not in models_all]
        if missing:
            raise ValueError(f"models_fixed 중 score 컬럼이 없는 모델이 있다: {missing}")

    if use_cost:
        missing_cost = [m for m in models if m not in cost_map_all]
        if missing_cost:
            raise ValueError(f"use_cost=True인데 |total_cost 컬럼이 없는 모델이 있다: {missing_cost}")
        cost_map = {m: cost_map_all[m] for m in models}
    else:
        cost_map = {}

    keep = []
    for c in ["orig_row","sample_id", "prompt", "eval_name"]:
        if c in df.columns:
            keep.append(c)

    keep += models
    if use_cost:
        keep += [cost_map[m] for m in models]

    seen = set()
    keep = [c for c in keep if not (c in seen or seen.add(c))]

    df = df[keep].copy()
    return df, models, cost_map


def parse_args():
    ap = argparse.ArgumentParser(description="Train on RouterBench-like CSV with common pipeline")
    ap.add_argument("--data", type=str, default=DEFAULT_DATA)
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()
    config.b_type = "mlp"
    args.b_type = "mlp"
    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        config.embedder_model = str(args.embedder_model)

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_routerbench_like_csv(args.data, use_cost=bool(config.use_cost))
    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()
