import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse

script_name = os.path.basename(__file__)
dataset_name = script_name.split("_train.py")[0]

if not dataset_name or dataset_name == script_name:
    raise ValueError(f"Filename format error: {script_name}")

cur_dir = os.path.dirname(os.path.abspath(__file__))
base_line_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(base_line_dir)
project_root = os.path.dirname(root_dir)
data_dir = os.path.join(project_root, "Data")

target_config_dir = os.path.join(root_dir, "ACQB", dataset_name)
if not os.path.exists(target_config_dir):
    raise FileNotFoundError(f"Config dir not found: {target_config_dir}")

common_dir = os.path.join(base_line_dir, "_common")
qucb_env_dir = cur_dir

sys.path.insert(0, target_config_dir)
sys.path.insert(1, common_dir)
sys.path.insert(2, qucb_env_dir)
sys.path.insert(3, root_dir)

sys.modules.pop("queue_config", None)
sys.modules.pop("queue_env", None)

from queue_config import QueueConfig
from baseline_common import add_common_args, set_full_determinism, run_pipeline_with_env
from baseline_loaders import load_standardized_csv

try:
    from queue_env import queue_env as qucb_queue_env
except ImportError:
    raise ImportError(f"Cannot find queue_env.py in {qucb_env_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    default_path = os.path.join(data_dir, "sprout_dataset.csv")
    ap.add_argument("--data", type=str, default=default_path)
    ap.add_argument("--explore_S_policy", type=str, default="round_robin", choices=["round_robin", "random"])
    ap.add_argument("--qucb_s", type=float, default=1.0)
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

    config.assort_K = 1
    config.explore_enabled = False
    config.debug_verbose = False
    config.qucb_explore_S_policy = str(args.explore_S_policy)
    config.qucb_s = float(args.qucb_s)

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_standardized_csv(
        path=str(args.data),
        use_cost=bool(config.use_cost)
    )

    run_pipeline_with_env(
        df, 
        models, 
        cost_map, 
        args, 
        config, 
        baseline_queue_env_func=qucb_queue_env
    )

if __name__ == "__main__":
    main()