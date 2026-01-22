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
    raise ValueError(f"Filename does not match '_train.py' format: {script_name}")

print(f"[Info] Detected dataset name from filename: '{dataset_name}'")

cur_dir = os.path.dirname(os.path.abspath(__file__))
base_line_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(base_line_dir)

target_config_dir = os.path.join(root_dir, "train_by_model", dataset_name)

if not os.path.exists(target_config_dir):
    raise FileNotFoundError(f"Config folder not found: {target_config_dir}")

common_dir = os.path.join(base_line_dir, "_common")

sys.path.insert(0, target_config_dir)
sys.path.insert(1, common_dir)
sys.path.insert(2, root_dir)

sys.modules.pop("queue_config", None)

from queue_config import QueueConfig
sys.path.insert(0, cur_dir)
from baseline_common import add_common_args, set_full_determinism, run_pipeline_with_env
from baseline_loaders import load_mixinstruct_hf

try:
    from queue_env import queue_env as qths_queue_env
except ImportError:
    sys.path.append(root_dir)
    from queue_env import queue_env as qths_queue_env


def parse_args():
    ap = argparse.ArgumentParser(description="Baseline (4) QThS on MixInstruct(HF)")
    ap.add_argument("--data", type=str, default="llm-blender/mix-instruct")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--metric", type=str, default="bertscore")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--rho_out", type=float, default=1.0)
    ap.add_argument("--denom_C", type=float, default=1e6)
    ap.add_argument("--token_mode", type=str, default="chars4", choices=["chars4", "whitespace"])
    ap.add_argument("--default_B", type=float, default=7.0)
    ap.add_argument("--use_moe_effective", type=int, default=1)
    ap.add_argument("--moe_active_experts", type=int, default=2)
    ap.add_argument("--explore_S_policy", type=str, default="round_robin", choices=["round_robin", "random"])
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()

    config = QueueConfig()

    import queue_config as qc
    print(f"[Info] Loaded QueueConfig from: {qc.__file__}")

    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        config.embedder_model = str(args.embedder_model)

    config.assort_K = 1
    config.explore_enabled = False
    config.debug_verbose = False
    config.qths_explore_S_policy = str(args.explore_S_policy)

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_mixinstruct_hf(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        metric=str(args.metric),
        use_cost=bool(config.use_cost),
        max_rows=args.max_rows,
        rho_out=float(args.rho_out),
        denom_C=float(args.denom_C),
        token_mode=str(args.token_mode),
        default_B=float(args.default_B),
        use_moe_effective=bool(int(args.use_moe_effective)),
        moe_active_experts=int(args.moe_active_experts),
    )
    
    run_pipeline_with_env(
        df, 
        models, 
        cost_map, 
        args, 
        config, 
        baseline_queue_env_func=qths_queue_env
    )


if __name__ == "__main__":
    main()