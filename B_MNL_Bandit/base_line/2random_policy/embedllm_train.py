# B_MNL_Bandit/base_line/2random_policy/embedllm_train.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse

cur_dir = os.path.dirname(os.path.abspath(__file__))
base_line_dir = os.path.dirname(cur_dir)
common_dir = os.path.join(base_line_dir, "_common")
sys.path.insert(0, common_dir)

from baseline_common import load_queue_config, add_common_args, set_full_determinism, run_pipeline_with_env
from baseline_loaders import load_embedllm_hf_csv_pivot

from queue_env import queue_env as all_random_queue_env


def parse_args():
    ap = argparse.ArgumentParser(description="Baseline (2) ALL-RANDOM on EmbedLLM(HF csv pivot)")
    ap.add_argument("--data", type=str, default="RZ412/EmbedLLM")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--max_prompts", type=int, default=5000)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--default_B", type=float, default=7.0)
    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()

    QueueConfig = load_queue_config()
    config = QueueConfig()

    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        config.embedder_model = str(args.embedder_model)

    config.explore_enabled = False
    config.debug_verbose = False

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_embedllm_hf_csv_pivot(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        use_cost=bool(config.use_cost),
        max_prompts=int(args.max_prompts),
        seed=int(config.seed),
        chunksize=int(args.chunksize),
        default_B=float(args.default_B),
    )
    run_pipeline_with_env(df, models, cost_map, args, config, baseline_queue_env_func=all_random_queue_env)


if __name__ == "__main__":
    main()
