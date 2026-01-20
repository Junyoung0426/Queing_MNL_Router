# B_MNL_Bandit/base_line/5cqb_epsilon/routerbench_train.py
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
from baseline_loaders import load_routerbench_pkl

from queue_env import queue_env as cqb_eps_queue_env


def parse_args():
    ap = argparse.ArgumentParser(description="Baseline (5) CQB-epsilon on RouterBench (pkl)")
    ap.add_argument("--data", type=str, required=True, help="routerbench pkl path")
    ap.add_argument("--models", type=str, nargs="+", default=None, help="optional fixed model list")
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

    # CQB-ε uses exploration schedule (tau + eps), so keep explore_enabled=True
    config.explore_enabled = True
    config.debug_verbose = False

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_routerbench_pkl(
        path=str(args.data),
        use_cost=bool(config.use_cost),
        models_fixed=list(args.models) if args.models is not None else None,
    )

    run_pipeline_with_env(df, models, cost_map, args, config, baseline_queue_env_func=cqb_eps_queue_env)


if __name__ == "__main__":
    main()
