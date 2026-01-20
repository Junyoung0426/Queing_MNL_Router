# B_MNL_Bandit/base_line/_common/baseline_common.py
from __future__ import annotations

import os
import sys
import importlib.util
from typing import Any, Callable, Tuple


def _bandit_root_from_common_file() -> str:
    # .../B_MNL_Bandit/base_line/_common/baseline_common.py
    common_dir = os.path.dirname(os.path.abspath(__file__))     # .../base_line/_common
    base_line_dir = os.path.dirname(common_dir)                 # .../base_line
    bandit_dir = os.path.dirname(base_line_dir)                 # .../B_MNL_Bandit
    return bandit_dir


def _load_module_from_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_base_train():
    """
    B_MNL_Bandit/train.py를 파일 경로로 강제 로드한다.
    """
    bandit_dir = _bandit_root_from_common_file()
    if bandit_dir not in sys.path:
        sys.path.insert(0, bandit_dir)

    train_path = os.path.join(bandit_dir, "train.py")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.py not found: {train_path}")

    return _load_module_from_path("bandit_train", train_path)


def load_queue_config():
    """
    무조건 B_MNL_Bandit/queue_config.py를 파일 경로로 강제 로드한다.
    """
    bandit_dir = _bandit_root_from_common_file()
    if bandit_dir not in sys.path:
        sys.path.insert(0, bandit_dir)

    qc_path = os.path.join(bandit_dir, "queue_config.py")
    if not os.path.exists(qc_path):
        raise FileNotFoundError(f"queue_config.py not found: {qc_path}")

    qc_mod = _load_module_from_path("bandit_queue_config", qc_path)
    if not hasattr(qc_mod, "QueueConfig"):
        raise AttributeError(f"QueueConfig not found in: {qc_path}")

    return qc_mod.QueueConfig


def add_common_args(ap):
    base_train = load_base_train()
    return base_train.add_common_args(ap)


def set_full_determinism(seed: int):
    base_train = load_base_train()
    return base_train.set_full_determinism(int(seed))


def run_pipeline_with_env(
    df,
    models,
    cost_map,
    args,
    config,
    baseline_queue_env_func: Callable[..., Tuple[Any, ...]],
):
    """
    B_MNL_Bandit/train.run_pipeline을 호출하되, queue_env만 baseline env로 교체한다.
    """
    base_train = load_base_train()
    base_train.queue_env = baseline_queue_env_func
    return base_train.run_pipeline(df, models, cost_map, args, config)
