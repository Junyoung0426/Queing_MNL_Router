# B_MNL_Bandit/base_line/5cqb_epsilon/queue_env.py
from __future__ import annotations
import sys 
import os
import math
import importlib.util
from typing import List, Tuple, Optional

import numpy as np


def _load_module_from_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bandit_root_from_this_file() -> str:
    # .../B_MNL_Bandit/base_line/5cqb_epsilon/queue_env.py
    cur_dir = os.path.dirname(os.path.abspath(__file__))       # .../5cqb_epsilon
    base_line_dir = os.path.dirname(cur_dir)                   # .../base_line
    bandit_dir = os.path.dirname(base_line_dir)                # .../B_MNL_Bandit
    return bandit_dir


_BANDIT_DIR = _bandit_root_from_this_file()

# <- 이 블록 추가 (base queue_env 로드 전에)
if _BANDIT_DIR not in sys.path:
    sys.path.insert(0, _BANDIT_DIR)

_BASE_QUEUE_ENV_PATH = os.path.join(_BANDIT_DIR, "queue_env.py")
_base_env_mod = _load_module_from_path("bandit_base_queue_env_mod", _BASE_QUEUE_ENV_PATH)
BaseQueueEnv = _base_env_mod.QueueEnv


class CQBEpsilonEnv(BaseQueueEnv):
    """
    Baseline (5): CQB-ε
      - Keep original forced exploration action (last arrival job + S round-robin)
      - Only override explore coin schedule:
          tau = T/10 (pure exploration phase)
          e ~ Bernoulli(1/sqrt(T)) for t > tau
      - We implement by overriding self.E_prev after each step so that
        at round (t+1) the base env's do_explore condition matches CQB-ε:
            do_explore = (A_prev == 1) and (E_prev == 1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        T = int(getattr(self.config, "max_steps", 100000))
        self._cqb_T = max(1, T)
        self._cqb_tau = max(1, self._cqb_T // 10)              # tau = T/10
        self._cqb_eps = 1.0 / math.sqrt(float(self._cqb_T))     # eps = 1/sqrt(T)

        seed = int(getattr(self.config, "seed", 0))
        self._rng_cqb_eps = np.random.RandomState(seed + 2026)

    def step(self):
        # run the original step (does router/oracle/regret/logging exactly same)
        ret = super().step()

        # overwrite explore coin for NEXT round (t+1)
        if bool(getattr(self, "explore_enabled", True)):
            t = int(self.steps)  # current round index after super().step()

            if (t + 1) <= self._cqb_tau:
                self.E_prev = 1
            else:
                self.E_prev = 1 if (self._rng_cqb_eps.rand() < self._cqb_eps) else 0
        else:
            self.E_prev = 0

        return ret


def queue_env(X_ctx, acc_mat, util_mat, config, router, model_names=None, row_ids=None, sample_ids=None):
    env = CQBEpsilonEnv(
        X_ctx,
        acc_mat,
        util_mat,
        config,
        router,
        model_names=model_names,
        row_ids=row_ids,
        sample_ids=sample_ids,
    )
    return env.run()
