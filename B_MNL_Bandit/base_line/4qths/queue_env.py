# B_MNL_Bandit/base_line/4qths_policy/queue_env.py
from __future__ import annotations

import os
import sys
import importlib.util
from typing import List

import numpy as np


def _bandit_root_from_here() -> str:
    # Path: .../B_MNL_Bandit/base_line/4qths_policy/queue_env.py
    policy_dir = os.path.dirname(os.path.abspath(__file__))
    base_line_dir = os.path.dirname(policy_dir)
    bandit_dir = os.path.dirname(base_line_dir)
    return bandit_dir


def _load_module_from_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Force load the base QueueEnv from the root to avoid name collision
_BANDIT = _bandit_root_from_here()
if _BANDIT not in sys.path:
    sys.path.insert(0, _BANDIT)

_base_qenv_mod = _load_module_from_path(
    "bandit_queue_env_for_qths",
    os.path.join(_BANDIT, "queue_env.py"),
)
BaseQueueEnv = _base_qenv_mod.QueueEnv


class QueueEnvQThS(BaseQueueEnv):
    """
    Baseline (4): QThS (Queueing Thompson Sampling)
      - Implements Algorithm 2 style from Krishnasamy et al. (2021).
      - K=1 assumption (Single arm selection).
      - Queue Discipline: FIFO (Always selects the head of the queue).
      - Exploration Schedule: Bernoulli( min(1, 3 * N * (log t)^2 / t ) ).
      - Exploitation: Samples from Beta(succ+1, fail+1) for each arm and picks the max.
      - Updates: Tracks T_S (counts) and succ_S (successes).
      - Note: No Neural Network training is performed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if int(self.K) != 1:
            raise ValueError(
                f"[QThS] assort_K must be 1, got K={self.K}. "
                f"Please set config.assort_K=1 in the train script."
            )

        # Exploration policy for S: "round_robin" or "random"
        self.explore_S_policy: str = str(getattr(self.config, "qths_explore_S_policy", "round_robin"))
        if self.explore_S_policy not in ("round_robin", "random"):
            raise ValueError(f"[QThS] unknown policy={self.explore_S_policy}")

        # Arm statistics (indexed by all_combis index)
        self.T_S = np.zeros(self.C_size, dtype=np.int64)      # Pull counts
        self.succ_S = np.zeros(self.C_size, dtype=np.int64)   # Success counts

        self.rr_idx = 0

        # RNG for Beta sampling (Determinism)
        self.rng_beta = np.random.RandomState(int(self.config.seed) + 999)

    @staticmethod
    def _log_squared_safe(t: int) -> float:
        if t <= 1:
            return 0.0
        return float(np.log(t) ** 2)

    def _sample_explore_coin(self, t: int) -> bool:
        """
        Calculates exploration probability: min(1, (3 * N * log^2(t)) / t)
        """
        if t <= 1:
            return True

        log_t_sq = self._log_squared_safe(t)
        N_arms = float(self.C_size)
        p = (3.0 * N_arms * log_t_sq) / float(t)
        p = float(min(1.0, max(0.0, p)))
        return bool(self.rng_explore.rand() < p)

    def _pick_S_explore(self) -> int:
        if self.explore_S_policy == "random":
            return int(self.rng_explore.randint(self.C_size))
        c = int(self.rr_idx)
        self.rr_idx = (self.rr_idx + 1) % int(self.C_size)
        return c

    def _pick_S_thompson(self) -> int:
        """
        Exploitation: Sample ~ Beta(succ+1, fail+1) for each arm.
        Returns the index of the arm with the highest sample.
        """
        
        T = self.T_S
        succ = self.succ_S
        fail = T - succ

        a = succ.astype(np.float64) + 1.0
        b = fail.astype(np.float64) + 1.0

        samples = self.rng_beta.beta(a, b)
        return int(np.argmax(samples))

    def step(self):
        self.steps += 1
        t = int(self.steps)

        # Synchronize u_dep for router/oracle
        u_dep = float(self.rng_feedback.rand())

        # Snapshot for regret calculation
        X_t_snapshot_ctx = [ctx for (_, ctx) in self.queue_router]

        R_alg_t = 0.0
        R_star_t = 0.0

        # Reset debug caches
        self._last_S_router = None
        self._last_ctx_router = None
        self._last_uid_router = None
        self._last_choice_router = None
        self._last_router_metrics = None

        self._last_choice_oracle = None
        self._last_oracle_metrics = None

        # -----------------------------
        # 1. Router Decision: QThS
        # -----------------------------
        if len(self.queue_router) > 0:
            self.cnt_decision += 1

            # Job Selection: FIFO (Head of the queue)
            uid_r, ctx_idx_r = self.queue_router.pop(0)

            # Exploration Logic (Independent of config flag)
            do_explore = self._sample_explore_coin(t)
            
            if do_explore:
                self.cnt_explore += 1
                c_idx = self._pick_S_explore()
            else:
                c_idx = self._pick_S_thompson()

            S_t: List[int] = list(self.all_combis[int(c_idx)])

            self._last_S_router = list(S_t)
            self._last_ctx_router = int(ctx_idx_r)
            self._last_uid_router = int(uid_r)

            # Calculate true departure rate
            dep_alg_true = float(self._dep_rate_from_odds_row(self.odds_mat[ctx_idx_r], S_t))
            R_alg_t = dep_alg_true

            # Sample feedback
            departed, chosen_model, j_local = self.sample_mnl_choice_u(ctx_idx_r, S_t, u=u_dep)
            self._last_choice_router = (bool(departed), int(chosen_model) if chosen_model is not None else None)

            # Log stats
            self.dep_prob_router_hist.append(float(dep_alg_true))
            self.dep_event_router_hist.append(1.0 if departed else 0.0)
            self.dep_router_step_idx.append(int(t))

            # Update Arm Statistics (Thompson Sampling)
            y = 1 if departed else 0
            self.T_S[int(c_idx)] += 1
            self.succ_S[int(c_idx)] += int(y)

            # Re-insert if not departed
            if not departed:
                self.queue_router.append((uid_r, ctx_idx_r))

            # Minimal debug metrics
            self._last_router_metrics = {
                "explore_used": bool(do_explore),
                "dep_alg_true": float(dep_alg_true),
            }

        # -----------------------------
        # 2. Oracle Queue Progression
        # -----------------------------
        if len(self.queue_oracle) > 0:
            q_ctx_arr = np.asarray([ctx for (_, ctx) in self.queue_oracle], dtype=np.int64)
            best_pos = int(np.argmax(self.max_departure_rates[q_ctx_arr]))

            uid_o, ctx_idx_o = self.queue_oracle.pop(best_pos)
            S_o = list(self.all_combis[int(self.best_S_idx[ctx_idx_o])])

            dep_oracle_true = float(self._dep_rate_from_odds_row(self.odds_mat[ctx_idx_o], S_o))
            departed_o, chosen_o, _ = self.sample_mnl_choice_u(ctx_idx_o, S_o, u=u_dep)
            self._last_choice_oracle = (bool(departed_o), int(chosen_o) if chosen_o is not None else None)

            self.dep_prob_oracle_hist.append(float(dep_oracle_true))
            self.dep_event_oracle_hist.append(1.0 if departed_o else 0.0)
            self.dep_oracle_step_idx.append(int(t))

            if not departed_o:
                self.queue_oracle.append((uid_o, ctx_idx_o))

        # -----------------------------
        # 3. Regret Calculation (Queue-Max Oracle)
        # -----------------------------
        if len(X_t_snapshot_ctx) > 0:
            ctx_arr = np.asarray(X_t_snapshot_ctx, dtype=np.int64)
            ctx_star = int(ctx_arr[np.argmax(self.max_departure_rates[ctx_arr])])
            R_star_t = float(self.max_departure_rates[ctx_star])

            c_star = int(self.best_S_idx[ctx_star])
            S_star = list(self.all_combis[c_star])
            dep_star = float(self._dep_rate_from_odds_row(self.odds_mat[ctx_star], S_star))

            self.dep_prob_star_hist.append(float(dep_star))
            self.dep_star_step_idx.append(int(t))

            self._last_oracle_metrics = {
                "ctx_star": int(ctx_star),
                "S_star": list(S_star),
                "dep_star": float(dep_star),
            }

        self.cum_regret += (float(R_star_t) - float(R_alg_t))
        self.regret_history.append(self.cum_regret)

        # -----------------------------
        # 4. Job Arrival
        # -----------------------------
        A_curr = 0
        if self.rng_arrival.rand() < self.config.arrival_rate:
            ctx_idx_new = int(self.rng_arrival.choice(self.job_pool))
            uid_new = self.next_uid
            self.next_uid += 1
            self.queue_router.append((uid_new, ctx_idx_new))
            self.queue_oracle.append((uid_new, ctx_idx_new))
            self.last_arrival_uid = uid_new
            A_curr = 1

        Q_r = len(self.queue_router)
        Q_o = len(self.queue_oracle)
        self.Q_regret_history.append((Q_r - Q_o))
        self.Q_router_history.append(Q_r)
        self.Q_oracle_history.append(Q_o)

        # State update for compatibility (though unused in QThS)
        self.A_prev = A_curr
        self.E_prev = 0

        # Periodic Logging
        if (self.steps % self.config.log_every == 0):
            avg_reg = self.cum_regret / max(1, self.steps)
            q_gap = self.Q_regret_history[-1] if self.Q_regret_history else 0.0
            print(f"[QThS step={self.steps}] regret(avg)={avg_reg:.6f} Q-gap={q_gap:.3f} Q_r={Q_r} Q_o={Q_o}")

        return False


def queue_env(X_ctx, acc_mat, util_mat, config, router, model_names=None, row_ids=None, sample_ids=None):
    env = QueueEnvQThS(
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