# B_MNL_Bandit/base_line/2random_policy/queue_env.py
from __future__ import annotations

import os
import sys
import importlib.util
from typing import List, Optional

import numpy as np


def _bandit_root_from_here() -> str:
    # Path: .../B_MNL_Bandit/base_line/2random_policy/queue_env.py
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

_base_qenv_mod = _load_module_from_path("bandit_queue_env", os.path.join(_BANDIT, "queue_env.py"))
BaseQueueEnv = _base_qenv_mod.QueueEnv


class QueueEnvAllRandom(BaseQueueEnv):
    """
    Baseline (2): ALL-RANDOM
      - Queue Discipline: Uniform Random (Picks a random job from the queue).
      - Action Selection (S): Uniform Random (Picks a random assortment from all combinations).
      - Training: No updates to the router (no learning).
      - Logging: Maintains standard oracle progression, regret calculation, and logging formats.
    """

    def step(self):
        self.steps += 1
        t = self.steps

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

        self._last_S_oracle = None
        self._last_ctx_oracle = None
        self._last_uid_oracle = None
        self._last_choice_oracle = None
        self._last_oracle_metrics = None

        # -----------------------------
        # 1. Router Decision: ALL RANDOM
        # -----------------------------
        if len(self.queue_router) > 0:
            self.cnt_decision += 1
            self.cnt_explore += 1  # All-Random is considered purely exploratory

            # (i) Random job selection from the router queue
            qpos = int(self.rng_router.randint(len(self.queue_router)))
            uid_r, ctx_idx_r = self.queue_router.pop(qpos)
            x_ctx = self.X_ctx[ctx_idx_r]

            # (ii) Random assortment selection from C
            c_idx = int(self.rng_router.randint(self.C_size))
            S_t = list(self.all_combis[c_idx])

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

            # No updates performed. Re-insert if not departed.
            if not departed:
                self.queue_router.append((uid_r, ctx_idx_r))

            # Compatibility for train.py
            try:
                self.router._T = int(self.cnt_decision)
            except Exception:
                pass

            # Minimal debug metrics
            self._last_router_metrics = {
                "explore_used": True,
                "dep_alg_true": float(dep_alg_true),
                "true_r_alg": self._true_r_list(self.r_mat[ctx_idx_r], S_t),
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
                "true_r_star": self._true_r_list(self.r_mat[ctx_star], S_star),
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

        # State updates for compatibility (unused but maintained)
        self.A_prev = A_curr
        self.E_prev = 0

        # Periodic Logging
        if (self.steps % self.config.log_every == 0):
            avg_reg = self.cum_regret / max(1, self.steps)
            q_gap = self.Q_regret_history[-1] if self.Q_regret_history else 0.0
            print(f"[ALL_RANDOM step={self.steps}] regret(avg)={avg_reg:.6f} Q-gap={q_gap:.3f} Q_r={Q_r} Q_o={Q_o}")

        return False


def queue_env(X_ctx, acc_mat, util_mat, config, router, model_names=None, row_ids=None, sample_ids=None):
    env = QueueEnvAllRandom(
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