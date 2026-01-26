from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


TAU_RE = re.compile(r"tn([0-9\.m\-]+)_tp([0-9\.m\-]+)")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if x == x]
    return float(np.mean(ys)) if ys else float("nan")


def _nanmax(xs: List[float]) -> float:
    ys = [x for x in xs if x == x]
    return float(np.max(ys)) if ys else float("nan")


def _find_tau_tag(p: Path) -> Optional[str]:
    for part in p.parts:
        if TAU_RE.match(part):
            return part
    return None


def _q_abs_tail_mean(run_dir: Path, tail_frac: float = 0.2) -> float:
    q_files = sorted(run_dir.glob("Qregret_history_lam_*.csv"))
    if not q_files:
        return float("nan")

    vals = []
    for qp in q_files:
        try:
            df = pd.read_csv(qp)
        except Exception:
            continue
        if "Q_diff" not in df.columns:
            continue
        d = df["Q_diff"].to_numpy(dtype=float)
        if d.size == 0:
            continue
        a = np.abs(d)
        st = int((1.0 - float(tail_frac)) * a.size)
        st = min(max(st, 0), a.size)
        vals.append(float(np.mean(a[st:])))
    return _nanmean(vals)


def _score_run(run_dir: Path, expected_lams: int = 3) -> Optional[Dict[str, Any]]:
    summ = sorted(run_dir.glob("summary_lam_*.json"))
    if not summ:
        return None

    regrets = []
    for f in summ:
        d = _load_json(f)
        regrets.append(_safe_float(d.get("avg_regret")))

    regrets = [x for x in regrets if x == x]
    if not regrets:
        return None

    complete = 1 if len(summ) >= int(expected_lams) else 0
    return {
        "run_dir": str(run_dir),
        "complete": complete,
        "run_worst_avg_regret": float(np.max(regrets)),
        "run_mean_avg_regret": float(np.mean(regrets)),
        "Q_gap_abs_tail_mean": float(_q_abs_tail_mean(run_dir)),
    }


def main(root: str, topk: int = 3, expected_lams: int = 3, require_complete: bool = True):
    rootp = Path(root).expanduser().resolve()
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for d in rootp.rglob("*"):
        if not d.is_dir():
            continue
        if not any(d.glob("summary_lam_*.json")):
            continue
        tau = _find_tau_tag(d)
        if tau is None:
            continue
        s = _score_run(d, expected_lams=expected_lams)
        if s is None:
            continue
        groups.setdefault(tau, []).append(s)

    combos = []
    for tau, runs in groups.items():
        if require_complete and (not all(int(r["complete"]) == 1 for r in runs)):
            continue
        worsts = [_safe_float(r["run_worst_avg_regret"]) for r in runs]
        means = [_safe_float(r["run_mean_avg_regret"]) for r in runs]
        qtail = [_safe_float(r["Q_gap_abs_tail_mean"]) for r in runs]
        combos.append({
            "tau": tau,
            "n_runs": int(len(runs)),
            "worst_over_all": _nanmax(worsts),
            "mean_over_all": _nanmean(means),
            "Q_gap_abs_tail_mean": _nanmean(qtail),
        })

    if not combos:
        print("no combos found")
        return

    def key(x):
        return (
            x["worst_over_all"],
            x["mean_over_all"],
            x["Q_gap_abs_tail_mean"],
        )

    combos.sort(key=key)
    out = {
        "root": str(rootp),
        "ranking_rule": "worst(avg_regret) -> mean(avg_regret) -> Q_abs_tail_mean",
        "topk": int(topk),
        "top": combos[:max(1, int(topk))],
    }

    out_path = rootp / f"best_top{int(topk)}_tau.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", str(out_path))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(p, topk=k)
