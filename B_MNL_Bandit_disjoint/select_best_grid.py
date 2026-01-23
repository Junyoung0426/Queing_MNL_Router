import json
from pathlib import Path
import numpy as np
import pandas as pd


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def _is_nan(x):
    return not (x == x)


def _read_q_metrics(combo_dir: Path):
    q_files = sorted(combo_dir.glob("Qregret_history_lam_*.csv"))
    if not q_files:
        return float("nan"), float("nan"), float("nan")

    tail_means = []
    p95s = []
    gap_abs_means = []

    for qp in q_files:
        try:
            df = pd.read_csv(qp)
        except Exception:
            continue

        if "Q_router" in df.columns:
            q = df["Q_router"].to_numpy(dtype=float)
            if q.size > 0:
                st = int(0.8 * q.size)
                tail_means.append(float(np.mean(q[st:])))
                p95s.append(float(np.percentile(q, 95)))

        if "Q_diff" in df.columns:
            d = df["Q_diff"].to_numpy(dtype=float)
            if d.size > 0:
                gap_abs_means.append(float(np.mean(np.abs(d))))

    def _m(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float("nan")

    return _m(tail_means), _m(p95s), _m(gap_abs_means)


def score_combo(combo_dir: Path):
    files = sorted(combo_dir.glob("summary_lam_*.json"))
    if not files:
        return None

    regrets = []
    for f in files:
        d = _load_json(f)
        regrets.append(_safe_float(d.get("avg_regret")))

    regrets = [x for x in regrets if x == x]
    if not regrets:
        return None

    mean_reg = float(sum(regrets) / len(regrets))
    q_tail_mean, q_p95, q_gap_abs_mean = _read_q_metrics(combo_dir)

    any_one = _load_json(files[0])
    return {
        "combo_dir": str(combo_dir),
        "mean_avg_regret": float(mean_reg),
        "n_lams_found": int(len(files)),

        "Q_tail_mean": float(q_tail_mean),
        "Q_p95": float(q_p95),
        "Q_gap_abs_mean": float(q_gap_abs_mean),

        "target_explore_rate": _safe_float(any_one.get("target_explore_rate", float("nan"))),
        "alpha_coef": _safe_float(any_one.get("alpha_coef", float("nan"))),
        "c1": _safe_float(any_one.get("c1", float("nan"))),
        "mean_explore_rate": _safe_float(any_one.get("mean_explore_rate", float("nan"))),
        "cqb_tau": int(any_one.get("cqb_tau", -1)) if str(any_one.get("cqb_tau", "")).lstrip("-").isdigit() else -1,
    }


def main(alg_root: str, topk: int = 3):
    root = Path(alg_root).resolve()
    scores = []

    for er_dir in sorted(root.glob("er*")):
        if not er_dir.is_dir():
            continue
        for alpha_dir in sorted(er_dir.glob("alpha*")):
            if not alpha_dir.is_dir():
                continue
            s = score_combo(alpha_dir)
            if s is not None:
                scores.append(s)

    if not scores:
        print("no valid combos found under:", str(root))
        return

    def rank_key(s):
        return (
            s["mean_avg_regret"],
            0 if not _is_nan(s["Q_tail_mean"]) else 1, s["Q_tail_mean"],
            0 if not _is_nan(s["Q_p95"]) else 1, s["Q_p95"],
        )

    scores.sort(key=rank_key)
    top = scores[: max(1, int(topk))]

    out = {
        "alg_root": str(root),
        "ranking_rule": "sort by mean_avg_regret, then Q_tail_mean, then Q_p95 (all smaller is better)",
        "topk": int(topk),
        "top": top,
    }

    out_path = root / "best_top3.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", str(out_path))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], topk=3)
