import json
from pathlib import Path


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def score_combo(combo_dir: Path):
    files = sorted(combo_dir.glob("summary_lam_*.json"))
    if not files:
        return None

    regrets = []
    qgaps = []
    for f in files:
        d = _load_json(f)
        regrets.append(_safe_float(d.get("avg_regret")))
        qgaps.append(_safe_float(d.get("final_Q_gap")))

    regrets = [x for x in regrets if x == x]
    qgaps = [x for x in qgaps if x == x]

    if not regrets:
        return None

    mean_reg = sum(regrets) / len(regrets)
    mean_qgap = sum(qgaps) / len(qgaps) if qgaps else float("nan")

    any_one = _load_json(files[0])
    out = {
        "combo_dir": str(combo_dir),
        "mean_avg_regret": float(mean_reg),
        "mean_final_Q_gap": float(mean_qgap),
        "n_lams_found": int(len(files)),
        "target_explore_rate": _safe_float(any_one.get("target_explore_rate", float("nan"))),
        "alpha_coef": _safe_float(any_one.get("alpha_coef", float("nan"))),
        "c1": _safe_float(any_one.get("c1", float("nan"))),
        "mean_explore_rate": _safe_float(any_one.get("mean_explore_rate", float("nan"))),
        "cqb_tau": int(any_one.get("cqb_tau", -1)) if str(any_one.get("cqb_tau", "")).lstrip("-").isdigit() else -1,
    }
    return out


def main(alg_root: str):
    root = Path(alg_root).resolve()
    best = None

    for er_dir in sorted(root.glob("er*")):
        if not er_dir.is_dir():
            continue
        for alpha_dir in sorted(er_dir.glob("alpha*")):
            if not alpha_dir.is_dir():
                continue
            s = score_combo(alpha_dir)
            if s is None:
                continue
            if best is None:
                best = s
            else:
                if s["mean_avg_regret"] < best["mean_avg_regret"]:
                    best = s
                elif s["mean_avg_regret"] == best["mean_avg_regret"]:
                    if s["mean_final_Q_gap"] == s["mean_final_Q_gap"] and best["mean_final_Q_gap"] == best["mean_final_Q_gap"]:
                        if s["mean_final_Q_gap"] < best["mean_final_Q_gap"]:
                            best = s

    if best is None:
        print("no valid combos found under:", str(root))
        return

    out_path = root / "best_summary.json"
    out_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", str(out_path))
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
