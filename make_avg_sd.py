import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EXP_RE = re.compile(r"^exp(\d+)$")
REG_RE = re.compile(r"^regret_history_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")
Q_RE = re.compile(r"^Qregret_history_lam_([0-9]+(?:\.[0-9]+)?)\.csv$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--out_dirname", type=str, default="avg")
    ap.add_argument("--lambdas", type=float, nargs="*", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--min_runs", type=int, default=1)
    ap.add_argument("--include", type=str, nargs="*", default=None)
    ap.add_argument("--exclude", type=str, nargs="*", default=["plots", "avg"])
    return ap.parse_args()


def _list_exp_dirs(root_dir: Path) -> List[Path]:
    items: List[Tuple[int, Path]] = []
    for p in root_dir.iterdir():
        if not p.is_dir():
            continue
        m = EXP_RE.match(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _collect_algs(exp_dirs: List[Path], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    excl = set(exclude or [])
    incl = set(include) if include else None
    algs = set()
    for e in exp_dirs:
        for p in e.iterdir():
            if not p.is_dir():
                continue
            if p.name in excl:
                continue
            if incl is not None and p.name not in incl:
                continue
            algs.add(p.name)
    return sorted(algs)


def _infer_lams(exp_dirs: List[Path], alg: str) -> List[float]:
    lams = set()
    for e in exp_dirs:
        a = e / alg
        if not a.exists():
            continue
        for f in a.glob("regret_history_lam_*.csv"):
            m = REG_RE.match(f.name)
            if m:
                try:
                    lams.add(float(m.group(1)))
                except Exception:
                    pass
        for f in a.glob("Qregret_history_lam_*.csv"):
            m = Q_RE.match(f.name)
            if m:
                try:
                    lams.add(float(m.group(1)))
                except Exception:
                    pass
    return sorted(lams)


def _read_regret(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "cum_regret" in df.columns:
        return df["cum_regret"].to_numpy(dtype=np.float64)
    return df.iloc[:, 0].to_numpy(dtype=np.float64)


def _read_qdiff(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "Q_diff" in df.columns:
        return df["Q_diff"].to_numpy(dtype=np.float64)
    return df.iloc[:, 0].to_numpy(dtype=np.float64)


def _stack_minlen(arrs: List[np.ndarray], max_steps: Optional[int]) -> Optional[np.ndarray]:
    if len(arrs) == 0:
        return None
    L = min(int(a.shape[0]) for a in arrs)
    if max_steps is not None:
        L = min(L, int(max_steps))
    if L <= 0:
        return None
    return np.stack([a[:L] for a in arrs], axis=0)


def _mean_sd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1) if X.shape[0] >= 2 else np.zeros_like(mu)
    return mu, sd


def main():
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(str(root_dir))

    exp_dirs = _list_exp_dirs(root_dir)
    if len(exp_dirs) == 0:
        raise RuntimeError(f"no exp dirs under: {root_dir}")

    out_root = (root_dir / str(args.out_dirname)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    algs = _collect_algs(exp_dirs, args.include, args.exclude)
    if len(algs) == 0:
        raise RuntimeError("no alg dirs found in exp folders")

    for alg in algs:
        out_alg = (out_root / alg).resolve()
        out_alg.mkdir(parents=True, exist_ok=True)

        if args.lambdas is None or len(args.lambdas) == 0:
            lams = _infer_lams(exp_dirs, alg)
        else:
            lams = sorted([float(x) for x in args.lambdas])

        for lam in lams:
            lam2 = f"{float(lam):.2f}"

            reg_list: List[np.ndarray] = []
            q_list: List[np.ndarray] = []
            used_reg: List[str] = []
            used_q: List[str] = []

            for e in exp_dirs:
                reg_path = e / alg / f"regret_history_lam_{lam2}.csv"
                q_path = e / alg / f"Qregret_history_lam_{lam2}.csv"

                if reg_path.exists():
                    try:
                        reg_list.append(_read_regret(reg_path))
                        used_reg.append(e.name)
                    except Exception:
                        pass

                if q_path.exists():
                    try:
                        q_list.append(_read_qdiff(q_path))
                        used_q.append(e.name)
                    except Exception:
                        pass

            if len(reg_list) >= int(args.min_runs):
                X = _stack_minlen(reg_list, args.max_steps)
                if X is not None:
                    mu, sd = _mean_sd(X)
                    t = np.arange(1, mu.shape[0] + 1, dtype=np.int64)
                    pd.DataFrame(
                        {"t": t, "cum_regret_mean": mu, "cum_regret_sd": sd, "n_runs": int(X.shape[0])}
                    ).to_csv(out_alg / f"regret_mean_sd_lam_{lam2}.csv", index=False)
                    (out_alg / f"regret_used_runs_lam_{lam2}.txt").write_text("\n".join(used_reg), encoding="utf-8")

            if len(q_list) >= int(args.min_runs):
                Xq = _stack_minlen(q_list, args.max_steps)
                if Xq is not None:
                    mu, sd = _mean_sd(Xq)
                    t = np.arange(1, mu.shape[0] + 1, dtype=np.int64)
                    pd.DataFrame(
                        {"t": t, "Q_diff_mean": mu, "Q_diff_sd": sd, "n_runs": int(Xq.shape[0])}
                    ).to_csv(out_alg / f"qgap_mean_sd_lam_{lam2}.csv", index=False)
                    (out_alg / f"qgap_used_runs_lam_{lam2}.txt").write_text("\n".join(used_q), encoding="utf-8")

    print("Saved avg to:", str(out_root))


if __name__ == "__main__":
    main()
