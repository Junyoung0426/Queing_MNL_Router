#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Path robustness: allow running from anywhere
# - mixinstruct_train.py / sprout_train.py / routerbench_train.py are in same folder (train/)
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from mixinstruct_train import load_mixinstruct_hf
from sprout_train import load_sprout_hf
from routerbench_train import load_routerbench_pkl


# -------------------------
# Param size (B unit) from model name
# -------------------------
def parse_param_b(
    model_name: str,
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Tuple[float, bool, str]:
    """
    return: (param_b, param_in_name, source)
      - param_b: B 단위 파라미터 규모 (7B -> 7)
      - param_in_name: 모델명에 B 정보가 실제로 있었는지
      - source: parsed_dense / parsed_moe_total / parsed_moe_effective / default
    """
    s = str(model_name).lower()

    # MoE: 8x7B, 8×7B
    m = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*b", s)
    if m:
        n_exp = float(m.group(1))
        exp_b = float(m.group(2))
        if use_moe_effective:
            return float(moe_active_experts) * exp_b, True, "parsed_moe_effective"
        return n_exp * exp_b, True, "parsed_moe_total"

    # Dense: 7B, 13b
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1)), True, "parsed_dense"

    return float(default_B), False, "default"


# -------------------------
# Summary core
# -------------------------
def _safe_mean(x: pd.Series) -> Optional[float]:
    if x is None or len(x) == 0:
        return None
    v = pd.to_numeric(x, errors="coerce")
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    return float(v.mean())


def summarize_df(
    df: pd.DataFrame,
    models: List[str],
    cost_map: Dict[str, str],
    dataset_name: str,
    metric_name: str,
    # param parse knobs
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Dict[str, Any]:
    n = int(len(df))
    K = int(len(models))

    perf_mat = df[models].astype(float).to_numpy()  # (n, K)

    # tie-included best/worst masks
    row_max = np.max(perf_mat, axis=1, keepdims=True)
    row_min = np.min(perf_mat, axis=1, keepdims=True)
    best_mask = (perf_mat == row_max)
    worst_mask = (perf_mat == row_min)

    # 공동 1등/꼴등 포함 횟수
    best_count = best_mask.sum(axis=0).astype(int).tolist()
    worst_count = worst_mask.sum(axis=0).astype(int).tolist()

    # tie diagnostics
    best_ties_per_row = best_mask.sum(axis=1)
    worst_ties_per_row = worst_mask.sum(axis=1)
    best_tie_rate = float(np.mean(best_ties_per_row > 1)) if n > 0 else 0.0
    worst_tie_rate = float(np.mean(worst_ties_per_row > 1)) if n > 0 else 0.0
    avg_best_ties = float(np.mean(best_ties_per_row)) if n > 0 else 0.0
    avg_worst_ties = float(np.mean(worst_ties_per_row)) if n > 0 else 0.0

    per_model: Dict[str, Any] = {}
    for j, m in enumerate(models):
        avg_perf = float(np.mean(perf_mat[:, j])) if n > 0 else None

        cost_col = cost_map.get(m)
        cost_available = (cost_col is not None) and (cost_col in df.columns)
        avg_cost = _safe_mean(df[cost_col]) if cost_available else None

        param_b, param_in_name, param_src = parse_param_b(
            m,
            default_B=default_B,
            use_moe_effective=use_moe_effective,
            moe_active_experts=moe_active_experts,
        )

        per_model[m] = {
            "avg_perf": avg_perf,
            "best_count": int(best_count[j]),   # 공동 1등 포함
            "worst_count": int(worst_count[j]), # 공동 꼴등 포함
            "avg_cost": avg_cost,
            "cost_available": bool(cost_available),
            "param_b": float(param_b),          # 7B -> 7
            "param_in_name": bool(param_in_name),
            "param_source": str(param_src),
        }

    out: Dict[str, Any] = {
        "dataset": dataset_name,
        "metric": metric_name,
        "n_rows": n,
        "n_models": K,
        "tie_stats": {
            "best_tie_rate": best_tie_rate,
            "worst_tie_rate": worst_tie_rate,
            "avg_best_ties_per_row": avg_best_ties,
            "avg_worst_ties_per_row": avg_worst_ties,
        },
        "models": per_model,
    }
    return out


def now_seoul_iso() -> str:
    tz = timezone(timedelta(hours=9))
    return datetime.now(tz=tz).isoformat(timespec="seconds")


def _ensure_dir(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Summarize per-model stats for mixinstruct/routerbench/sprout -> JSON")

    # which datasets
    ap.add_argument("--do_mixinstruct", type=int, default=1)
    ap.add_argument("--do_routerbench", type=int, default=1)
    ap.add_argument("--do_sprout", type=int, default=1)

    # routerbench
    ap.add_argument("--routerbench_pkl", type=str, default=None)

    # mixinstruct HF
    ap.add_argument("--mixinstruct_id", type=str, default="llm-blender/mix-instruct")
    ap.add_argument("--mixinstruct_split", type=str, default="train")
    ap.add_argument("--mixinstruct_metric", type=str, default="bertscore")
    ap.add_argument("--mixinstruct_infer_common_models", type=int, default=1)
    ap.add_argument("--mixinstruct_infer_common_n", type=int, default=2000)
    ap.add_argument("--mixinstruct_max_rows", type=int, default=None)

    # sprout HF
    ap.add_argument("--sprout_id", type=str, default="CARROT-LLM-Routing/SPROUT-o3mini")
    ap.add_argument("--sprout_split", type=str, default="train")
    ap.add_argument("--sprout_max_rows", type=int, default=None)

    # param parse knobs
    ap.add_argument("--default_B", type=float, default=7.0)
    ap.add_argument("--use_moe_effective", type=int, default=1)
    ap.add_argument("--moe_active_experts", type=int, default=2)

    # output dir
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(CUR_DIR, "model_perf_cost"),
        help="결과 저장 폴더",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = _ensure_dir(args.out_dir)

    report: Dict[str, Any] = {
        "generated_at": now_seoul_iso(),
        "datasets": {},
    }

    default_B = float(args.default_B)
    use_moe_effective = bool(int(args.use_moe_effective))
    moe_active_experts = int(args.moe_active_experts)

    # 1) MixInstruct
    if int(args.do_mixinstruct) == 1:
        df, models, cost_map = load_mixinstruct_hf(
            dataset_id=str(args.mixinstruct_id),
            split=str(args.mixinstruct_split),
            metric=str(args.mixinstruct_metric),
            use_cost=True,
            models_fixed=None,
            infer_common=bool(int(args.mixinstruct_infer_common_models)),
            infer_common_n=int(args.mixinstruct_infer_common_n),
            max_rows=args.mixinstruct_max_rows,
        )
        ds_obj = summarize_df(
            df=df,
            models=models,
            cost_map=cost_map,
            dataset_name="mix-instruct",
            metric_name=str(args.mixinstruct_metric),
            default_B=default_B,
            use_moe_effective=use_moe_effective,
            moe_active_experts=moe_active_experts,
        )
        report["datasets"]["mix-instruct"] = ds_obj
        _write_json(os.path.join(out_dir, "mix-instruct.json"), ds_obj)

    # 2) RouterBench
    if int(args.do_routerbench) == 1:
        if not args.routerbench_pkl:
            raise ValueError("--routerbench_pkl 필요하다 (do_routerbench=1)")
        df, models, cost_map = load_routerbench_pkl(str(args.routerbench_pkl), use_cost=True)
        ds_obj = summarize_df(
            df=df,
            models=models,
            cost_map=cost_map,
            dataset_name="routerbench",
            metric_name="(given in pkl)",
            default_B=default_B,
            use_moe_effective=use_moe_effective,
            moe_active_experts=moe_active_experts,
        )
        report["datasets"]["routerbench"] = ds_obj
        _write_json(os.path.join(out_dir, "routerbench.json"), ds_obj)

    # 3) SPROUT
    if int(args.do_sprout) == 1:
        df, models, cost_map = load_sprout_hf(
            dataset_id=str(args.sprout_id),
            split=str(args.sprout_split),
            use_cost=True,
            models_fixed=None,
            max_rows=args.sprout_max_rows,
        )
        ds_obj = summarize_df(
            df=df,
            models=models,
            cost_map=cost_map,
            dataset_name="sprout",
            metric_name="score",
            default_B=default_B,
            use_moe_effective=use_moe_effective,
            moe_active_experts=moe_active_experts,
        )
        report["datasets"]["sprout"] = ds_obj
        _write_json(os.path.join(out_dir, "sprout.json"), ds_obj)

    # 통합 JSON도 저장
    _write_json(os.path.join(out_dir, "all.json"), report)

    print(f"[ok] wrote JSONs into: {out_dir}")
    print(" - all.json")
    for k in report["datasets"].keys():
        print(f" - {k}.json")


if __name__ == "__main__":
    main()
