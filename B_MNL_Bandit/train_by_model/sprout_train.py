#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from queue_config import QueueConfig
from train import add_common_args, run_pipeline

def _parse_size_b(
    model_name: str,
    default: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> float:
    """
    모델명에서 13b, 7B, 8x7B 같은 패턴을 파싱해서 B 단위 규모를 만든다.
    MoE(8x7B)는 total(56B) 대신 active expert 기반(예: 2*7=14B)으로 둘 수도 있다.
    """
    import re

    s = str(model_name).lower()

    # MoE: 8x7B, 8×7B
    m = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*b", s)
    if m:
        n_exp = float(m.group(1))
        exp_b = float(m.group(2))
        if use_moe_effective:
            return float(moe_active_experts) * exp_b
        return n_exp * exp_b

    # Dense: 7B, 13b
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1))

    return float(default)


def load_sprout_hf(
    dataset_id: str,
    split: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    # compute proxy knobs
    rho_out: float = 1.0,
    denom_C: float = 1e6,
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    SPROUT wide table 변환.

    perf: d["score"]
    cost (compute proxy): ((in_tok + rho*out_tok) * B(model)) / denom_C
    """
    from datasets import load_dataset  # pip install datasets

    ds = load_dataset(dataset_id, split=split)
    meta_cols = {"key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"}
    model_cols = [c for c in ds.column_names if c not in meta_cols]

    models = list(models_fixed) if models_fixed is not None else sorted(model_cols)

    # 모델별 B proxy를 미리 만든다
    model_B = {
        m: _parse_size_b(
            m,
            default=default_B,
            use_moe_effective=use_moe_effective,
            moe_active_experts=moe_active_experts,
        )
        for m in models
    }

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        if max_rows is not None and len(rows) >= int(max_rows):
            break

        row: Dict[str, Any] = {
            "sample_id": ex.get("key", i),
            "prompt": str(ex.get("prompt", "")),
            "eval_name": str(ex.get("dataset", "sprout")),
        }

        ok = True
        for m in models:
            d = ex.get(m, None)
            if not isinstance(d, dict):
                ok = False
                break

            score = d.get("score", None)
            if score is None:
                ok = False
                break
            row[m] = float(score)

            if use_cost:
                in_tok = d.get("num_input_tokens", None)
                out_tok = d.get("num_output_tokens", None)
                if in_tok is None or out_tok is None:
                    ok = False
                    break

                B = float(model_B[m])
                raw = (float(in_tok) + float(rho_out) * float(out_tok)) * B
                row[f"{m}|total_cost"] = raw / float(denom_C)

        if ok:
            rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)
    need = ["prompt"] + models + ([f"{m}|total_cost" for m in models] if use_cost else [])
    df = df.dropna(subset=need).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("[SPROUT] 0 rows after enforcing completeness")

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    return df, models, cost_map


def parse_args():
    ap = argparse.ArgumentParser(description="Train on SPROUT(HF) with common pipeline (compute-proxy cost)")
    ap.add_argument("--data", type=str, default="CARROT-LLM-Routing/SPROUT-o3mini")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--models", type=str, nargs="+", default=None)
    ap.add_argument("--max_rows", type=int, default=None)

    # compute proxy knobs
    ap.add_argument("--rho_out", type=float, default=1.0)
    ap.add_argument("--denom_C", type=float, default=1e6)
    ap.add_argument("--default_B", type=float, default=7.0)
    ap.add_argument("--use_moe_effective", type=int, default=1)
    ap.add_argument("--moe_active_experts", type=int, default=2)

    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()

    df, models, cost_map = load_sprout_hf(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        use_cost=bool(config.use_cost),
        models_fixed=list(args.models) if args.models is not None else None,
        max_rows=args.max_rows,
        rho_out=float(args.rho_out),
        denom_C=float(args.denom_C),
        default_B=float(args.default_B),
        use_moe_effective=bool(int(args.use_moe_effective)),
        moe_active_experts=int(args.moe_active_experts),
    )
    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()
