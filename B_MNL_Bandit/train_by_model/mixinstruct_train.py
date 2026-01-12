#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from queue_config import QueueConfig
from train import add_common_args, run_pipeline
# -------------------------
# Cost proxy helpers
# -------------------------
def _parse_size_b(
    model_name: str,
    default: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> float:
    """
    모델명에서 13b, 7B, 8x7B 같은 패턴을 파싱해서 'B(십억 파라미터)' 규모 proxy를 만든다.

    - Dense: 7B -> 7
    - MoE: 8x7B
        * use_moe_effective=True  -> (active experts)*7 (기본 2*7=14)
        * use_moe_effective=False -> 8*7=56
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


def _approx_tokens(text: str, mode: str = "chars4") -> int:
    """
    토큰 수 proxy.
    - chars4: 언어 비독립적 근사(글자수/4). 한국어/코드에서 split()보다 안정적이다
    - whitespace: 공백 split 기반
    """
    s = str(text or "")
    if mode == "whitespace":
        return max(1, len(s.split()))
    # chars4
    return max(1, (len(s) + 3) // 4)


# -------------------------
# MixInstruct schema helpers
# -------------------------
def _build_prompt(ex: Dict[str, Any]) -> str:
    """
    mix-instruct 기본 스키마: instruction + input 조합
    일부 변형 스키마 대비 fallback 포함
    """
    if "prompt" in ex and ex["prompt"] is not None:
        return str(ex["prompt"])

    instr = str(ex.get("instruction", "") or "")
    inp = str(ex.get("input", "") or "")

    if instr and inp.strip():
        return instr + "\n\n" + inp
    if instr:
        return instr
    if inp:
        return inp

    return str(ex.get("text", "") or "")


def _infer_common_models(ds, n_scan: int = 2000) -> List[str]:
    """
    앞쪽 n_scan개를 훑어서 candidates에 항상 존재하는 model의 교집합을 추정한다.
    models를 안 주고 completeness를 강제하면 0 rows가 자주 나와서 이 옵션이 유용하다.
    """
    common: Optional[set] = None
    n = min(int(n_scan), len(ds))

    for i in range(n):
        ex = ds[i]
        candidates = ex.get("candidates", []) or []
        ms = {str(c.get("model")) for c in candidates if c.get("model") is not None}
        if not ms:
            continue
        common = ms if common is None else (common & ms)

    if not common:
        raise ValueError("[mix-instruct] failed to infer common models. candidates/model 필드를 확인해라")

    return sorted(common)


# -------------------------
# Loader
# -------------------------
def load_mixinstruct_hf(
    dataset_id: str,
    split: str,
    metric: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
    infer_common: bool = False,
    infer_common_n: int = 2000,
    max_rows: Optional[int] = None,
    # compute proxy knobs
    rho_out: float = 1.0,
    denom_C: float = 1e6,
    token_mode: str = "chars4",
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    목표: 한 프롬프트(row)에 대해 '모든 모델'의 perf(+cost)가 존재하는 wide table로 변환한다.

    perf:
      - perf = candidates[i]["scores"][metric] 그대로 사용한다
      - 기본 metric은 bertscore로 둔다

    cost (compute proxy):
      - cost = ((n_in(prompt) + rho_out * n_out(output)) * B_eff(model)) / denom_C
      - mix-instruct에는 실제 $ cost가 없어서 proxy로 통일한다
    """
    from datasets import load_dataset  # pip install datasets

    ds = load_dataset(dataset_id, split=split)

    # 모델 리스트 결정
    if models_fixed is not None and len(models_fixed) > 0:
        models = list(models_fixed)
    elif infer_common:
        models = _infer_common_models(ds, n_scan=int(infer_common_n))
        print(f"[mix-instruct] inferred common models: K={len(models)}")
    else:
        first = ds[0]
        candidates0 = first.get("candidates", []) or []
        models = sorted({str(c.get("model")) for c in candidates0 if c.get("model") is not None})
        if len(models) == 0:
            raise ValueError("[mix-instruct] cannot infer models from first sample")

    model_set = set(models)

    # 모델별 B proxy 캐시
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
    kept = 0
    seen = 0

    for idx, ex in enumerate(ds):
        seen += 1
        if max_rows is not None and kept >= int(max_rows):
            break

        candidates = ex.get("candidates", []) or []
        if not candidates:
            continue

        prompt = _build_prompt(ex)

        # input token은 모델마다 동일하니 row당 1번만 계산한다
        in_tok = _approx_tokens(prompt, mode=token_mode) if use_cost else 0

        perf: Dict[str, float] = {}
        cost: Dict[str, float] = {}

        for c in candidates:
            m = c.get("model", None)
            if m is None:
                continue
            m = str(m)
            if m not in model_set:
                continue

            scores = c.get("scores", {}) or {}
            if metric not in scores:
                continue

            # perf
            perf[m] = float(scores[metric])

            # cost (compute proxy)
            if use_cost:
                out_text = c.get("text", "")
                out_tok = _approx_tokens(out_text, mode=token_mode)
                B = float(model_B[m])
                raw_cost = (float(in_tok) + float(rho_out) * float(out_tok)) * B
                cost[m] = raw_cost / float(denom_C)

        # completeness 강제: "한 쿼리에 모든 LLM 값이 있어야 한다"
        if len(perf) != len(models):
            continue
        if use_cost and len(cost) != len(models):
            continue

        row: Dict[str, Any] = {
            "sample_id": ex.get("id", idx),
            "prompt": prompt,
            "eval_name": "mix-instruct",
            "oracle_model_to_route_to": "",
        }
        for m in models:
            row[m] = perf[m]
            if use_cost:
                row[f"{m}|total_cost"] = cost[m]

        rows.append(row)
        kept += 1

    df = pd.DataFrame(rows).reset_index(drop=True)

    need = ["prompt"] + models + ([f"{m}|total_cost" for m in models] if use_cost else [])
    df = df.dropna(subset=need).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "[mix-instruct] 0 rows after enforcing completeness.\n"
            "해결 1) --infer_common_models 1 --infer_common_n 2000 같이 common model subset을 자동 추정한다\n"
            "해결 2) --models 로 항상 등장하는 모델 subset을 직접 지정한다\n"
            "해결 3) --max_rows로 작게 시작해서 스키마/metric 정상 여부부터 확인한다"
        )

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    print(
        f"[mix-instruct] loaded rows={len(df)} (scanned={seen}), "
        f"K_models={len(models)}, metric={metric}, "
        f"cost=compute_proxy(in+{rho_out}*out)*B/denom_C (token_mode={token_mode}, denom_C={denom_C})"
    )
    return df, models, cost_map


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train on mix-instruct (HF) using common train.py pipeline")

    ap.add_argument("--data", type=str, default="llm-blender/mix-instruct")
    ap.add_argument("--hf_split", type=str, default="train", help="train/validation/test")

    # bertscore를 기본으로 둔다
    ap.add_argument("--metric", type=str, default="bertscore", help="scores key in candidates[*].scores")

    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="고정 모델 목록. 안 주면 첫 샘플에서 infer한다",
    )
    ap.add_argument(
        "--infer_common_models",
        type=int,
        default=0,
        help="1이면 앞쪽 n개를 훑어 '항상 존재하는 모델 교집합'으로 models를 자동 설정한다",
    )
    ap.add_argument("--infer_common_n", type=int, default=2000)
    ap.add_argument("--max_rows", type=int, default=None, help="로딩 row 상한(디버그용)")

    # compute proxy knobs
    ap.add_argument("--rho_out", type=float, default=1.0, help="output token weight")
    ap.add_argument("--denom_C", type=float, default=1e6, help="cost scaling denominator")
    ap.add_argument("--token_mode", type=str, default="chars4", choices=["chars4", "whitespace"])
    ap.add_argument("--default_B", type=float, default=7.0)
    ap.add_argument("--use_moe_effective", type=int, default=1, help="1이면 MoE는 active-expert 기준으로 B를 잡는다")
    ap.add_argument("--moe_active_experts", type=int, default=2)

    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()

    df, models, cost_map = load_mixinstruct_hf(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        metric=str(args.metric),
        use_cost=bool(config.use_cost),
        models_fixed=list(args.models) if args.models is not None else None,
        infer_common=bool(int(args.infer_common_models)),
        infer_common_n=int(args.infer_common_n),
        max_rows=args.max_rows,
        rho_out=float(args.rho_out),
        denom_C=float(args.denom_C),
        token_mode=str(args.token_mode),
        default_B=float(args.default_B),
        use_moe_effective=bool(int(args.use_moe_effective)),
        moe_active_experts=int(args.moe_active_experts),
    )

    run_pipeline(df, models, cost_map, args, config)


if __name__ == "__main__":
    main()
