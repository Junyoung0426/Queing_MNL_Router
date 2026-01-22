import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
current_dir = os.path.dirname(os.path.abspath(__file__))          # .../train_by_model/embedllm
parent_dir = os.path.dirname(current_dir)                         # .../train_by_model
root_dir = os.path.dirname(parent_dir)                            # .../B_MNL_Bandit_disjoint

for p in (current_dir, root_dir):
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, current_dir)  
sys.path.insert(1, root_dir)      

sys.modules.pop("queue_config", None)
sys.modules.pop("train", None)


from train import add_common_args, run_pipeline, set_full_determinism
from queue_config import QueueConfig

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
    """
    import re

    s = str(model_name).lower()

    # MoE: 8x7B, 8×7B
    m = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*b", s)
    if m:
        n_exp = float(m.group(1))
        exp_b = float(m.group(2))
        return float(moe_active_experts) * exp_b if use_moe_effective else (n_exp * exp_b)

    # Dense: 7B, 13b
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)
    if m:
        return float(m.group(1))

    return float(default)


def _approx_tokens(text: str, mode: str = "chars4") -> int:
    """
    토큰 수 proxy.
    - chars4: 글자수/4 근사
    - whitespace: 공백 split 기반
    """
    s = str(text or "")
    if mode == "whitespace":
        return max(1, len(s.split()))
    return max(1, (len(s) + 3) // 4)


# -------------------------
# HF split loader (split 없으면 fallback)
# -------------------------
def _load_hf_split_auto(dataset_id: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_id, split=split)
    except Exception as e:
        ds_dict = load_dataset(dataset_id)
        if hasattr(ds_dict, "keys"):
            keys = list(ds_dict.keys())
            if len(keys) == 0:
                raise ValueError(f"[mix-instruct] dataset has no splits: {dataset_id}") from e
            chosen = keys[0]
            print(f"[mix-instruct] split '{split}' not found. fallback to '{chosen}' (available={keys})")
            return ds_dict[chosen]
        return ds_dict


# -------------------------
# MixInstruct schema helpers
# -------------------------
def _build_prompt(ex: Dict[str, Any]) -> str:
    """
    mix-instruct 기본 스키마: instruction + input 조합
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


def _infer_models_from_first_sample(ds) -> List[str]:
    first = ds[0]
    candidates0 = first.get("candidates", []) or []
    models = sorted({str(c.get("model")) for c in candidates0 if c.get("model") is not None})
    if len(models) == 0:
        raise ValueError("[mix-instruct] cannot infer models from first sample (candidates/model empty)")
    return models


# -------------------------
# Loader
# -------------------------
def load_mixinstruct_hf(
    dataset_id: str,
    split: str,
    metric: str,
    use_cost: bool,
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
    wide table 변환.
    - 모델 리스트는 첫 샘플에서 자동 추론하고 고정한다
    - completeness 강제: 한 row에 models 전부의 perf(+cost)가 있어야 남긴다
    """
    ds = _load_hf_split_auto(dataset_id, split=split)

    models = _infer_models_from_first_sample(ds)
    model_set = set(models)

    print(f"[mix-instruct] inferred models from first sample: K={len(models)}")
    # print(models)  # 필요하면 주석 해제

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

            # cost proxy
            if use_cost:
                out_text = c.get("text", "") or ""
                out_tok = _approx_tokens(out_text, mode=token_mode)
                B = float(model_B[m])
                raw_cost = (float(in_tok) + float(rho_out) * float(out_tok)) * B
                cost[m] = raw_cost / float(denom_C)

        # completeness 강제
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
            "첫 샘플에서 잡힌 모델들이 다른 샘플에 자주 빠지면 row가 0이 될 수 있다.\n"
            "그때는 (1) 교집합 스캔을 다시 넣거나, (2) 결측 허용 로직으로 파이프라인을 바꿔야 한다."
        )

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    print(
        f"[mix-instruct] loaded rows={len(df)} (scanned={seen}), "
        f"K_models={len(models)}, metric={metric}, "
        f"use_cost={use_cost}"
    )
    return df, models, cost_map


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train on mix-instruct (HF) using common train.py pipeline")

    ap.add_argument("--data", type=str, default="llm-blender/mix-instruct")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--metric", type=str, default="bertscore", help="scores key in candidates[*].scores")
    ap.add_argument("--max_rows", type=int, default=None, help="로딩 row 상한(디버그용)")

    # compute proxy knobs
    ap.add_argument("--rho_out", type=float, default=1.0, help="output token weight")
    ap.add_argument("--denom_C", type=float, default=1e6, help="cost scaling denominator")
    ap.add_argument("--token_mode", type=str, default="chars4", choices=["chars4", "whitespace"])
    ap.add_argument("--default_B", type=float, default=7.0)
    ap.add_argument("--use_moe_effective", type=int, default=1)
    ap.add_argument("--moe_active_experts", type=int, default=2)

    add_common_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    config = QueueConfig()
    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "embedder_model", None) is not None:
        config.embedder_model = str(args.embedder_model)

    set_full_determinism(int(config.seed))

    df, models, cost_map = load_mixinstruct_hf(
        dataset_id=str(args.data),
        split=str(args.hf_split),
        metric=str(args.metric),
        use_cost=bool(config.use_cost),
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
