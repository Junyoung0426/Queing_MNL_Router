# Base line/_common/baseline_loaders.py
from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd


# ============================================================
# RouterBench(pkl)
# ============================================================
def infer_models_and_cost_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "oracle_model"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return models, cost_map


def load_routerbench_pkl(
    path: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_pickle(path)
    if "prompt" not in df.columns:
        raise ValueError("df에 'prompt' 컬럼이 필요하다")

    models_all, cost_map_all = infer_models_and_cost_map(df)

    if models_fixed is None:
        models = models_all
    else:
        models = list(models_fixed)
        missing = [m for m in models if m not in models_all]
        if missing:
            raise ValueError(f"models_fixed 중 df에 없는 모델이 있다: {missing}")

    cost_map = {m: cost_map_all[m] for m in models} if use_cost else {}

    keep = []
    for c in ["sample_id", "prompt", "eval_name"]:
        if c in df.columns:
            keep.append(c)

    keep += models
    if use_cost:
        keep += [cost_map[m] for m in models]

    seen = set()
    keep = [c for c in keep if not (c in seen or seen.add(c))]
    return df[keep].copy(), models, cost_map


# ============================================================
# Cost proxy helpers (MixInstruct / SPROUT / EmbedLLM 공용)
# ============================================================
def _parse_size_b(
    model_name: str,
    default: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> float:
    """
    모델명에서 13b, 7B, 8x7B 같은 패턴을 파싱해 B 규모 proxy를 만든다.
    MoE는 effective(active experts)로 둘 수도 있다.
    """
    import re

    s = str(model_name).lower()

    m = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*b", s)  # 8x7B
    if m:
        n_exp = float(m.group(1))
        exp_b = float(m.group(2))
        return float(moe_active_experts) * exp_b if use_moe_effective else (n_exp * exp_b)

    m = re.search(r"(\d+(?:\.\d+)?)\s*b", s)  # 7B, 13b
    if m:
        return float(m.group(1))

    return float(default)


def _approx_tokens(text: str, mode: str = "chars4") -> int:
    s = str(text or "")
    if mode == "whitespace":
        return max(1, len(s.split()))
    return max(1, (len(s) + 3) // 4)


# ============================================================
# MixInstruct (HF)
# ============================================================
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


def _build_prompt(ex: Dict[str, Any]) -> str:
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
        raise ValueError("[mix-instruct] cannot infer models from first sample")
    return models


def load_mixinstruct_hf(
    dataset_id: str,
    split: str,
    metric: str,
    use_cost: bool,
    max_rows: Optional[int] = None,
    rho_out: float = 1.0,
    denom_C: float = 1e6,
    token_mode: str = "chars4",
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    ds = _load_hf_split_auto(dataset_id, split=split)

    models = _infer_models_from_first_sample(ds)
    model_set = set(models)
    print(f"[mix-instruct] inferred models: K={len(models)}")

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

            perf[m] = float(scores[metric])

            if use_cost:
                out_text = c.get("text", "") or ""
                out_tok = _approx_tokens(out_text, mode=token_mode)
                B = float(model_B[m])
                raw_cost = (float(in_tok) + float(rho_out) * float(out_tok)) * B
                cost[m] = raw_cost / float(denom_C)

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
        raise ValueError("[mix-instruct] 0 rows after completeness enforcement")

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    print(f"[mix-instruct] loaded rows={len(df)} (scanned={seen}), use_cost={use_cost}, metric={metric}")
    return df, models, cost_map


# ============================================================
# SPROUT (HF)
# ============================================================
def load_sprout_hf(
    dataset_id: str,
    split: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    rho_out: float = 1.0,
    denom_C: float = 1e6,
    default_B: float = 7.0,
    use_moe_effective: bool = True,
    moe_active_experts: int = 2,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)

    meta_cols = {"key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"}
    model_cols = [c for c in ds.column_names if c not in meta_cols]
    models = list(models_fixed) if models_fixed is not None else sorted(model_cols)

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
        raise ValueError("[SPROUT] 0 rows after completeness enforcement")

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    print(f"[SPROUT] loaded rows={len(df)} K={len(models)} use_cost={use_cost}")
    return df, models, cost_map


# ============================================================
# EmbedLLM (HF csv pivot)
# ============================================================
def load_embedllm_hf_csv_pivot(
    dataset_id: str,
    split: str,
    use_cost: bool,
    max_prompts: int,
    seed: int = 0,
    chunksize: int = 200_000,
    default_B: float = 7.0,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    from huggingface_hub import hf_hub_download

    split2file = {"train": "train.csv", "validation": "val.csv", "val": "val.csv", "test": "test.csv"}
    filename = split2file.get(split, split)
    csv_path = hf_hub_download(repo_id=dataset_id, filename=filename, repo_type="dataset")

    model_order_path = hf_hub_download(repo_id=dataset_id, filename="model_order.csv", repo_type="dataset")
    model_df = pd.read_csv(model_order_path)
    if "model_name" not in model_df.columns:
        raise ValueError("[EmbedLLM] model_order.csv missing 'model_name'")

    models = model_df["model_name"].astype(str).tolist()
    K = len(models)
    model_set = set(models)

    model_cost_proxy = {m: _parse_size_b(m, default=default_B, use_moe_effective=False) for m in models}

    rng = np.random.RandomState(int(seed))
    store: Dict[int, Dict[str, Any]] = {}
    completed: List[int] = []

    reader = pd.read_csv(csv_path, chunksize=int(chunksize))
    for chunk in reader:
        req = {"prompt_id", "model_name", "label", "prompt"}
        if not req.issubset(set(chunk.columns)):
            raise ValueError(f"[EmbedLLM] required columns missing in {filename}")

        for r in chunk.itertuples(index=False):
            if len(completed) >= int(max_prompts):
                break

            pid = int(getattr(r, "prompt_id"))
            mname = str(getattr(r, "model_name"))
            if mname not in model_set:
                continue

            rec = store.get(pid)
            if rec is None:
                rec = {"prompt": str(getattr(r, "prompt")), "labels": {}, "cnt": 0}
                store[pid] = rec

            labels = rec["labels"]
            if mname not in labels:
                labels[mname] = float(getattr(r, "label"))
                rec["cnt"] += 1
                if rec["cnt"] == K:
                    completed.append(pid)

        if len(completed) >= int(max_prompts):
            break

    rows: List[Dict[str, Any]] = []
    for pid in completed:
        rec = store.get(pid)
        if rec is None or rec["cnt"] != K:
            continue
        labels = rec["labels"]
        if any(m not in labels for m in models):
            continue

        row: Dict[str, Any] = {"sample_id": pid, "prompt": rec["prompt"], "eval_name": "embedllm"}
        for m in models:
            row[m] = float(labels[m])
            if use_cost:
                row[f"{m}|total_cost"] = float(model_cost_proxy[m])
        rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("[EmbedLLM] 0 complete prompts collected. max_prompts를 늘리거나 split을 바꿔야 한다")

    cost_map = {m: f"{m}|total_cost" for m in models} if use_cost else {}
    print(f"[EmbedLLM] collected prompts={len(df)} K={len(models)} use_cost={use_cost}")
    return df, models, cost_map
