from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


def infer_models_and_cost_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    base_meta_cols = {
        "orig_row",
        "sample_id",
        "prompt",
        "eval_name",
        "oracle_model_to_route_to",
        "oracle_model",
        "dataset",
        "Unnamed: 0",
        "key",
        "golden_answer",
    }
    models = [c for c in df.columns if ("|" not in c) and (c not in base_meta_cols)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return sorted(models), cost_map


def load_standardized_csv(
    path: str,
    use_cost: bool,
    models_fixed: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_csv(path)

    if "orig_row" not in df.columns:
        df["orig_row"] = np.arange(len(df), dtype=np.int64)

    if "prompt" not in df.columns:
        raise ValueError(f"Missing prompt column. Columns: {df.columns.tolist()}")

    models_all, cost_map_all = infer_models_and_cost_map(df)

    if models_fixed is None:
        models = models_all
    else:
        models = list(models_fixed)
        missing = [m for m in models if m not in models_all]
        if missing:
            raise ValueError(f"Missing models in data: {missing}")

    if use_cost:
        missing_cost = [m for m in models if m not in cost_map_all]
        if missing_cost:
            raise ValueError(f"Missing cost columns: {missing_cost}")
        cost_map = {m: cost_map_all[m] for m in models}
    else:
        cost_map = {}

    keep_cols = []
    for c in ["orig_row", "sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "dataset"]:
        if c in df.columns:
            keep_cols.append(c)

    keep_cols += models

    if use_cost:
        keep_cols += [cost_map[m] for m in models if m in cost_map]

    seen = set()
    keep_unique = [c for c in keep_cols if not (c in seen or seen.add(c))]

    df_clean = df[keep_unique].copy()
    return df_clean, models, cost_map
