import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def infer_models_and_cost_map(df: pd.DataFrame):
    base = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to", "orig_row"}
    models = [c for c in df.columns if ("|" not in c) and (c not in base)]
    cost_map = {c.split("|")[0]: c for c in df.columns if c.endswith("|total_cost")}
    return sorted(models), cost_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--use_cost", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "prompt" not in df.columns:
        raise ValueError("missing prompt")

    if "orig_row" not in df.columns:
        df["orig_row"] = np.arange(len(df), dtype=np.int64)

    models, cost_map = infer_models_and_cost_map(df)

    acc = df[models].astype(np.float32).to_numpy()
    np.save(cache_dir / "acc.npy", acc)

    if bool(args.use_cost):
        cost = np.zeros_like(acc, dtype=np.float32)
        for j, m in enumerate(models):
            col = cost_map.get(m, None)
            if col is not None and col in df.columns:
                cost[:, j] = df[col].astype(np.float32).to_numpy()
        np.save(cache_dir / "cost.npy", cost)

    embedder = SentenceTransformer(str(args.embedder_model), device=str(args.device))
    X = embedder.encode(
        df["prompt"].astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(cache_dir / "X.npy", X)

    np.save(cache_dir / "orig_row.npy", df["orig_row"].to_numpy(dtype=np.int64))

    if "sample_id" in df.columns:
        np.save(cache_dir / "sample_id.npy", df["sample_id"].to_numpy())
    else:
        np.save(cache_dir / "sample_id.npy", np.arange(len(df), dtype=np.int64))

    (cache_dir / "models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "data": str(data_path),
                "N": int(len(df)),
                "K": int(len(models)),
                "embedder_model": str(args.embedder_model),
                "use_cost": bool(args.use_cost),
                "columns": list(df.columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
