import sys
import argparse
import subprocess
from pathlib import Path
import importlib.util
import shlex


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _run(cmd_list, cwd: Path, dry_run: bool):
    print("CMD:", " ".join(map(str, cmd_list)))
    if dry_run:
        return
    subprocess.run(cmd_list, cwd=str(cwd), check=True)


def _has_seed_in_extra(extra_args_list):
    for tok in extra_args_list:
        if tok == "--seed":
            return True
        if tok.startswith("--seed="):
            return True
    return False


def _load_queue_config_instance(qc_path: Path):
    try:
        mod_name = f"_queue_config_{qc_path.stem}_{abs(hash(str(qc_path)))}"
        spec = importlib.util.spec_from_file_location(mod_name, str(qc_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "QueueConfig"):
            return None
        return mod.QueueConfig()
    except Exception:
        return None


def _get_cfg(BASE_DIR: Path, dataset: str):
    qc_path = (BASE_DIR / "train_by_model" / dataset / "queue_config.py").resolve()
    cfg = None
    if qc_path.exists():
        cfg = _load_queue_config_instance(qc_path)

    if cfg is None:
        fallback = (BASE_DIR / "train_by_model" / "routerbench" / "queue_config.py").resolve()
        if fallback.exists():
            cfg = _load_queue_config_instance(fallback)

    return cfg


def _get_base_seed(BASE_DIR: Path, dataset: str) -> int:
    cfg = _get_cfg(BASE_DIR, dataset)
    if cfg is None:
        return 42
    if not hasattr(cfg, "seed"):
        return 42
    return int(getattr(cfg, "seed"))


def _get_assort_k(BASE_DIR: Path, dataset: str) -> int:
    cfg = _get_cfg(BASE_DIR, dataset)
    if cfg is None:
        return 1
    if not hasattr(cfg, "assort_K"):
        return 1
    try:
        return int(getattr(cfg, "assort_K"))
    except Exception:
        return 1


def _get_arrival_rate(BASE_DIR: Path, dataset: str):
    cfg = _get_cfg(BASE_DIR, dataset)
    if cfg is None:
        return None
    if not hasattr(cfg, "arrival_rate"):
        return None
    try:
        return float(getattr(cfg, "arrival_rate"))
    except Exception:
        return None


def _format_ar_tag(x):
    if x is None:
        return "arNA"
    try:
        s = f"{float(x):g}"
        return f"ar{s}"
    except Exception:
        return "arNA"


def main():
    BASE_DIR = Path(__file__).resolve().parent

    DEFAULT_DATA_DIR = Path("/home/sjy990426/Desktop/LLM_Router/Queing_MNL_Router/Data")

    parser = argparse.ArgumentParser(description="Run training sequentially (routerbench/sprout/embedllm).")
    parser.add_argument("save_path", type=str, help="Root output directory (e.g., ./result)")
    parser.add_argument("--run", type=int, required=True, help="Run index (e.g., 1,2,3...)")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["routerbench", "sprout", "embedllm"],
        choices=["routerbench", "sprout", "embedllm"],
    )

    parser.add_argument("--job_pool_size", type=int, required=True, help="Number of samples to draw for training")
    parser.add_argument("--lam_list", type=float, nargs="+", required=True, help="Lambda list (space-separated)")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--routerbench_csv", type=str, default=str(DEFAULT_DATA_DIR / "routerbench_dataset.csv"))
    parser.add_argument("--sprout_csv", type=str, default=str(DEFAULT_DATA_DIR / "sprout_dataset.csv"))
    parser.add_argument("--embedllm_csv", type=str, default=str(DEFAULT_DATA_DIR / "embedllm_dataset.csv"))

    parser.add_argument(
        "--algs",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to run by output_name (e.g., AQCB, CQB_eps, Q_UCB). If omitted, run all.",
    )

    parser.add_argument("--extra_args", type=str, default="", help="Extra args forwarded to every script")
    args = parser.parse_args()

    if args.run < 1:
        raise ValueError("--run must be >= 1")

    save_root = Path(args.save_path).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    exp_tag = f"exp{int(args.run)}"

    targets = [
        ("base_line/2random_policy", "RAND"),
        ("base_line/3qucb", "Q_UCB"),
        ("base_line/4qths", "Q_THS"),
        ("base_line/5cqb_epsilon", "CQB_eps"),
        ("train_by_model/routerbench", "AQCB"),
    ]

    dataset_spec = {
        "routerbench": {"script": "routerbench_train.py", "dargs": ["--data", args.routerbench_csv]},
        "sprout": {"script": "sprout_train.py", "dargs": ["--data", args.sprout_csv]},
        "embedllm": {"script": "embedllm_train.py", "dargs": ["--data", args.embedllm_csv]},
    }

    common_args = [
        "--job_pool_size", str(args.job_pool_size),
        "--lam_list", *[str(x) for x in args.lam_list],
    ]

    extra_args = shlex.split(args.extra_args) if args.extra_args.strip() else []
    extra_has_seed = _has_seed_in_extra(extra_args)

    allow_algs = None
    if args.algs is not None and len(args.algs) > 0:
        allow_algs = set(args.algs)

    print("BASE_DIR:", str(BASE_DIR))
    print("SAVE_ROOT:", str(save_root))
    print("RUN:", args.run, "=>", exp_tag)
    print("DATASETS:", args.datasets)
    print("job_pool_size:", args.job_pool_size, "lam_list:", args.lam_list)
    if allow_algs is not None:
        print("algs:", sorted(list(allow_algs)))
    if args.extra_args.strip():
        print("extra_args:", args.extra_args.strip())
    print()

    for ds in args.datasets:
        script_name = dataset_spec[ds]["script"]
        ds_args = dataset_spec[ds]["dargs"]

        base_seed = _get_base_seed(BASE_DIR, ds)
        auto_seed = base_seed + (int(args.run) - 1)

        assort_k = _get_assort_k(BASE_DIR, ds)
        arrival_rate = _get_arrival_rate(BASE_DIR, ds)
        ar_tag = _format_ar_tag(arrival_rate)

        if allow_algs is not None:
            targets_filtered = [t for t in targets if t[1] in allow_algs]
        else:
            targets_filtered = list(targets)

        if assort_k >= 2:
            skip_names = {"RAND", "Q_UCB", "Q_THS"}
            targets_run = [t for t in targets_filtered if t[1] not in skip_names]
        else:
            targets_run = list(targets_filtered)

        print(f"\n========== DATASET: {ds} ==========")
        print("assort_K:", assort_k)
        print("arrival_rate:", arrival_rate, "=>", ar_tag)
        if assort_k >= 2:
            print("[Skip] assort_K>=2 -> skipping:", ["RAND", "Q_UCB", "Q_THS"])
        if allow_algs is not None:
            print("[Filter] running only:", sorted(list(allow_algs)))
        if not extra_has_seed:
            print(f"[Seed] base_seed={base_seed} + (run-1)={args.run-1} => final_seed={auto_seed}")
        else:
            print("[Seed] Skipping auto-seed injection because --seed is provided in --extra_args")

        for folder_path, output_name in targets_run:
            if folder_path == "train_by_model/routerbench":
                target_folder = (BASE_DIR / "train_by_model" / ds).resolve()
            else:
                target_folder = (BASE_DIR / folder_path).resolve()

            script_path = (target_folder / script_name).resolve()
            if not _exists(script_path):
                print(f"[SKIP] script not found: {script_path}")
                continue

            current_output_dir = (save_root / ds / ar_tag / exp_tag / output_name).resolve()
            current_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(script_path),
                *common_args,
                "--output_dir", str(current_output_dir),
                *ds_args,
            ]

            if not extra_has_seed:
                cmd += ["--seed", str(auto_seed)]

            cmd += extra_args

            print(f"\n--- Running: {ds}/{ar_tag}/{exp_tag}/{output_name} ---")
            try:
                _run(cmd, cwd=target_folder, dry_run=args.dry_run)
                print(f"--- Finished: {ds}/{ar_tag}/{exp_tag}/{output_name} ---")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {ds}/{ar_tag}/{exp_tag}/{output_name}: {e}")
                continue

    print("\nALL DONE")


if __name__ == "__main__":
    main()

#python3 B_MNL_Bandit_disjoint/run_all.py result --run 1 --datasets sprout --job_pool_size 5000 --lam_list 0.0 0.1 1 5 --algs AQCB
