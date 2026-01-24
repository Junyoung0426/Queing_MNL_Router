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
    if cfg is None or not hasattr(cfg, "seed"):
        return 42
    return int(getattr(cfg, "seed"))


def _get_assort_k(BASE_DIR: Path, dataset: str) -> int:
    cfg = _get_cfg(BASE_DIR, dataset)
    if cfg is None or not hasattr(cfg, "assort_K"):
        return 1
    try:
        return int(getattr(cfg, "assort_K"))
    except Exception:
        return 1


def _get_arrival_rate(BASE_DIR: Path, dataset: str):
    cfg = _get_cfg(BASE_DIR, dataset)
    if cfg is None or not hasattr(cfg, "arrival_rate"):
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


def _tag_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m")


def _try_write_best(alg_root: Path, dry_run: bool):
    script = (Path(__file__).resolve().parent / "select_best_grid.py").resolve()
    if not script.exists():
        print("[Best] select_best_grid.py not found:", str(script))
        return
    cmd = [sys.executable, str(script), str(alg_root)]
    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("[Best] failed:", e)

def _is_done(output_dir: Path, lam_list: list[float]) -> bool:
    if not output_dir.exists():
        return False
    cfg = output_dir / "config_full.json"
    models = output_dir / "models.json"
    if not cfg.exists() or not models.exists():
        return False
    for lam in lam_list:
        if not (output_dir / f"summary_lam_{float(lam):.4f}.json").exists():
            return False
    return True


def main():
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_DATA_DIR = (BASE_DIR.parent / "Data").resolve()

    parser = argparse.ArgumentParser(description="Run training sequentially (routerbench/sprout/embedllm).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("save_path", type=str, help="Root output directory (e.g., ./result)")
    parser.add_argument("--run", type=int, required=True, help="Run index (e.g., 1,2,3...)")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["routerbench", "sprout", "embedllm"],
        choices=["routerbench", "sprout", "embedllm"],
    )

    parser.add_argument("--job_pool_size", type=int, required=True)
    parser.add_argument("--lam_list", type=float, nargs="+", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--arrival_rate", type=float, default=None)
    parser.add_argument("--routerbench_csv", type=str, default=str(DEFAULT_DATA_DIR / "routerbench_dataset.csv"))
    parser.add_argument("--sprout_csv", type=str, default=str(DEFAULT_DATA_DIR / "sprout_dataset.csv"))
    parser.add_argument("--embedllm_csv", type=str, default=str(DEFAULT_DATA_DIR / "embedllm_dataset.csv"))

    parser.add_argument(
        "--algs",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to run by output_name (e.g., ACQB-CL, CQB_eps, Q_UCB). If omitted, run all.",
    )

    parser.add_argument("--extra_args", type=str, default="")

    # None이면 sweep 안 함(=config 그대로)
    parser.add_argument("--exp_rates", type=float, nargs="+", default=None)
    parser.add_argument("--alpha_coefs", type=float, nargs="+", default=None)

    args = parser.parse_args()

    if args.run < 1:
        raise ValueError("--run must be >= 1")

    do_search = (args.exp_rates is not None) or (args.alpha_coefs is not None)

    save_root = Path(args.save_path).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    exp_tag = f"exp{int(args.run)}"

    targets = [
        ("base_line/2random_policy", "RAND"),
        ("base_line/3qucb", "Q_UCB"),
        ("base_line/4qths", "Q_THS"),
        ("base_line/5cqb_epsilon", "CQB_eps"),
        ("train_by_model/routerbench", "ACQB-CL"),
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
    print("SEARCH:", do_search)
    if args.exp_rates is not None:
        print("exp_rates:", args.exp_rates)
    if args.alpha_coefs is not None:
        print("alpha_coefs:", args.alpha_coefs)
    if allow_algs is not None:
        print("algs:", sorted(list(allow_algs)))
    if args.extra_args.strip():
        print("extra_args:", args.extra_args.strip())
    print()

    er_list = [float(x) for x in args.exp_rates] if args.exp_rates is not None else [None]
    alpha_list = [float(x) for x in args.alpha_coefs] if args.alpha_coefs is not None else [None]

    for ds in args.datasets:
        script_name = dataset_spec[ds]["script"]
        ds_args = dataset_spec[ds]["dargs"]

        base_seed = _get_base_seed(BASE_DIR, ds)
        auto_seed = base_seed + (int(args.run) - 1)

        assort_k = _get_assort_k(BASE_DIR, ds)
        arrival_rate = float(args.arrival_rate) if args.arrival_rate is not None else _get_arrival_rate(BASE_DIR, ds)
        ar_tag = _format_ar_tag(arrival_rate)

        if allow_algs is not None:
            targets_filtered = [t for t in targets if t[1] in allow_algs]
        else:
            targets_filtered = list(targets)

        if assort_k >= 2:
            skip_names = {"Q_UCB", "Q_THS"}
            targets_run = [t for t in targets_filtered if t[1] not in skip_names]
        else:
            targets_run = list(targets_filtered)

        print(f"\n========== DATASET: {ds} ==========")
        print("assort_K:", assort_k)
        print("arrival_rate:", arrival_rate, "=>", ar_tag)

        for er in er_list:
            if er is not None and arrival_rate is not None and float(er) > float(arrival_rate) + 1e-12:
                print(f"[Skip] target_explore_rate={er} > arrival_rate={arrival_rate}")
                continue

            er_tag = f"er{_tag_float(er)}" if er is not None else "erCFG"

            for acoef in alpha_list:
                a_tag = f"alpha{_tag_float(acoef)}" if acoef is not None else "alphaCFG"

                for folder_path, output_name in targets_run:
                    if folder_path == "train_by_model/routerbench":
                        target_folder = (BASE_DIR / "train_by_model" / ds).resolve()
                    else:
                        target_folder = (BASE_DIR / folder_path).resolve()

                    script_path = (target_folder / script_name).resolve()
                    if not _exists(script_path):
                        print(f"[SKIP] script not found: {script_path}")
                        continue

                    # output_dir
                    if do_search:
                        current_output_dir = (save_root / ds / ar_tag / exp_tag / output_name / er_tag / a_tag).resolve()
                    else:
                        current_output_dir = (save_root / ds / ar_tag / exp_tag / output_name).resolve()

                    if do_search:
                        run_tag = f"{ds}/{ar_tag}/{exp_tag}/{output_name}/{er_tag}/{a_tag}"
                    else:
                        run_tag = f"{ds}/{ar_tag}/{exp_tag}/{output_name}"

                    current_output_dir.mkdir(parents=True, exist_ok=True)

                    if (not args.force) and _is_done(current_output_dir, [float(x) for x in args.lam_list]):
                        print(f"[SKIP DONE] {run_tag} -> {current_output_dir}")
                        continue



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

                    if args.arrival_rate is not None:
                        cmd += ["--arrival_rate", str(float(args.arrival_rate))]
                    if er is not None:
                        cmd += ["--target_explore_rate", str(float(er))]
                    if acoef is not None:
                        cmd += ["--alpha_coef", str(float(acoef))]

                    if do_search:
                        run_tag = f"{ds}/{ar_tag}/{exp_tag}/{output_name}/{er_tag}/{a_tag}"
                    else:
                        run_tag = f"{ds}/{ar_tag}/{exp_tag}/{output_name}"

                    print(f"\n--- Running: {run_tag} ---")
                    try:
                        _run(cmd, cwd=target_folder, dry_run=args.dry_run)
                        print(f"--- Finished: {run_tag} ---")
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] {run_tag}: {e}")
                        continue

    print("\nALL DONE")

    # best_summary.json 생성은 search(=exp_rates/alpha_coefs 중 하나라도 준 경우)에서만
    if do_search:
        print("\n[Best] scanning best combo per ALG ...")
        for ds in args.datasets:
            arrival_rate = float(args.arrival_rate) if args.arrival_rate is not None else _get_arrival_rate(BASE_DIR, ds)
            ar_tag = _format_ar_tag(arrival_rate)
            exp_tag = f"exp{int(args.run)}"

            for _, output_name in targets:
                if allow_algs is not None and output_name not in allow_algs:
                    continue
                alg_root = (save_root / ds / ar_tag / exp_tag / output_name).resolve()
                if alg_root.exists():
                    _try_write_best(alg_root, args.dry_run)


if __name__ == "__main__":
    main()
