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


def _load_default_seed_from_queue_config(qc_path: Path):
    try:
        mod_name = f"_queue_config_{qc_path.stem}_{abs(hash(str(qc_path)))}"
        spec = importlib.util.spec_from_file_location(mod_name, str(qc_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "QueueConfig"):
            return None
        cfg = mod.QueueConfig()
        if not hasattr(cfg, "seed"):
            return None
        return int(cfg.seed)
    except Exception:
        return None


def _get_base_seed(BASE_DIR: Path, dataset: str) -> int:
    qc_path = (BASE_DIR / "train_by_model" / dataset / "queue_config.py").resolve()
    seed = None
    if qc_path.exists():
        seed = _load_default_seed_from_queue_config(qc_path)

    if seed is None:
        fallback = (BASE_DIR / "train_by_model" / "routerbench" / "queue_config.py").resolve()
        if fallback.exists():
            seed = _load_default_seed_from_queue_config(fallback)

    if seed is None:
        seed = 42
    return int(seed)


def main():
    BASE_DIR = Path(__file__).resolve().parent
    
    # [설정] 기본 데이터 폴더 경로
    DEFAULT_DATA_DIR = Path("/home/sjy990426/Desktop/LLM_Router/Queing_MNL_Router/Data")

    parser = argparse.ArgumentParser(description="Run training sequentially (routerbench/sprout/embedllm).")
    parser.add_argument("save_path", type=str, help="결과 저장 루트 폴더 (예: ./result)")
    parser.add_argument("--run", type=int, required=True, help="실험 run 번호 (예: 1,2,3...)")

    parser.add_argument(
        "--datasets",
        nargs="+",
        # [변경] mixinstruct 제외
        default=["routerbench", "sprout", "embedllm"],
        choices=["routerbench", "sprout", "embedllm"],
    )

    parser.add_argument("--job_pool_size", type=int, required=True, help="train에서 sample n개 뽑는 크기")
    parser.add_argument("--lam_list", type=float, nargs="+", required=True, help="lambda 리스트 (공백으로 여러 개)")

    parser.add_argument("--dry_run", action="store_true")

    # [변경] CSV 파일 경로 인자
    parser.add_argument("--routerbench_csv", type=str, default=str(DEFAULT_DATA_DIR / "routerbench_dataset.csv"))
    parser.add_argument("--sprout_csv", type=str, default=str(DEFAULT_DATA_DIR / "sprout_dataset.csv"))
    parser.add_argument("--embedllm_csv", type=str, default=str(DEFAULT_DATA_DIR / "embedllm_dataset.csv"))
    # parser.add_argument("--mixinstruct_csv", type=str, default=str(DEFAULT_DATA_DIR / "mixinstruct_dataset.csv")) # 주석 처리

    parser.add_argument("--extra_args", type=str, default="", help="모든 스크립트에 공통으로 더 넘길 인자")
    args = parser.parse_args()

    if args.run < 1:
        raise ValueError("--run은 1 이상의 정수여야 한다")

    save_root = Path(args.save_path).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    exp_tag = f"exp{int(args.run)}"

    # 실행할 알고리즘 목록
    targets = [
        ("base_line/2random_policy", "RAND"),
        ("base_line/3qucb", "Q_UCB"),
        ("base_line/4qths", "Q_THS"),
        ("base_line/5cqb_epsilon", "CQB_eps"),
        ("train_by_model/routerbench", "AQCB"),
    ]

    # 데이터셋별 실행 설정
    dataset_spec = {
        "routerbench": {
            "script": "routerbench_train.py",
            "dargs": ["--data", args.routerbench_csv],
        },
        "sprout": {
            "script": "sprout_train.py",
            "dargs": ["--data", args.sprout_csv],
        },
        "embedllm": {
            "script": "embedllm_train.py",
            "dargs": ["--data", args.embedllm_csv],
        },
        # "mixinstruct": {
        #     "script": "mixinstruct_train.py",
        #     "dargs": ["--data", args.mixinstruct_csv],
        # },
    }

    common_args = [
        "--job_pool_size", str(args.job_pool_size),
        "--lam_list", *[str(x) for x in args.lam_list],
    ]

    extra_args = shlex.split(args.extra_args) if args.extra_args.strip() else []
    extra_has_seed = _has_seed_in_extra(extra_args)

    print("BASE_DIR:", str(BASE_DIR))
    print("SAVE_ROOT:", str(save_root))
    print("RUN:", args.run, "=>", exp_tag)
    print("DATASETS:", args.datasets)
    print("job_pool_size:", args.job_pool_size, "lam_list:", args.lam_list)
    if args.extra_args.strip():
        print("extra_args:", args.extra_args.strip())
    print()

    for ds in args.datasets:
        script_name = dataset_spec[ds]["script"]
        ds_args = dataset_spec[ds]["dargs"]

        base_seed = _get_base_seed(BASE_DIR, ds)
        auto_seed = base_seed + (int(args.run) - 1)

        print(f"\n========== DATASET: {ds} ==========")
        if not extra_has_seed:
            print(f"[Seed] base_seed={base_seed} + (run-1)={args.run-1} => final_seed={auto_seed}")
        else:
            print("[Seed] extra_args에 --seed가 있어 자동 seed 주입을 안 한다")

        for folder_path, output_name in targets:
            if folder_path == "train_by_model/routerbench":
                target_folder = (BASE_DIR / "train_by_model" / ds).resolve()
            else:
                target_folder = (BASE_DIR / folder_path).resolve()

            script_path = (target_folder / script_name).resolve()
            
            if not _exists(script_path):
                print(f"[SKIP] script not found: {script_path}")
                continue

            current_output_dir = (save_root / ds / exp_tag / output_name).resolve()
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

            print(f"\n--- Running: {ds}/{exp_tag}/{output_name} ---")
            try:
                _run(cmd, cwd=target_folder, dry_run=args.dry_run)
                print(f"--- Finished: {ds}/{exp_tag}/{output_name} ---")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {ds}/{exp_tag}/{output_name}: {e}")
                continue

    print("\nALL DONE")


if __name__ == "__main__":
    main()