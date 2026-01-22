# run_all.py
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
    """
    qc_path의 QueueConfig().seed를 읽는다.
    import 충돌 피하려고 파일 경로 기반으로 모듈을 로드한다.
    실패하면 None 반환한다.
    """
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
    """
    dataset별 train_by_model/{dataset}/queue_config.py의 기본 seed를 읽는다.
    없으면 train_by_model/routerbench/queue_config.py로 fallback한다.
    그마저도 없으면 42를 쓴다.
    """
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

    parser = argparse.ArgumentParser(description="Run training sequentially (routerbench/sprout/embedllm).")
    parser.add_argument("save_path", type=str, help="결과를 저장할 루트 폴더 (예: ./result)")
    parser.add_argument("--run", type=int, required=True, help="실험 run 번호 (예: 1,2,3...)")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["routerbench", "sprout", "embedllm"],
        choices=["routerbench", "sprout", "embedllm"],
    )
    parser.add_argument("--job_pool_size", type=int, default=1000)
    parser.add_argument("--lam_list", type=float, nargs="+", default=[0.0])
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--routerbench_data", type=str, default="routerbench_0shot.pkl")
    parser.add_argument("--sprout_data", type=str, default="CARROT-LLM-Routing/SPROUT-o3mini")
    parser.add_argument("--sprout_split", type=str, default="train")
    parser.add_argument("--embedllm_data", type=str, default="RZ412/EmbedLLM")
    parser.add_argument("--embedllm_split", type=str, default="train")

    parser.add_argument("--extra_args", type=str, default="", help="모든 스크립트에 공통으로 더 넘길 인자")
    args = parser.parse_args()

    if args.run < 1:
        raise ValueError("--run은 1 이상의 정수여야 한다")

    save_root = Path(args.save_path).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    exp_tag = f"exp{int(args.run)}"

    targets = [
        ("base_line/2random_policy", "RAND"),
        ("base_line/3qucb", "Q_UCB"),
        ("base_line/4qths", "Q_THS"),
        ("base_line/5qcb_epsilon", "QCB_eps"),
        ("train_by_model/routerbench", "AQCB"),
    ]

    routerbench_data_path = Path(args.routerbench_data).expanduser()
    if not routerbench_data_path.is_absolute():
        routerbench_data_path = (BASE_DIR / routerbench_data_path).resolve()

    dataset_spec = {
        "routerbench": {
            "script": "routerbench_train.py",
            "dargs": ["--data", str(routerbench_data_path)],
        },
        "sprout": {
            "script": "sprout_train.py",
            "dargs": ["--data", str(args.sprout_data), "--hf_split", str(args.sprout_split)],
        },
        "embedllm": {
            "script": "embedllm_train.py",
            "dargs": ["--data", str(args.embedllm_data), "--hf_split", str(args.embedllm_split)],
        },
    }

    # 공통 args (lam_list는 nargs+라서 리스트로 펼쳐서 넣는다)
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
        if ds not in dataset_spec:
            print(f"[SKIP] unknown dataset={ds}")
            continue

        script_name = dataset_spec[ds]["script"]
        ds_args = dataset_spec[ds]["dargs"]

        base_seed = _get_base_seed(BASE_DIR, ds)
        auto_seed = base_seed + (int(args.run) - 1)

        print(f"\n========== DATASET: {ds} ==========")
        if not extra_has_seed:
            print(f"[Seed] base_seed={base_seed} + (run-1)={args.run-1} => final_seed={auto_seed}")
        else:
            print("[Seed] extra_args에 --seed가 있어 자동 seed 주입을 건너뛴다")

        for folder_path, output_name in targets:
            if folder_path.startswith("train_by_model/"):
                target_folder = (BASE_DIR / "train_by_model" / ds).resolve()
            else:
                target_folder = (BASE_DIR / folder_path).resolve()

            script_path = (target_folder / script_name).resolve()
            if not _exists(script_path):
                print(f"[SKIP] script not found: {script_path}")
                continue

            # 저장 경로: <save_root>/<dataset>/exp<run>/<algo>/
            current_output_dir = (save_root / ds / exp_tag / output_name).resolve()
            current_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(script_path),
                *common_args,
                "--output_dir", str(current_output_dir),
                *ds_args,
            ]

            # 자동 seed 주입
            if not extra_has_seed:
                cmd += ["--seed", str(auto_seed)]

            # 사용자 추가 인자
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
#python run_all.py ./result --run 1 --datasets routerbench