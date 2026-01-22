import os
import subprocess
import argparse
import sys

# 1. 스크립트가 위치한 '진짜 경로'를 찾습니다. (이게 핵심!)
# 사용자가 어디서 명령어를 입력했든 상관없이, 이 파일이 있는 폴더를 기준점으로 잡습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# argparse 설정
parser = argparse.ArgumentParser(description="Run routerbench training sequentially.")
parser.add_argument("save_path", type=str, help="결과를 저장할 폴더 경로")
args = parser.parse_args()

# 출력 경로는 사용자가 실행한 위치 기준(터미널 위치)로 생성됩니다.
base_output_dir = args.save_path
base_args = "--data routerbench_0shot.pkl --job_pool_size 1000 --lam_list 0.0"

# 실행할 대상 폴더 (BASE_DIR 기준 상대 경로)
targets = [
    ("base_line/2random_policy", "2random_policy"),
    ("base_line/3qucb", "3qucb"),
    ("base_line/4qths", "4qths"),
    ("base_line/5cqb_epsilon", "5cqb_epsilon"),
    ("train_by_model/routerbench", "router_model")
]

print(f"🚀 전체 실행 시작! (Script Location: {BASE_DIR})")
print(f"📂 결과 저장 위치: {base_output_dir}\n")

for folder_path, output_name in targets:
    # 2. BASE_DIR과 대상 폴더를 합쳐서 '절대 경로'를 만듭니다.
    target_folder_full_path = os.path.join(BASE_DIR, folder_path)
    script_path = os.path.join(target_folder_full_path, "routerbench_train.py")
    
    if os.path.exists(script_path):
        current_output_dir = os.path.join(base_output_dir, output_name)
        
        # 명령어 생성
        cmd = f"{sys.executable} {script_path} {base_args} --output_dir {current_output_dir}"
        
        print(f"--- Running: {output_name} ---")
        # print(f"Path: {script_path}") # 디버깅용
        
        try:
            # cwd=BASE_DIR 옵션을 주지 않습니다. 
            # 데이터 파일(pkl)을 찾을 때 경로 문제가 생길 수 있으므로, 
            # 필요하다면 데이터 파일 경로도 절대경로로 바꿔야 할 수 있습니다.
            # 일단 현재 상태로 실행합니다.
            subprocess.run(cmd, shell=True, check=True)
            print(f"--- Finished: {output_name} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {output_name}: {e}\n")
    else:
        print(f"Warning: {script_path} not found. Skipping...")