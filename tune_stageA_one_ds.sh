set -e

DS="$1"

PAIRS=(
  "0.1 0.9"
  "0.2 0.8"
  "0.3 0.7"
  "0.4 0.6"
  "0.5 0.5"
)

if [ "$DS" = "embedllm" ]; then
  ARS_K1=(0.3 0.4 0.5)
  ARS_K2=(0.45 0.55 0.65)
elif [ "$DS" = "routerbench" ]; then
  ARS_K1=(0.5 0.6 0.7)
  ARS_K2=(0.65 0.75 0.85)
elif [ "$DS" = "sprout" ]; then
  ARS_K1=(0.7 0.8 0.9)
  ARS_K2=(0.75 0.85 0.95)
else
  echo "unknown dataset: $DS"
  exit 1
fi

SAVE_ROOT="tune/${DS}/stageA"
LAM_LIST=(0 1 5)
JOB_POOL=1000

for p in "${PAIRS[@]}"; do
  set -- $p
  TN="$1"
  TP="$2"

  for K in 1 2; do
    if [ "$K" -eq 1 ]; then ARS=("${ARS_K1[@]}"); else ARS=("${ARS_K2[@]}"); fi

    for AR in "${ARS[@]}"; do
      for R in 1 2 3; do
        python3 B_MNL_Bandit_disjoint/run_all.py "${SAVE_ROOT}/tn${TN}_tp${TP}" \
          --run "$R" \
          --datasets "$DS" \
          --job_pool_size "$JOB_POOL" \
          --lam_list "${LAM_LIST[@]}" \
          --arrival_rate "$AR" \
          --assort_K "$K" \
          --algs "ACQB-CL" \
          --extra_args "--b_type mlp --offline_epochs 10 --supcon_uc_tau_neg ${TN} --supcon_uc_tau_pos ${TP} --supcon_uc_neg_cap 64 --supcon_temp 0.07"
      done
    done
  done
done
