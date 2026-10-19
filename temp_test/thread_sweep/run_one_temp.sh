#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_one_temp.sh LABEL THREADS_OR_UNLIMITED}"
threads="${2:?usage: run_one_temp.sh LABEL THREADS_OR_UNLIMITED}"
root="temp_test/thread_sweep/${label}"
mkdir -p "${root}"

source /mnt/beegfs/home/phuang2023/miniconda3/etc/profile.d/conda.sh
conda activate rlor

export CUDA_VISIBLE_DEVICES=6,7
if [[ "${threads}" == "unlimited" ]]; then
    unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
else
    export OMP_NUM_THREADS="${threads}"
    export MKL_NUM_THREADS="${threads}"
    export OPENBLAS_NUM_THREADS="${threads}"
    export NUMEXPR_NUM_THREADS="${threads}"
fi

python -u run.py \
    --problem mvpdtsp --graph_size 20 --num_vehicles 3 \
    --objective proposed --seed 1234 \
    --warm_up 2 --max_grad_norm 0.05 \
    --batch_size 600 --epoch_size 12000 --epoch_end 200 --T_train 250 \
    --val_dataset ./datasets/pdp_20.pkl --val_size 1000 --val_batch_size 1000 \
    --T_max 1500 --run_name "temp_thread_${label}" \
    --output_dir "${root}/outputs" --log_dir "${root}/logs" --no_tb \
    2>&1 | tee "${root}/training_temp.log"
