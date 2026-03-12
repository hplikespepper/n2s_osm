#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
START_TIME=$(date +%s)

on_error() {
	local exit_code=$?
	local line_no=$1
	echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no} (exit code: ${exit_code})." >&2
	echo "Please check the logs above for details." >&2
	exit "$exit_code"
}

trap 'on_error $LINENO' ERR

echo "[INFO] Starting experiment..."

for graph_size in 50 100; do
	case "${graph_size}" in
		50)
			warm_up=1.5
			max_grad_norm=0.15
			batch_size=600
			epoch_size=12000
			lr_model=8e-5
			lr_critic=2e-5
			val_dataset='./datasets/pdp_50.pkl'
			val_batch_size=1000
			T_max=2000
			;;
		100)
			warm_up=1
			max_grad_norm=0.3
			batch_size=256
			epoch_size=12000
			lr_model=8e-5
			lr_critic=2e-5
			val_dataset='./datasets/pdp_100.pkl'
			val_batch_size=1000
			T_max=3000
			;;
		*)
			echo "[ERROR] Unsupported graph_size=${graph_size}" >&2
			exit 1
			;;
	esac

	echo "[INFO] Running training with graph_size=${graph_size}..."
	echo "[INFO] Params: batch_size=${batch_size}, epoch_size=${epoch_size}, warm_up=${warm_up}, max_grad_norm=${max_grad_norm}, lr_model=${lr_model}, lr_critic=${lr_critic}, val_batch_size=${val_batch_size}, T_max=${T_max}"
	RUN_START=$(date +%s)

	CUDA_VISIBLE_DEVICES=0,1 python run.py \
		--problem mvpdtsp \
		--graph_size "${graph_size}" \
		--num_vehicles 2 \
		--warm_up "${warm_up}" \
		--max_grad_norm "${max_grad_norm}" \
		--batch_size "${batch_size}" \
		--epoch_size "${epoch_size}" \
		--lr_model "${lr_model}" \
		--lr_critic "${lr_critic}" \
		--val_dataset "${val_dataset}" \
		--val_batch_size "${val_batch_size}" \
		--T_max "${T_max}" \
		--run_name "mvpdtsp${graph_size}_makespan_log" \
		--makespan

	RUN_END=$(date +%s)
	RUN_ELAPSED=$((RUN_END - RUN_START))
	echo "[INFO] graph_size=${graph_size} completed in ${RUN_ELAPSED}s."
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Experiment completed in ${ELAPSED}s."
