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

CUDA_VISIBLE_DEVICES=0,1 python run.py \
	--problem mvpdtsp \
	--graph_size 20 \
	--num_vehicles 2 \
	--warm_up 2 \
	--max_grad_norm 0.05 \
	--run_name 'mvpdtsp20_makespan_log' \
	--makespan

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Experiment completed in ${ELAPSED}s."
