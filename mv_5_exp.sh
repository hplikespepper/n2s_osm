#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
START_TIME=$(date +%s)
DEFAULT_EPOCH_END=200

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

find_matching_runs() {
	local graph_size=$1
	local run_root="outputs/mvpdtsp_${graph_size}"
	local run_prefix="mvpdtsp${graph_size}_mv5_makespanw4_log_"

	if [[ ! -d "${run_root}" ]]; then
		return 0
	fi

	find "${run_root}" -mindepth 1 -maxdepth 1 -type d -name "${run_prefix}*" | sort
}

get_latest_checkpoint() {
	local run_dir=$1

	find "${run_dir}" -maxdepth 1 -type f -name 'epoch-*.pt' | sort -V | tail -n 1
}

get_checkpoint_epoch() {
	local ckpt_path=$1
	local ckpt_name
	ckpt_name=$(basename "${ckpt_path}")
	ckpt_name=${ckpt_name#epoch-}
	ckpt_name=${ckpt_name%.pt}
	echo "${ckpt_name}"
}

is_completed_run() {
	local run_dir=$1
	local training_time="${run_dir}/training_time.json"
	local latest_ckpt
	local latest_epoch

	if [[ -f "${training_time}" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "${training_time}"; then
		return 0
	fi

	latest_ckpt=$(get_latest_checkpoint "${run_dir}")
	if [[ -n "${latest_ckpt}" ]]; then
		latest_epoch=$(get_checkpoint_epoch "${latest_ckpt}")
		if [[ "${latest_epoch}" -ge $((DEFAULT_EPOCH_END - 1)) ]]; then
			return 0
		fi
	fi

	return 1
}

find_resume_checkpoint() {
	local graph_size=$1
	local run_dir
	local latest_ckpt

	while IFS= read -r run_dir; do
		[[ -z "${run_dir}" ]] && continue
		if is_completed_run "${run_dir}"; then
			continue
		fi

		latest_ckpt=$(get_latest_checkpoint "${run_dir}")
		if [[ -n "${latest_ckpt}" ]]; then
			echo "${latest_ckpt}"
			return 0
		fi
	done < <(find_matching_runs "${graph_size}" | sort -r)

	return 1
}

has_completed_experiment() {
	local graph_size=$1
	local run_dir

	while IFS= read -r run_dir; do
		[[ -z "${run_dir}" ]] && continue
		if is_completed_run "${run_dir}"; then
			return 0
		fi
	done < <(find_matching_runs "${graph_size}")

	return 1
}

on_error() {
	local exit_code=$?
	local line_no=$1
	echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no} (exit code: ${exit_code})." >&2
	echo "Please check the logs above for details." >&2
	exit "$exit_code"
}

trap 'on_error $LINENO' ERR

cd "${SCRIPT_DIR}"

echo "[INFO] Starting 5-vehicle experiments for graph_size=50 and 100..."

GRAPH_SIZES=${GRAPH_SIZES:-"50 100"}

for graph_size in ${GRAPH_SIZES}; do
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
			T_max=3000
			;;
		100)
			warm_up=1
			max_grad_norm=0.3
			batch_size=600
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

	echo "[INFO] Running training with graph_size=${graph_size}, num_vehicles=5..."
	echo "[INFO] Params: batch_size=${batch_size}, epoch_size=${epoch_size}, warm_up=${warm_up}, max_grad_norm=${max_grad_norm}, lr_model=${lr_model}, lr_critic=${lr_critic}, val_batch_size=${val_batch_size}, T_max=${T_max}"
	echo "[INFO] Objective: distance + 4 * makespan"

	if has_completed_experiment "${graph_size}"; then
		echo "[INFO] graph_size=${graph_size} already has a completed 5-vehicle experiment. Skipping."
		continue
	fi

	resume_ckpt=""
	if resume_ckpt=$(find_resume_checkpoint "${graph_size}"); then
		resume_epoch=$(get_checkpoint_epoch "${resume_ckpt}")
		echo "[INFO] Found interrupted 5-vehicle experiment for graph_size=${graph_size}."
		echo "[INFO] Resuming from ${resume_ckpt} (last saved epoch=${resume_epoch})."
	fi

	RUN_START=$(date +%s)

	cmd=(
		python run.py
		--problem mvpdtsp
		--graph_size "${graph_size}"
		--num_vehicles 5
		--warm_up "${warm_up}"
		--max_grad_norm "${max_grad_norm}"
		--batch_size "${batch_size}"
		--epoch_size "${epoch_size}"
		--lr_model "${lr_model}"
		--lr_critic "${lr_critic}"
		--val_dataset "${val_dataset}"
		--val_batch_size "${val_batch_size}"
		--T_max "${T_max}"
		--run_name "mvpdtsp${graph_size}_mv5_makespanw4_log"
		--makespan
	)

	if [[ -n "${resume_ckpt}" ]]; then
		cmd+=(--resume "${resume_ckpt}")
	fi

	"${cmd[@]}"

	RUN_END=$(date +%s)
	RUN_ELAPSED=$((RUN_END - RUN_START))
	echo "[INFO] graph_size=${graph_size}, num_vehicles=5 completed in ${RUN_ELAPSED}s."
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] 5-vehicle experiments completed in ${ELAPSED}s."
