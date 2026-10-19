#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
cd "${N2S_DIR}"

find_latest_summary_run() {
    local root=$1
    local summary_name=$2
    local candidate
    while IFS= read -r candidate; do
        if [[ -f "${candidate}/${summary_name}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort -r)
    return 1
}

if [[ -z "${RUN_DIR:-}" ]]; then
    if ! RUN_DIR=$(find_latest_summary_run result_n2s/Fleet_Scalability summary.csv); then
        echo "[ERROR] No Fleet Scalability run containing summary.csv was found." >&2
        exit 1
    fi
fi

INPUT_CSV=${INPUT_CSV:-"${RUN_DIR}/summary.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RUN_DIR}/figures"}
PLOT_FORMATS=${PLOT_FORMATS:-png,pdf}
PLOT_DPI=${PLOT_DPI:-300}
[[ -f "${INPUT_CSV}" ]] || { echo "[ERROR] Missing summary: ${INPUT_CSV}" >&2; exit 1; }

"${PYTHON}" experiments/fleet_scalability_plot.py \
    --input "${INPUT_CSV}" --output-dir "${OUTPUT_DIR}" \
    --formats "${PLOT_FORMATS}" --dpi "${PLOT_DPI}"
echo "[INFO] Fleet Scalability figures: ${OUTPUT_DIR}"
