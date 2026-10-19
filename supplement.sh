#!/usr/bin/env bash
# Supplement the missing 20-node, 2-vehicle MC and OR-Tools baselines.
# Run from any directory. OR-Tools is resolved from the current Python or the
# existing Conda environment named "ortools".
# Example: VAL_SIZE=256 bash supplement.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trap 'echo "[ERROR] supplement.sh failed at line ${LINENO}." >&2' ERR
cd "${SCRIPT_DIR}"

MC_PYTHON=${MC_PYTHON:-${PYTHON:-python}}
ORTOOLS_PYTHON=${ORTOOLS_PYTHON:-}
VAL_SIZE=${VAL_SIZE:-256}
NUM_MC_SAMPLES=${NUM_MC_SAMPLES:-10000}
SEED=${SEED:-42}
MC_NUM_WORKERS=${MC_NUM_WORKERS:-${NUM_WORKERS:-8}}
ORTOOLS_FULL_NUM_WORKERS=${ORTOOLS_FULL_NUM_WORKERS:-${NUM_WORKERS:-16}}
ORTOOLS_LIGHT_NUM_WORKERS=${ORTOOLS_LIGHT_NUM_WORKERS:-${NUM_WORKERS:-8}}
TIME_LIMIT=15
GRAPH_SIZE=20
NUM_VEHICLES=2

# All three baselines run entirely on CPU. Each worker handles one instance.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${MC_PYTHON}" -c 'import numpy, torch; print("[INFO] MC dependencies available; CPU execution")'

# Keep OR-Tools isolated from the training environment because its protobuf
# dependency conflicts with the version used there. An explicit interpreter
# takes precedence; otherwise reuse the current Python or the named Conda env.
if [[ -z "${ORTOOLS_PYTHON}" ]]; then
    if "${MC_PYTHON}" -c 'import numpy, torch, ortools' >/dev/null 2>&1; then
        ORTOOLS_PYTHON=${MC_PYTHON}
    else
        MC_PYTHON_PATH=$(command -v "${MC_PYTHON}" 2>/dev/null || true)
        if [[ -n "${MC_PYTHON_PATH}" ]]; then
            SIBLING_ORTOOLS_PYTHON="$(dirname "$(dirname "$(dirname "${MC_PYTHON_PATH}")")")/ortools/bin/python"
            if [[ -x "${SIBLING_ORTOOLS_PYTHON}" ]]; then
                ORTOOLS_PYTHON=${SIBLING_ORTOOLS_PYTHON}
            fi
        fi
    fi
fi

if [[ -z "${ORTOOLS_PYTHON}" ]] && command -v conda >/dev/null 2>&1; then
        ORTOOLS_ENV_PREFIX=$(conda env list | awk '$1 == "ortools" {print $NF; exit}')
        if [[ -n "${ORTOOLS_ENV_PREFIX}" && -x "${ORTOOLS_ENV_PREFIX}/bin/python" ]]; then
            ORTOOLS_PYTHON="${ORTOOLS_ENV_PREFIX}/bin/python"
        fi
fi

if [[ -z "${ORTOOLS_PYTHON}" ]]; then
    echo '[ERROR] Could not find OR-Tools. Activate the "ortools" Conda environment or set ORTOOLS_PYTHON=/path/to/python.' >&2
    exit 1
fi

echo "[INFO] OR-Tools Python: ${ORTOOLS_PYTHON}"
"${ORTOOLS_PYTHON}" -c 'import numpy, torch, ortools; print("[INFO] OR-Tools dependencies available; CPU execution")'

DATASET="datasets/pdp_${GRAPH_SIZE}.pkl"
if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] Missing ${DATASET}" >&2
    exit 1
fi

make_run_dir() {
    local result_root=$1
    mkdir -p "${SCRIPT_DIR}/${result_root}"
    mktemp -d "${SCRIPT_DIR}/${result_root}/run_$(date +%Y%m%d_%H%M%S)_XXXXXX"
}

write_timing_summary() {
    local python_bin=$1
    local result_path=$2
    local summary_path=$3
    "${python_bin}" - "${result_path}" "${summary_path}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_path, summary_path = map(Path, sys.argv[1:])
with result_path.open() as f:
    result = json.load(f)
instances = result["instances"]
count = len(instances)
row = {
    "graph_size": result["graph_size"],
    "num_vehicles": result["num_vehicles"],
    "num_instances": count,
}
if "num_mc_samples" in result:
    row.update({
        "num_mc_samples": result["num_mc_samples"],
        "num_workers": result["num_workers"],
        "seed": result["seed"],
        "objective": result["objective"],
        "method": result["method"],
        "makespan_weight": result["makespan_weight"],
    })
else:
    row.update({
        "time_limit_s": result["time_limit"],
        "num_workers": result["num_workers"],
        "objective": result["objective"],
        "method": result["method"],
        "pair_relocate": result["pair_relocate"],
        "makespan_weight": result["makespan_weight"],
    })
row.update({
    "mean_instance_solve_time_s": sum(x["solve_time"] for x in instances) / count,
    "solve_wall_time_s": result["solve_wall_time"],
    "wall_time_per_instance_s": result["solve_wall_time"] / count,
    "result_file": result_path.name,
})
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
print(f"[TIME] Mean instance solve time: {row['mean_instance_solve_time_s']:.6f}s; "
      f"wall time / instances: {row['wall_time_per_instance_s']:.6f}s")
PY
}

run_mc() {
    local run_dir name output
    run_dir=$(make_run_dir result_mc)
    name="mc_${GRAPH_SIZE}_mv${NUM_VEHICLES}"
    output="${run_dir}/${name}.json"
    echo "[INFO] Running ${name}; results: ${run_dir}"
    "${MC_PYTHON}" MonteCarlo_mvpdp.py \
        --val_dataset "./${DATASET}" \
        --graph_size "${GRAPH_SIZE}" \
        --num_vehicles "${NUM_VEHICLES}" \
        --val_size "${VAL_SIZE}" \
        --num_mc_samples "${NUM_MC_SAMPLES}" \
        --num_workers "${MC_NUM_WORKERS}" \
        --seed "${SEED}" \
        --objective distance+makespan \
        --greedy \
        --output "${output}" 2>&1 | tee "${run_dir}/${name}.log"
    write_timing_summary "${MC_PYTHON}" "${output}" "${run_dir}/timing_summary.csv"
    "${MC_PYTHON}" baseline_summary.py "${run_dir}"
}

run_ortools() {
    local mode=$1
    local result_root=$2
    local workers=$3
    local include_time_suffix=$4
    local run_dir name output
    run_dir=$(make_run_dir "${result_root}")
    name="ortools_${GRAPH_SIZE}_mv${NUM_VEHICLES}"
    if [[ "${include_time_suffix}" == "yes" ]]; then
        name="${name}_t${TIME_LIMIT}"
    fi
    output="${run_dir}/${name}.json"
    echo "[INFO] Running ${name}, pair_relocate=${mode}, ${TIME_LIMIT}s/instance; results: ${run_dir}"
    "${ORTOOLS_PYTHON}" ortools_baseline.py \
        --val_dataset "./${DATASET}" \
        --graph_size "${GRAPH_SIZE}" \
        --num_vehicles "${NUM_VEHICLES}" \
        --val_size "${VAL_SIZE}" \
        --time_limit "${TIME_LIMIT}" \
        --num_workers "${workers}" \
        --objective distance+makespan \
        --pair_relocate "${mode}" \
        --output "${output}" 2>&1 | tee "${run_dir}/${name}.log"
    write_timing_summary "${ORTOOLS_PYTHON}" "${output}" "${run_dir}/timing_summary.csv"
    "${ORTOOLS_PYTHON}" baseline_summary.py "${run_dir}"
}

echo "[INFO] Supplement: graph_size=${GRAPH_SIZE}, vehicles=${NUM_VEHICLES}, val_size=${VAL_SIZE}"
run_mc
run_ortools full result_ortools_full "${ORTOOLS_FULL_NUM_WORKERS}" yes
run_ortools light result_ortools_light "${ORTOOLS_LIGHT_NUM_WORKERS}" no
echo "[INFO] All three supplemental experiments completed."
