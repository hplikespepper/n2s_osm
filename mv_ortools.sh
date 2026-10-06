#!/usr/bin/env bash
# Run from any directory after `conda activate ortools`.
# Example: VAL_SIZE=256 TIME_LIMIT=15 NUM_WORKERS=8 bash mv_ortools.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trap 'echo "[ERROR] mv_ortools.sh failed at line ${LINENO}." >&2' ERR
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
VAL_SIZE=${VAL_SIZE:-256}
TIME_LIMIT=${TIME_LIMIT:-15}
NUM_WORKERS=${NUM_WORKERS:-8}

# OR-Tools search and cost evaluation run entirely on CPU.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PYTHON}" -c 'import numpy, torch, ortools; print("[INFO] Python dependencies available; CPU execution")'
for graph_size in 50 100; do
    if [[ ! -f "datasets/pdp_${graph_size}.pkl" ]]; then
        echo "[ERROR] Missing datasets/pdp_${graph_size}.pkl" >&2
        exit 1
    fi
done

mkdir -p "${SCRIPT_DIR}/result_ortools_light"
RUN_DIR=$(mktemp -d "${SCRIPT_DIR}/result_ortools_light/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "[INFO] Results: ${RUN_DIR}"
echo "[INFO] val_size=${VAL_SIZE}, time_limit=${TIME_LIMIT}s/instance, workers=${NUM_WORKERS}, pair_relocate=light"

# Run the four configurations sequentially; each uses CPU worker processes.
for graph_size in 50 100; do
    for num_vehicles in 2 5; do
        name="ortools_${graph_size}_mv${num_vehicles}"
        output="${RUN_DIR}/${name}.json"
        echo "[INFO] Running ${name} with objective: D + $((num_vehicles - 1)) * M"
        "${PYTHON}" ortools_baseline.py \
            --val_dataset "./datasets/pdp_${graph_size}.pkl" \
            --graph_size "${graph_size}" \
            --num_vehicles "${num_vehicles}" \
            --val_size "${VAL_SIZE}" \
            --time_limit "${TIME_LIMIT}" \
            --num_workers "${NUM_WORKERS}" \
            --objective distance+makespan \
            --pair_relocate light \
            --output "${output}" 2>&1 | tee "${RUN_DIR}/${name}.log"

        # Write each timing row immediately so completed experiments survive
        # a failure in a later configuration. All durations are in seconds.
        "${PYTHON}" - "${output}" "${RUN_DIR}/timing_summary.csv" <<'PY'
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
    "time_limit_s": result["time_limit"],
    "num_workers": result["num_workers"],
    "objective": result["objective"],
    "method": result["method"],
    "pair_relocate": result["pair_relocate"],
    "makespan_weight": result["makespan_weight"],
    # Mean duration within a worker, excluding pool startup and final reporting.
    "mean_instance_solve_time_s": sum(x["solve_time"] for x in instances) / count,
    # Includes pool startup/shutdown; excludes dataset loading and JSON writing.
    "solve_wall_time_s": result["solve_wall_time"],
    "wall_time_per_instance_s": result["solve_wall_time"] / count,
    "result_file": result_path.name,
}
write_header = not summary_path.exists()
with summary_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"[TIME] Mean instance solve time: {row['mean_instance_solve_time_s']:.6f}s; "
      f"wall time / instances: {row['wall_time_per_instance_s']:.6f}s")
PY
        # Refresh dataset-level quality and timing summaries after each configuration.
        "${PYTHON}" baseline_summary.py "${RUN_DIR}"
    done
done

echo "[INFO] All four experiments completed. Results and logs: ${RUN_DIR}"
echo "[INFO] Timing summary: ${RUN_DIR}/timing_summary.csv"

echo "[INFO] Experiment summary: ${RUN_DIR}/experiment_summary.csv"
