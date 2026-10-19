#!/usr/bin/env bash
# 可视化最近一次 epoch-198 评估结果的前 5 个实例（索引 0–4）。
# RESULTS_FILE 可指定其他结果 JSON；PYTHON、VAL_DATASET、GRAPHML 可覆盖默认值。
# 附加参数传给 vis_osm_real.py，例如 --zoom 14 或 --dpi 300。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "[ERROR] Visualization failed at line ${LINENO}." >&2' ERR

PYTHON=${PYTHON:-/home/data/phuang/miniconda3/envs/rlor/bin/python}
RESULT_ROOT=./result_n2s/mvpdtsp_osm20_mv2
OUTPUT_DIR="${RESULT_ROOT}/visualization"
VAL_DATASET=${VAL_DATASET:-./datasets/osm_val_20.pkl}
GRAPHML=${GRAPHML:-${VAL_DATASET%.pkl}.graphml}

# 按修改时间选择最新结果，并在绘图前检查是否包含足够的实例。
RESULTS_FILE=$("${PYTHON}" - "${RESULT_ROOT}" "${RESULTS_FILE:-}" <<'PY'
import json
import sys
from pathlib import Path

root, requested = sys.argv[1:]
if requested:
    path = Path(requested)
else:
    candidates = list(Path(root).glob('epoch198_*/mvpdtsp_osm_results_*.json'))
    if not candidates:
        sys.exit(f'[ERROR] No epoch-198 evaluation JSON found under {root}')
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
if not path.is_file():
    sys.exit(f'[ERROR] Results file not found: {path}')
with path.open() as handle:
    data = json.load(handle)
if data.get('problem') != 'mvpdtsp_osm':
    sys.exit('[ERROR] Expected mvpdtsp_osm evaluation results')
if len(data.get('instances', [])) < 5:
    sys.exit('[ERROR] Evaluation results must contain at least 5 instances')
print(path)
PY
)

for input_file in "${VAL_DATASET}" "${GRAPHML}"; do
    if [[ ! -f "${input_file}" ]]; then
        echo "[ERROR] File not found: ${input_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
echo "[INFO] Results: ${RESULTS_FILE}"
echo "[INFO] Output: ${OUTPUT_DIR}"
for index in 0 1 2 3 4; do
    echo "[INFO] Visualizing instance ${index}"
    "${PYTHON}" -u vis_osm_real.py \
        --results "${RESULTS_FILE}" \
        --val_dataset "${VAL_DATASET}" --graphml "${GRAPHML}" \
        --index "${index}" --output_dir "${OUTPUT_DIR}" \
        --basemap osm "$@"
done
echo "[INFO] Saved 5 instance visualizations to ${OUTPUT_DIR}"
