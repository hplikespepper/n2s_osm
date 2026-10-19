#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
/home/data/phuang/miniconda3/envs/rlor/bin/python -u analyze_boca_raton.py "$@"
