#!/usr/bin/env bash
# 20 个客户节点、一个物理仓库、2 辆车；使用预生成 OSM 数据。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 默认使用两张显卡；需要更换时直接修改这一行。
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting training: mvpdtsp_osm, graph_size=20, num_vehicles=2"
/home/data/phuang/miniconda3/envs/rlor/bin/python -u run.py \
    --problem mvpdtsp_osm \
    --graph_size 20 \
    --num_vehicles 2 \
    --warm_up 2 \
    --max_grad_norm 0.05 \
    --train_dataset './datasets/osm_train_20.pkl' \
    --val_m 1 \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 100 \
    --val_batch_size 100 \
    --objective proposed \
    --run_name 'mvpdtsp_osm20_mv2_proposed' \
    "$@"
