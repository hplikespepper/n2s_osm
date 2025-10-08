#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# # Experiment 1: graph_size 20
# # batch_size 2x (600->1200), lr 2x (8e-5->1.6e-4, 2e-5->4e-5)
# echo "Starting Experiment 1: graph_size=20, warm_up=2, max_grad_norm=0.0"
# CUDA_VISIBLE_DEVICES=6,7 python run.py \
#     --problem pdtsp_osm \
#     --graph_size 20 \
#     --warm_up 2 \
#     --max_grad_norm 0.0 \
#     --val_m 1 \
#     --val_dataset './datasets/osm_val_20.pkl' \
#     --run_name 'n2s_osm_20' \
#     --batch_size 1200 \
#     --epoch_size 24000 \
#     --lr_model 1.6e-4 \
#     --lr_critic 4e-5


export CUDA_VISIBLE_DEVICES=4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29611

# Experiment 2: graph_size 50
# batch_size ~1.33x (600->800), lr ~1.33x (8e-5->1.1e-4, 2e-5->2.7e-5)
echo "Starting Experiment 2: graph_size=50, warm_up=1.5, max_grad_norm=0.15"
python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --warm_up 1.5 \
    --max_grad_norm 0.15 \
    --val_m 1 \
    --val_dataset './datasets/osm_val_50.pkl' \
    --run_name 'n2s_osm_50' \
    --batch_size 800 \
    --epoch_size 16000 \
    --lr_model 1.1e-4 \
    --lr_critic 2.7e-5

# # Experiment 3: graph_size 100
# # batch_size ~0.67x (600->400), lr ~0.67x (8e-5->5.3e-5, 2e-5->1.3e-5)
# echo "Starting Experiment 3: graph_size=100, warm_up=1, max_grad_norm=0.3"
# CUDA_VISIBLE_DEVICES=6,7 python run.py \
#     --problem pdtsp_osm \
#     --graph_size 100 \
#     --warm_up 1 \
#     --max_grad_norm 0.3 \
#     --val_m 1 \
#     --val_dataset './datasets/osm_val_100.pkl' \
#     --run_name 'n2s_osm_100'

echo "All experiments completed successfully!"
