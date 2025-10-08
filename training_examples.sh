#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# # ============================================================================
# # Step 1: Generate Training Datasets in PARALLEL (充分利用CPU)
# # ============================================================================
# echo "=========================================="
# echo "Step 1: Generating training datasets in parallel..."
# echo "=========================================="

# # 并行启动3个数据生成进程
# echo "Starting 3 parallel dataset generation processes..."

# # Generate dataset for graph_size 20 (50k samples) - 后台运行
# echo "[1/3] Starting graph_size=20 dataset generation (background)..."
# python create_osm_training_dataset.py \
#     --graph_size 20 \
#     --num_samples 50000 \
#     --output './datasets/osm_train_20.pkl' \
#     --seed 1234 \
#     > dataset_gen_20.log 2>&1 &
# PID_20=$!
# echo "  → Process ID: $PID_20 (log: dataset_gen_20.log)"

# # Generate dataset for graph_size 50 (50k samples) - 后台运行
# echo "[2/3] Starting graph_size=50 dataset generation (background)..."
# python create_osm_training_dataset.py \
#     --graph_size 50 \
#     --num_samples 50000 \
#     --output './datasets/osm_train_50.pkl' \
#     --seed 2345 \
#     > dataset_gen_50.log 2>&1 &
# PID_50=$!
# echo "  → Process ID: $PID_50 (log: dataset_gen_50.log)"

# # Generate dataset for graph_size 100 (50k samples) - 后台运行
# echo "[3/3] Starting graph_size=100 dataset generation (background)..."
# python create_osm_training_dataset.py \
#     --graph_size 100 \
#     --num_samples 50000 \
#     --output './datasets/osm_train_100.pkl' \
#     --seed 3456 \
#     > dataset_gen_100.log 2>&1 &
# PID_100=$!
# echo "  → Process ID: $PID_100 (log: dataset_gen_100.log)"

# echo ""
# echo "All 3 dataset generation processes started!"
# echo "Monitor progress with: tail -f dataset_gen_20.log dataset_gen_50.log dataset_gen_100.log"
# echo ""
# echo "Waiting for all processes to complete..."

# # 等待所有后台进程完成
# wait $PID_20
# echo "✅ [1/3] graph_size=20 dataset completed!"

# wait $PID_50
# echo "✅ [2/3] graph_size=50 dataset completed!"

# wait $PID_100
# echo "✅ [3/3] graph_size=100 dataset completed!"

# echo ""
# echo "=========================================="
# echo "✅ All training datasets generated successfully!"
# echo "=========================================="
# echo ""

# ============================================================================
# Step 2: Train Models (fast with pre-generated datasets, ~2-4 hours per model)
# ============================================================================
echo "=========================================="
echo "Step 2: Training models..."
echo "=========================================="

# Experiment 1: graph_size 20
# batch_size 2x (600->1200), lr 2x (8e-5->1.6e-4, 2e-5->4e-5)
echo "Starting Experiment 1: graph_size=20, warm_up=2, max_grad_norm=0.0"
CUDA_VISIBLE_DEVICES=6,7 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --warm_up 2 \
    --max_grad_norm 0.05 \
    --train_dataset './datasets/osm_train_20.pkl' \
    --val_m 1 \
    --val_dataset './datasets/osm_val_20.pkl' \
    --run_name 'n2s_osm_20' \
    # --batch_size 1200 \
    # --epoch_size 24000 \
    # --lr_model 1.6e-4 \
    # --lr_critic 4e-5

# # Experiment 2: graph_size 50
# # batch_size ~1.33x (600->800), lr ~1.33x (8e-5->1.1e-4, 2e-5->2.7e-5)
# echo "Starting Experiment 2: graph_size=50, warm_up=1.5, max_grad_norm=0.15"
# CUDA_VISIBLE_DEVICES=6,7 python run.py \
#     --problem pdtsp_osm \
#     --graph_size 50 \
#     --warm_up 1.5 \
#     --max_grad_norm 0.15 \
#     --train_dataset './datasets/osm_train_50.pkl' \
#     --val_m 1 \
#     --val_dataset './datasets/osm_val_50.pkl' \
#     --run_name 'n2s_osm_50' \
#     --batch_size 800 \
#     --epoch_size 16000 \
#     --lr_model 1.1e-4 \
#     --lr_critic 2.7e-5

# # Experiment 3: graph_size 100
# # batch_size ~0.67x (600->400), lr ~0.67x (8e-5->5.3e-5, 2e-5->1.3e-5)
# echo "Starting Experiment 3: graph_size=100, warm_up=1, max_grad_norm=0.3"
# CUDA_VISIBLE_DEVICES=6,7 python run.py \
#     --problem pdtsp_osm \
#     --graph_size 100 \
#     --warm_up 1 \
#     --max_grad_norm 0.3 \
#     --train_dataset './datasets/osm_train_100.pkl' \
#     --val_m 1 \
#     --val_dataset './datasets/osm_val_100.pkl' \
#     --run_name 'n2s_osm_100'

echo "All experiments completed successfully!"
