#!/bin/bash
set -e  # Exit on error

echo "========================================"
echo "PDTSP_2V Training Pipeline"
echo "========================================"
echo ""


# ============================================================================
# Step 3: Train Model
# ============================================================================
echo "=========================================="
echo "Step 3: Training PDTSP_2V model..."
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Problem: pdtsp_2v"
echo "  Graph size: $GRAPH_SIZE"
echo "  Training samples: $TRAIN_SAMPLES"
echo "  Validation samples: $VAL_SAMPLES"
echo "  Batch size: 512"
echo "  Epochs: 100"
echo ""

# Adjust these parameters based on your GPU configuration
# Experiment 1: graph_size 20
# batch_size 2x (600->1200), lr 2x (8e-5->1.6e-4, 2e-5->4e-5)
echo "Starting Experiment 1: graph_size=20, warm_up=2, max_grad_norm=0.0"
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --problem pdtsp_2v \
    --graph_size 20 \
    --warm_up 2 \
    --max_grad_norm 0.05 \
    --val_m 1 \
    --val_dataset './datasets/pdp_20.pkl' \
    --run_name 'n2s_osm_2v_20'

echo ""
echo "========================================"
echo "✅ Training completed!"
echo "========================================"
