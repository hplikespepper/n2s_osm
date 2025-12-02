#!/bin/bash
set -e

echo "========================================"
echo "PDTSP_2V Evaluation Script"
echo "========================================"
echo ""

# Configuration
GRAPH_SIZE=20
MODEL_PATH="./outputs/pdtsp_2v_${GRAPH_SIZE}/epoch-99.pt"
VAL_FILE="./datasets/pdtsp_2v_val_${GRAPH_SIZE}.pkl"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "   Please train the model first or specify correct path."
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VAL_FILE" ]; then
    echo "❌ Error: Validation dataset not found at $VAL_FILE"
    echo "   Please generate validation data first."
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Validation data: $VAL_FILE"
echo "  Graph size: $GRAPH_SIZE"
echo ""

# Run evaluation
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_2v \
    --graph_size $GRAPH_SIZE \
    --val_dataset "$VAL_FILE" \
    --load_path "$MODEL_PATH" \
    --eval_only \
    --val_m 1 \
    --no_saving

echo ""
echo "========================================"
echo "✅ Evaluation completed!"
echo "========================================"
