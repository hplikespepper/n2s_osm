#!/bin/bash
# Quick start script for PDTSP_OSM training
# This script runs a small-scale test to verify everything is working

echo "========================================"
echo "PDTSP_OSM Quick Start Test"
echo "========================================"

# Activate conda environment (modify if needed)
# conda activate rlor

echo ""
echo "Step 1: Running setup validation tests..."
python test_osm_setup.py

if [ $? -ne 0 ]; then
    echo "❌ Setup tests failed. Please check the errors above."
    exit 1
fi

echo ""
echo "✅ Setup tests passed!"
echo ""
echo "Step 2: Running a small training test..."
echo "   Parameters:"
echo "   - Problem: pdtsp_osm"
echo "   - Graph size: 20 nodes"
echo "   - Batch size: 4"
echo "   - Epoch size: 20"
echo "   - Epochs: 1"
echo "   - T_train: 10"
echo "   - Single GPU mode (--no_DDP)"
echo ""

# Use CUDA_VISIBLE_DEVICES to select GPU (change to your available GPU)
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --batch_size 4 \
    --epoch_size 20 \
    --epoch_end 1 \
    --T_train 10 \
    --no_tb \
    --no_saving \
    --no_DDP \
    --osm_place "Boca Raton, Florida, USA" \
    --capacity 3

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Quick start test completed successfully!"
    echo "========================================"
    echo ""
    echo "You can now run full training with:"
    echo "  python run.py --problem pdtsp_osm --graph_size 20 --batch_size 64 --epoch_size 1000 --no_DDP"
    echo ""
    echo "Or customize the OSM location:"
    echo "  python run.py --problem pdtsp_osm --osm_place \"New York, USA\" --graph_size 20 --no_DDP"
    echo ""
    echo "Note: Add --no_DDP flag for single GPU training to avoid distributed training issues."
    echo ""
else
    echo ""
    echo "========================================"
    echo "❌ Training test failed"
    echo "========================================"
    echo "Please check the error messages above."
    exit 1
fi
