#!/bin/bash
# Create mixed-precision GGUF files using llama-quantize
#
# Prerequisites:
#   - llama.cpp compiled with quantization support
#   - Base model in F16 GGUF format
#   - Tensor-type config files (from 02_generate_tensor_configs.py)
#
# Usage:
#   1. Customize paths below for your environment
#   2. Run: ./03_create_mixed_models.sh

set -e

# CUSTOMIZE THESE PATHS FOR YOUR ENVIRONMENT
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-llama-quantize}"  # or full path like "/path/to/llama-quantize"
BASE_GGUF="${BASE_GGUF:-models/mistral-7b-f16.gguf}"  # Path to your F16 base model
OUTPUT_DIR="${OUTPUT_DIR:-models/mixed}"  # Where to save mixed models
TENSOR_TYPE_DIR="${TENSOR_TYPE_DIR:-outputs/tensor_configs}"  # Tensor config files

echo "================================================================================"
echo "CREATING MIXED-PRECISION GGUF FILES"
echo "================================================================================"
echo ""
echo "Using llama-quantize: $LLAMA_QUANTIZE"
echo "Base F16 model: $BASE_GGUF"
echo "Tensor-type files: $TENSOR_TYPE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if llama-quantize exists
if [ ! -f "$LLAMA_QUANTIZE" ]; then
    echo "ERROR: llama-quantize not found at $LLAMA_QUANTIZE"
    echo "Please build llama.cpp first"
    exit 1
fi

# Check if base model exists
if [ ! -f "$BASE_GGUF" ]; then
    echo "ERROR: Base F16 model not found at $BASE_GGUF"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if tensor-type files exist
if [ ! -f "$TENSOR_TYPE_DIR/aggressive_mixed.txt" ]; then
    echo "ERROR: Tensor-type files not found in $TENSOR_TYPE_DIR"
    echo "Please copy tensor-type files from local machine"
    exit 1
fi

# Configuration 1: Aggressive Mixed
# Layers 0-18: Q2_K, 19-28: Q4_K_M, 29-30: Q6_K, 31: Q8_0
echo "Creating: Aggressive Mixed (Q2_K early, Q4_K_M mid, Q6_K/Q8_0 late)..."
echo ""

$LLAMA_QUANTIZE \
    --tensor-type-file "$TENSOR_TYPE_DIR/aggressive_mixed.txt" \
    "$BASE_GGUF" \
    "$OUTPUT_DIR/mistral-7b-aggressive_mixed.gguf" \
    Q4_K_M 2>&1 | tee "$OUTPUT_DIR/aggressive_mixed.log"

if [ -f "$OUTPUT_DIR/mistral-7b-aggressive_mixed.gguf" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/mistral-7b-aggressive_mixed.gguf" | cut -f1)
    echo "✓ Created: mistral-7b-aggressive_mixed.gguf ($SIZE)"
else
    echo "✗ Failed to create aggressive_mixed.gguf"
fi

echo ""
echo "================================================================================"
echo ""

# Configuration 2: Balanced Mixed
# Layers 0-20: Q4_K_M, 21-28: Q6_K, 29-31: Q8_0
echo "Creating: Balanced Mixed (Q4_K_M majority, Q6_K/Q8_0 late)..."
echo ""

$LLAMA_QUANTIZE \
    --tensor-type-file "$TENSOR_TYPE_DIR/balanced_mixed.txt" \
    "$BASE_GGUF" \
    "$OUTPUT_DIR/mistral-7b-balanced_mixed.gguf" \
    Q4_K_M 2>&1 | tee "$OUTPUT_DIR/balanced_mixed.log"

if [ -f "$OUTPUT_DIR/mistral-7b-balanced_mixed.gguf" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/mistral-7b-balanced_mixed.gguf" | cut -f1)
    echo "✓ Created: mistral-7b-balanced_mixed.gguf ($SIZE)"
else
    echo "✗ Failed to create balanced_mixed.gguf"
fi

echo ""
echo "================================================================================"
echo ""

# Configuration 3: Conservative Mixed
# Layers 0-15: Q4_K_M, 16-25: Q6_K, 26-31: Q8_0
echo "Creating: Conservative Mixed (Q4_K_M early, Q6_K mid, Q8_0 late)..."
echo ""

$LLAMA_QUANTIZE \
    --tensor-type-file "$TENSOR_TYPE_DIR/conservative_mixed.txt" \
    "$BASE_GGUF" \
    "$OUTPUT_DIR/mistral-7b-conservative_mixed.gguf" \
    Q4_K_M 2>&1 | tee "$OUTPUT_DIR/conservative_mixed.log"

if [ -f "$OUTPUT_DIR/mistral-7b-conservative_mixed.gguf" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/mistral-7b-conservative_mixed.gguf" | cut -f1)
    echo "✓ Created: mistral-7b-conservative_mixed.gguf ($SIZE)"
else
    echo "✗ Failed to create conservative_mixed.gguf"
fi

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
echo ""
echo "Created files:"
ls -lh "$OUTPUT_DIR"/*.gguf 2>/dev/null | grep -v "f16.gguf\|q2k.gguf\|q4km.gguf\|q6k.gguf\|q8.gguf" || echo "No mixed GGUF files created"
echo ""
