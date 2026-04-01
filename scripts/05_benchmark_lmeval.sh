#!/bin/bash
# Comprehensive benchmark script using lm-evaluation-harness
# Matches RedHatAI's evaluation rigor

set -e

# Configuration
BASELINE_MODEL="${1:-models/mistral-7b-f16.gguf}"
QUANTIZED_MODEL="${2:-models/mistral-7b-conservative-mixed.gguf}"
OUTPUT_DIR="results/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "COMPREHENSIVE QUANTIZATION BENCHMARK"
echo "========================================================================"
echo "Baseline:  $BASELINE_MODEL"
echo "Quantized: $QUANTIZED_MODEL"
echo "Output:    $OUTPUT_DIR"
echo "========================================================================"

# Check if models exist
if [ ! -f "$BASELINE_MODEL" ]; then
    echo "ERROR: Baseline model not found: $BASELINE_MODEL"
    exit 1
fi

if [ ! -f "$QUANTIZED_MODEL" ]; then
    echo "ERROR: Quantized model not found: $QUANTIZED_MODEL"
    exit 1
fi

# Run benchmarks
python scripts/05_benchmark_lmeval.py \
    --baseline "$BASELINE_MODEL" \
    --quantized "$QUANTIZED_MODEL" \
    --output "$OUTPUT_DIR/results_${TIMESTAMP}.json" \
    --tasks all \
    --batch-size auto

echo ""
echo "✅ Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR/results_${TIMESTAMP}.json"
