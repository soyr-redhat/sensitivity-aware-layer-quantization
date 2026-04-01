#!/bin/bash
# Benchmark GGUF models using llama-cli for perplexity measurement
#
# This script runs perplexity benchmarks on all GGUF variants (uniform + mixed)
# and compares quality vs model size

set -e

LLAMA_CLI="/workspace/llama.cpp/build/bin/llama-cli"
GGUF_DIR="/workspace/gguf_models"
OUTPUT_DIR="/workspace/outputs/benchmarks"
TEST_FILE="/workspace/test_data.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "BENCHMARKING GGUF VARIANTS WITH LLAMA-CLI"
echo "================================================================================"
echo ""

# Check if llama-cli exists
if [ ! -f "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI"
    exit 1
fi

# Create test data if it doesn't exist
if [ ! -f "$TEST_FILE" ]; then
    echo "Creating test data file..."
    cat > "$TEST_FILE" <<'EOF'
The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity measurement.
Machine learning models use statistical patterns to make predictions about data.
Natural language processing enables computers to understand and generate human language.
Quantization reduces model size by using lower precision number representations.
Deep neural networks consist of multiple layers of interconnected neurons.
EOF
    echo "✓ Test data created at $TEST_FILE"
fi

echo ""
echo "Test file: $TEST_FILE"
echo "Test file size: $(wc -c < "$TEST_FILE") bytes"
echo ""

# Function to benchmark a single model
benchmark_model() {
    local model_path="$1"
    local model_name="$2"
    local output_file="$OUTPUT_DIR/${model_name}_results.txt"

    if [ ! -f "$model_path" ]; then
        echo "  ✗ Model not found: $model_path"
        return 1
    fi

    local file_size=$(du -h "$model_path" | cut -f1)
    local file_size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path")

    echo "----------------------------------------"
    echo "Model: $model_name"
    echo "Path: $model_path"
    echo "Size: $file_size ($(echo "scale=2; $file_size_bytes / 1024 / 1024 / 1024" | bc) GB)"
    echo ""

    # Run perplexity test
    echo "Running perplexity test..."

    # Use llama-cli with perplexity mode
    # -ngl 0 = CPU only (adjust if GPU available)
    # -c 512 = context size
    # -f = input file
    # --perplexity = calculate perplexity

    $LLAMA_CLI \
        -m "$model_path" \
        -f "$TEST_FILE" \
        -c 512 \
        -ngl 0 \
        --perplexity \
        -n 0 \
        2>&1 | tee "$output_file"

    # Extract perplexity from output
    local ppl=$(grep -i "perplexity" "$output_file" | tail -1 || echo "N/A")

    echo ""
    echo "Result: $ppl"
    echo "Full output saved to: $output_file"
    echo ""

    # Save summary
    echo "$model_name|$file_size|$ppl" >> "$OUTPUT_DIR/summary.txt"
}

# Clear previous summary
rm -f "$OUTPUT_DIR/summary.txt"
echo "Model|Size|Perplexity" > "$OUTPUT_DIR/summary.txt"

echo "================================================================================"
echo "BENCHMARKING UNIFORM QUANTIZATION BASELINES"
echo "================================================================================"
echo ""

# Benchmark uniform baselines
benchmark_model "$GGUF_DIR/mistral-7b-q2k.gguf" "Q2_K"
benchmark_model "$GGUF_DIR/mistral-7b-q4km.gguf" "Q4_K_M"
benchmark_model "$GGUF_DIR/mistral-7b-q6k.gguf" "Q6_K"
benchmark_model "$GGUF_DIR/mistral-7b-q8.gguf" "Q8_0"

echo ""
echo "================================================================================"
echo "BENCHMARKING MIXED-PRECISION VARIANTS"
echo "================================================================================"
echo ""

# Benchmark mixed-precision models
benchmark_model "$GGUF_DIR/mixed/mistral-7b-aggressive_mixed.gguf" "aggressive_mixed"
benchmark_model "$GGUF_DIR/mixed/mistral-7b-balanced_mixed.gguf" "balanced_mixed"
benchmark_model "$GGUF_DIR/mixed/mistral-7b-conservative_mixed.gguf" "conservative_mixed"

echo ""
echo "================================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================================"
echo ""

# Display summary table
column -t -s '|' "$OUTPUT_DIR/summary.txt"

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
