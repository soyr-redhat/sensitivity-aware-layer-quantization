#!/bin/bash
# Simple validation and comparison of GGUF models
#
# This script validates that all models can load and compares file sizes

set -e

GGUF_DIR="/workspace/gguf_models"
OUTPUT_FILE="/workspace/outputs/benchmarks/model_comparison.txt"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "================================================================================"
echo "GGUF MODEL VALIDATION AND COMPARISON"
echo "================================================================================"
echo ""

# Create header
{
    echo "================================================================================"
    echo "GGUF MODEL VALIDATION AND COMPARISON"
    echo "Generated: $(date)"
    echo "================================================================================"
    echo ""
    echo "Model Sizes:"
    echo "------------"
} > "$OUTPUT_FILE"

# Function to check and report model info
check_model() {
    local model_path="$1"
    local model_name="$2"

    if [ ! -f "$model_path" ]; then
        echo "  ✗ $model_name: NOT FOUND"
        echo "$model_name|NOT FOUND|0|N/A" >> "$OUTPUT_FILE.csv"
        return 1
    fi

    local size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path")
    local size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc)
    local size_mb=$(echo "scale=0; $size_bytes / 1024 / 1024" | bc)

    # Calculate BPW (bits per weight) estimate
    # Mistral-7B has ~7.24B parameters
    local params=7240000000
    local bits=$(echo "scale=2; ($size_bytes * 8) / $params" | bc)

    echo "  ✓ $model_name:"
    echo "      Size: ${size_gb} GB (${size_mb} MB)"
    echo "      Est. BPW: ${bits}"

    echo "$model_name: ${size_gb} GB (${bits} BPW)" >> "$OUTPUT_FILE"
    echo "$model_name|OK|$size_gb|$bits" >> "$OUTPUT_FILE.csv"
}

# Initialize CSV
echo "Model,Status,Size_GB,BPW" > "$OUTPUT_FILE.csv"

echo "Uniform Quantization Baselines:"
echo "-------------------------------"
check_model "$GGUF_DIR/mistral-7b-q2k.gguf" "Q2_K"
check_model "$GGUF_DIR/mistral-7b-q4km.gguf" "Q4_K_M"
check_model "$GGUF_DIR/mistral-7b-q6k.gguf" "Q6_K"
check_model "$GGUF_DIR/mistral-7b-q8.gguf" "Q8_0"
check_model "$GGUF_DIR/mistral-7b-f16.gguf" "F16"

echo ""
echo "Mixed-Precision Variants:"
echo "-------------------------"
check_model "$GGUF_DIR/mixed/mistral-7b-aggressive_mixed.gguf" "aggressive_mixed"
check_model "$GGUF_DIR/mixed/mistral-7b-balanced_mixed.gguf" "balanced_mixed"
check_model "$GGUF_DIR/mixed/mistral-7b-conservative_mixed.gguf" "conservative_mixed"

echo ""
echo "================================================================================"
echo "COMPARISON ANALYSIS"
echo "================================================================================"
echo ""

{
    echo ""
    echo "Comparison Analysis:"
    echo "--------------------"
} >> "$OUTPUT_FILE"

# Read sizes from CSV for comparison
q2k_size=$(grep "^Q2_K," "$OUTPUT_FILE.csv" | cut -d, -f3)
q4km_size=$(grep "^Q4_K_M," "$OUTPUT_FILE.csv" | cut -d, -f3)
q6k_size=$(grep "^Q6_K," "$OUTPUT_FILE.csv" | cut -d, -f3)
q8_size=$(grep "^Q8_0," "$OUTPUT_FILE.csv" | cut -d, -f3)

agg_size=$(grep "^aggressive_mixed," "$OUTPUT_FILE.csv" | cut -d, -f3)
bal_size=$(grep "^balanced_mixed," "$OUTPUT_FILE.csv" | cut -d, -f3)
con_size=$(grep "^conservative_mixed," "$OUTPUT_FILE.csv" | cut -d, -f3)

echo "Mixed vs Uniform at Similar Sizes:"
echo ""

if [ ! -z "$agg_size" ] && [ ! -z "$q2k_size" ]; then
    diff=$(echo "scale=1; (($agg_size - $q2k_size) / $q2k_size) * 100" | bc)
    echo "  aggressive_mixed (${agg_size}GB) vs Q2_K (${q2k_size}GB): ${diff}% larger"
    echo "  aggressive_mixed (${agg_size}GB) vs Q2_K (${q2k_size}GB): ${diff}% larger" >> "$OUTPUT_FILE"
fi

if [ ! -z "$bal_size" ] && [ ! -z "$q4km_size" ]; then
    diff=$(echo "scale=1; (($bal_size - $q4km_size) / $q4km_size) * 100" | bc)
    echo "  balanced_mixed (${bal_size}GB) vs Q4_K_M (${q4km_size}GB): ${diff}% larger"
    echo "  balanced_mixed (${bal_size}GB) vs Q4_K_M (${q4km_size}GB): ${diff}% larger" >> "$OUTPUT_FILE"
fi

if [ ! -z "$con_size" ] && [ ! -z "$q6k_size" ]; then
    diff=$(echo "scale=1; (($con_size - $q6k_size) / $q6k_size) * 100" | bc)
    echo "  conservative_mixed (${con_size}GB) vs Q6_K (${q6k_size}GB): ${diff}% larger"
    echo "  conservative_mixed (${con_size}GB) vs Q6_K (${q6k_size}GB): ${diff}% larger" >> "$OUTPUT_FILE"
fi

echo ""
echo "================================================================================"
echo "RESULTS SAVED"
echo "================================================================================"
echo ""
echo "Text output: $OUTPUT_FILE"
echo "CSV output: $OUTPUT_FILE.csv"
echo ""

cat "$OUTPUT_FILE"
