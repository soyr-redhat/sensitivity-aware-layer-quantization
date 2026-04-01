#!/bin/bash
# Deploy and run optimizer on mixtral-dev pod
set -e

POD_NAME="mixtral-dev"
NAMESPACE="user-sbowerma"

echo "========================================================================"
echo "Deploying SALQ Optimizer to mixtral-dev pod"
echo "========================================================================"
echo "Pod: $POD_NAME"
echo "Namespace: $NAMESPACE"
echo "========================================================================"

# Check if pod exists
if ! oc get pod $POD_NAME -n $NAMESPACE &>/dev/null; then
    echo "ERROR: Pod $POD_NAME not found in namespace $NAMESPACE"
    exit 1
fi

echo ""
echo "Step 1: Copying optimizer script to pod..."
oc cp scripts/02_optimize_layer_config.py $POD_NAME:/workspace/optimize.py -n $NAMESPACE

echo "Step 2: Setting up llama.cpp and dependencies..."
oc exec $POD_NAME -n $NAMESPACE -- bash -c '
set -e

echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq wget build-essential cmake git nvidia-cuda-toolkit > /dev/null 2>&1

echo "Installing Python dependencies..."
pip install -q numpy

# Check if llama.cpp is already built
if [ ! -f /workspace/llama.cpp/llama-quantize ]; then
    echo "Building llama.cpp with CUDA support..."
    cd /workspace
    if [ ! -d llama.cpp ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    cd llama.cpp
    git pull
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j $(nproc)
    echo "✓ llama.cpp built successfully"
else
    echo "✓ llama.cpp already built"
fi

echo "Dependencies ready!"
'

echo ""
echo "Step 3: Downloading model (Qwen2.5-3B Q8_0)..."
oc exec $POD_NAME -n $NAMESPACE -- bash -c '
set -e
mkdir -p /workspace/models

if [ ! -f /workspace/models/qwen-3b-q8.gguf ]; then
    echo "Downloading model..."
    wget -q --show-progress \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf" \
        -O /workspace/models/qwen-3b-q8.gguf
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi
'

echo ""
echo "Step 4: Creating test data..."
oc exec $POD_NAME -n $NAMESPACE -- bash -c '
set -e
mkdir -p /workspace/data

cat > /workspace/data/test.txt << '\''EOF'\''
The development of artificial intelligence has transformed numerous industries over the past decade. Machine learning algorithms now power everything from recommendation systems to autonomous vehicles. Natural language processing capabilities have advanced to the point where AI systems can engage in coherent conversations and generate human-like text. Computer vision systems can now identify objects, faces, and scenes with remarkable accuracy.

Deep learning architectures, particularly transformer models, have revolutionized the field of natural language understanding. These models are trained on vast amounts of text data, learning complex patterns and relationships between words and concepts. The attention mechanism allows these models to weigh the importance of different parts of the input, enabling them to handle long-range dependencies effectively.

Quantum computing represents another frontier in computational technology. Unlike classical computers that use bits representing 0 or 1, quantum computers use quantum bits or qubits that can exist in superposition states. This property allows quantum computers to explore multiple solutions simultaneously, potentially solving certain problems exponentially faster than classical computers.

Climate change poses one of the most significant challenges facing humanity in the 21st century. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Scientists emphasize the urgent need to reduce greenhouse gas emissions and transition to renewable energy sources.

Space exploration continues to captivate human imagination and drive technological innovation. Recent missions to Mars have provided unprecedented insights into the planet'\''s geology and potential for past life. Private companies are now joining government agencies in the race to explore space, developing reusable rockets and planning lunar bases.
EOF

# Repeat for enough tokens
for i in {1..5}; do cat /workspace/data/test.txt; done > /workspace/data/test_large.txt
echo "✓ Test data created"
'

echo ""
echo "========================================================================"
echo "Running Optimizer (25 evaluations on H200 MIG slice)"
echo "========================================================================"
echo ""
echo "This will:"
echo "  - Test 25 different layer quantization configurations"
echo "  - Measure perplexity on GPU for each config"
echo "  - Save the best performing model"
echo "  - Target size: 2.8 GB"
echo ""
echo "Starting optimization... (this will take 15-30 minutes)"
echo ""

oc exec $POD_NAME -n $NAMESPACE -- bash -c '
set -e
cd /workspace

python3 optimize.py \
    --base-model models/qwen-3b-q8.gguf \
    --test-data data/test_large.txt \
    --llama-quantize llama.cpp/build/bin/llama-quantize \
    --llama-perplexity llama.cpp/build/bin/llama-perplexity \
    --num-layers 36 \
    --max-evals 25 \
    --target-size 2.8 \
    --output-dir results \
    --save-config results/qwen-3b-optimized.txt \
    --save-model qwen-3b-optimized \
    --model-output-dir models
'

echo ""
echo "========================================================================"
echo "✅ OPTIMIZATION COMPLETE"
echo "========================================================================"
echo ""

# Show results
echo "Optimization Results:"
oc exec $POD_NAME -n $NAMESPACE -- cat /workspace/results/optimization_results.json

echo ""
echo "Optimized Model:"
oc exec $POD_NAME -n $NAMESPACE -- ls -lh /workspace/models/qwen-3b-optimized.gguf

echo ""
echo "========================================================================"
echo "To retrieve the optimized model:"
echo "  oc cp $POD_NAME:/workspace/models/qwen-3b-optimized.gguf ./models/qwen-3b-optimized.gguf"
echo ""
echo "To retrieve the configuration:"
echo "  oc cp $POD_NAME:/workspace/results/qwen-3b-optimized.txt ./configs/qwen-3b-optimized.txt"
echo ""
echo "To retrieve the results:"
echo "  oc cp $POD_NAME:/workspace/results/optimization_results.json ./results/qwen-3b-results.json"
echo "========================================================================"
