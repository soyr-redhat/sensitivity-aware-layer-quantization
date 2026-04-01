#!/bin/bash
# Deploy tensor-type files to cluster and run quantization
#
# This script:
# 1. Copies tensor-type files to the cluster pod
# 2. Copies the quantization script to the cluster
# 3. Runs llama-quantize to create mixed-precision GGUF files

set -e

POD_NAME="mixtral-dev"
NAMESPACE="user-sbowerma"
LOCAL_TENSOR_DIR="/Users/sbowerma/Code/moe-dynamic-quant/outputs/tensor_types"
CLUSTER_TENSOR_DIR="/workspace/outputs/tensor_types"
CLUSTER_SCRIPT="/workspace/create_mixed_gguf.sh"

echo "================================================================================"
echo "DEPLOYING TO CLUSTER AND RUNNING QUANTIZATION"
echo "================================================================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if pod exists
echo "Checking cluster pod..."
if ! kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
    echo "ERROR: Pod $POD_NAME not found in namespace $NAMESPACE"
    echo "Please create the pod first with: kubectl apply -f deployment/dev-pod.yaml"
    exit 1
fi

echo "✓ Pod $POD_NAME is running"
echo ""

# Create directory on cluster
echo "Creating directories on cluster..."
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- mkdir -p "$CLUSTER_TENSOR_DIR"
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- mkdir -p /workspace/gguf_models/mixed
echo "✓ Directories created"
echo ""

# Copy tensor-type files
echo "Copying tensor-type files to cluster..."
for file in "$LOCAL_TENSOR_DIR"/*.txt; do
    filename=$(basename "$file")
    echo "  Copying $filename..."
    kubectl cp "$file" "$NAMESPACE/$POD_NAME:$CLUSTER_TENSOR_DIR/$filename"
done
echo "✓ Tensor-type files copied"
echo ""

# Verify files
echo "Verifying files on cluster..."
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- ls -lh "$CLUSTER_TENSOR_DIR"
echo ""

# Check if llama.cpp exists
echo "Checking llama.cpp installation..."
if kubectl exec "$POD_NAME" -n "$NAMESPACE" -- test -f /workspace/llama.cpp/build/bin/llama-quantize; then
    echo "✓ llama-quantize found"
else
    echo "ERROR: llama-quantize not found on cluster"
    echo "Please build llama.cpp on the cluster first"
    exit 1
fi
echo ""

# Check if base F16 model exists
echo "Checking for base F16 model..."
if kubectl exec "$POD_NAME" -n "$NAMESPACE" -- test -f /workspace/gguf_models/mistral-7b-f16.gguf; then
    SIZE=$(kubectl exec "$POD_NAME" -n "$NAMESPACE" -- du -h /workspace/gguf_models/mistral-7b-f16.gguf | cut -f1)
    echo "✓ Base model found ($SIZE)"
else
    echo "ERROR: Base F16 model not found at /workspace/gguf_models/mistral-7b-f16.gguf"
    echo "Please convert Mistral-7B to F16 GGUF first"
    exit 1
fi
echo ""

# Copy and run quantization script
echo "Copying quantization script to cluster..."
kubectl cp "$(dirname "$0")/14b_create_mixed_gguf_cluster.sh" "$NAMESPACE/$POD_NAME:$CLUSTER_SCRIPT"
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- chmod +x "$CLUSTER_SCRIPT"
echo "✓ Script copied"
echo ""

echo "================================================================================"
echo "RUNNING QUANTIZATION ON CLUSTER"
echo "================================================================================"
echo ""
echo "This will create 3 mixed-precision GGUF files:"
echo "  1. aggressive_mixed: Q2_K early, Q4_K_M mid, Q6_K/Q8_0 late"
echo "  2. balanced_mixed: Q4_K_M majority, Q6_K/Q8_0 late"
echo "  3. conservative_mixed: Q4_K_M early, Q6_K mid, Q8_0 late"
echo ""
echo "Starting quantization..."
echo ""

# Run the script
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- /bin/bash "$CLUSTER_SCRIPT"

echo ""
echo "================================================================================"
echo "QUANTIZATION COMPLETE"
echo "================================================================================"
echo ""
echo "Files created on cluster at: /workspace/gguf_models/mixed/"
echo ""
echo "To download files to local machine, run:"
echo "  kubectl cp $NAMESPACE/$POD_NAME:/workspace/gguf_models/mixed/mistral-7b-aggressive_mixed.gguf ./outputs/"
echo "  kubectl cp $NAMESPACE/$POD_NAME:/workspace/gguf_models/mixed/mistral-7b-balanced_mixed.gguf ./outputs/"
echo "  kubectl cp $NAMESPACE/$POD_NAME:/workspace/gguf_models/mixed/mistral-7b-conservative_mixed.gguf ./outputs/"
echo ""
