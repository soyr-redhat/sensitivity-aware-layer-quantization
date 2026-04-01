# Running Mixed-Precision Quantization on Cluster

This document describes how to create mixed-precision GGUF files on the cluster.

## Prerequisites

The following should already be set up on the cluster:
- ✅ llama.cpp built at `/workspace/llama.cpp/build/bin/llama-quantize`
- ✅ Base F16 model at `/workspace/gguf_models/mistral-7b-f16.gguf`
- ✅ Uniform GGUF models (Q2_K, Q4_K_M, Q6_K, Q8_0) in `/workspace/gguf_models/`

## Files to Transfer to Cluster

Transfer these files from your local machine to the cluster pod:

### 1. Tensor-type mapping files
Located at: `/Users/sbowerma/Code/moe-dynamic-quant/outputs/tensor_types/`
- `aggressive_mixed.txt` (224 lines, 19×Q2_K + 10×Q4_K_M + 2×Q6_K + 1×Q8_0)
- `balanced_mixed.txt` (224 lines, 21×Q4_K_M + 8×Q6_K + 3×Q8_0)
- `conservative_mixed.txt` (224 lines, 16×Q4_K_M + 10×Q6_K + 6×Q8_0)

Transfer to: `/workspace/outputs/tensor_types/`

### 2. Quantization script
Located at: `/Users/sbowerma/Code/moe-dynamic-quant/scripts/14b_create_mixed_gguf_cluster.sh`

Transfer to: `/workspace/create_mixed_gguf.sh`

## Transfer Methods

### Option A: Using kubectl (if available)
```bash
# Create directory
kubectl exec mixtral-dev -n user-sbowerma -- mkdir -p /workspace/outputs/tensor_types

# Copy tensor-type files
kubectl cp outputs/tensor_types/aggressive_mixed.txt user-sbowerma/mixtral-dev:/workspace/outputs/tensor_types/
kubectl cp outputs/tensor_types/balanced_mixed.txt user-sbowerma/mixtral-dev:/workspace/outputs/tensor_types/
kubectl cp outputs/tensor_types/conservative_mixed.txt user-sbowerma/mixtral-dev:/workspace/outputs/tensor_types/

# Copy script
kubectl cp scripts/14b_create_mixed_gguf_cluster.sh user-sbowerma/mixtral-dev:/workspace/create_mixed_gguf.sh
```

### Option B: Manual copy/paste
1. Connect to cluster pod: `kubectl exec -it mixtral-dev -n user-sbowerma -- /bin/bash`
2. Create directories: `mkdir -p /workspace/outputs/tensor_types`
3. For each tensor-type file:
   - On local: `cat outputs/tensor_types/aggressive_mixed.txt`
   - On cluster: `cat > /workspace/outputs/tensor_types/aggressive_mixed.txt` (paste, then Ctrl+D)
4. For the script:
   - On local: `cat scripts/14b_create_mixed_gguf_cluster.sh`
   - On cluster: `cat > /workspace/create_mixed_gguf.sh` (paste, then Ctrl+D)
   - Make executable: `chmod +x /workspace/create_mixed_gguf.sh`

### Option C: Using git
If the cluster has network access:
```bash
cd /workspace
git clone https://github.com/<your-repo>/moe-dynamic-quant.git
cd moe-dynamic-quant
# Files are now available in the repo
```

## Running Quantization on Cluster

Once files are transferred, connect to the cluster pod and run:

```bash
kubectl exec -it mixtral-dev -n user-sbowerma -- /bin/bash

# Inside the pod:
cd /workspace
bash create_mixed_gguf.sh
```

This will create 3 mixed-precision GGUF files:
1. `/workspace/gguf_models/mixed/mistral-7b-aggressive_mixed.gguf` (~3.5-4GB estimated)
2. `/workspace/gguf_models/mixed/mistral-7b-balanced_mixed.gguf` (~4.5-5GB estimated)
3. `/workspace/gguf_models/mixed/mistral-7b-conservative_mixed.gguf` (~5-6GB estimated)

## Expected Output

The script will:
1. Check for llama-quantize binary
2. Check for base F16 model
3. Check for tensor-type files
4. Run llama-quantize 3 times (one per config)
5. Display file sizes and success/failure status

Each quantization run should take 5-15 minutes depending on hardware.

## Troubleshooting

### Error: "llama-quantize not found"
Run the build script first:
```bash
cd /workspace
bash /path/to/build_llamacpp.sh
```

### Error: "Base F16 model not found"
Convert the model first:
```bash
cd /workspace
python3 /path/to/convert_to_gguf.py
```

### Error: "Tensor-type files not found"
Verify files were transferred correctly:
```bash
ls -la /workspace/outputs/tensor_types/
cat /workspace/outputs/tensor_types/aggressive_mixed.txt | head -20
```

### Error: "parse_tensor_type: malformed tensor type"
The tensor-type files should contain ONLY lines like:
```
blk.0.attn_q.weight=Q2_K
blk.0.attn_k.weight=Q2_K
```
No comment lines (starting with `#`) or blank lines.

## Next Steps After Creation

1. Benchmark the mixed-precision models with llama-cpp-python
2. Compare memory usage vs uniform quantization
3. Compare perplexity/quality metrics
4. Analyze the quality-size tradeoff on the Pareto frontier

## Downloading Results (Optional)

To download created files to local machine:
```bash
kubectl cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-aggressive_mixed.gguf ./outputs/
kubectl cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-balanced_mixed.gguf ./outputs/
kubectl cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-conservative_mixed.gguf ./outputs/
```
