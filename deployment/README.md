# OpenShift Deployment

Run the SALQ optimizer on OpenShift with GPU acceleration.

## Prerequisites

- OpenShift cluster with GPU nodes (NVIDIA)
- GPU operator installed
- Sufficient GPU resources available

## Deploy Optimization Job

```bash
# Deploy the job
oc apply -f deployment/optimizer-job.yaml

# Watch the job progress
oc logs -f job/layer-optimizer

# Get results when complete
oc logs job/layer-optimizer | grep -A 50 "OPTIMIZATION COMPLETE"
```

## Retrieve Optimized Model

```bash
# Get the pod name
POD=$(oc get pods -l job-name=layer-optimizer -o jsonpath='{.items[0].metadata.name}')

# Copy optimized model to local machine
oc cp ${POD}:/workspace/models/qwen-3b-optimized.gguf ./models/qwen-3b-optimized.gguf

# Copy configuration file
oc cp ${POD}:/workspace/results/optimized.txt ./configs/qwen-3b-optimized.txt

# Copy benchmark results
oc cp ${POD}:/workspace/results/optimization_results.json ./results/
```

## Clean Up

```bash
# Delete the job when done
oc delete job layer-optimizer
```

## Configuration

Edit `optimizer-job.yaml` to customize:

- **Model**: Change the download URL (line 23)
- **Layers**: Update `--num-layers` (line 55)
- **Target size**: Update `--target-size` (line 57)
- **Evaluations**: Update `--max-evals` (line 56)
- **GPU resources**: Modify `resources.requests.nvidia.com/gpu` (line 67)

## GPU Detection

The optimizer automatically detects NVIDIA GPUs using `nvidia-smi` and offloads layers for faster perplexity computation. When running on OpenShift with GPU nodes:

- Job requests 1 GPU (`nvidia.com/gpu: 1`)
- Optimizer offloads all layers to GPU (`-ngl 999`)
- Perplexity computation is GPU-accelerated

## Example Output

```
[20/20] Testing: 23×Q2_K, 10×Q4_K, 3×Q8_0
  Estimated size: 2.78 GB (3.42 BPW)
  Quantizing...
  Measuring perplexity...
  Result: PPL = 2.9533
  ⭐ New best model! Saved to /workspace/results/best_model.gguf

OPTIMIZATION COMPLETE

Results:
{
  "best_config": ["Q2_K", "Q2_K", ...],
  "best_perplexity": 2.9533,
  "best_size_gb": 2.78
}

Optimized model:
-rw-r--r-- 1 root root 2.8G qwen-3b-optimized.gguf
```

## For MIG Slices

If using NVIDIA MIG (Multi-Instance GPU) slices:

```yaml
resources:
  requests:
    nvidia.com/mig-1g.10gb: 1  # For H100/H200 MIG slice
  limits:
    nvidia.com/mig-1g.10gb: 1
```

The optimizer will detect and use the MIG slice automatically.
