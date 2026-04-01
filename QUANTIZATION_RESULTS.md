# Mixed-Precision Quantization Results

**Date**: April 1, 2025  
**Model**: Mistral-7B-Instruct-v0.2  
**Method**: Activation-guided layer-wise quantization

## ✅ Successfully Created Models

All three mixed-precision GGUF variants have been created on the cluster at `/workspace/gguf_models/mixed/`

| Model | File Size | Bits Per Weight | Compression | Description |
|-------|-----------|-----------------|-------------|-------------|
| **aggressive_mixed** | 3.3 GB | 3.82 BPW | 76% | Q2_K early layers, Q4_K mid, Q6_K/Q8_0 late |
| **balanced_mixed** | 4.7 GB | 5.56 BPW | 65% | Q4_K majority, Q6_K/Q8_0 late layers |
| **conservative_mixed** | 5.1 GB | 6.03 BPW | 62% | Q4_K early, Q6_K mid, Q8_0 late layers |

### Original Model
- **F16 baseline**: 13.8 GB (16.00 BPW)

## Layer Configurations

### Aggressive Mixed (3.82 BPW)
- **Layers 0-18 (59%)**: Q2_K - Low sensitivity, maximum compression
- **Layers 19-28 (31%)**: Q4_K - Medium sensitivity  
- **Layers 29-30 (6%)**: Q6_K - High sensitivity
- **Layer 31 (3%)**: Q8_0 - Critical final layer

### Balanced Mixed (5.56 BPW)
- **Layers 0-20 (66%)**: Q4_K - Early/mid layers
- **Layers 21-28 (25%)**: Q6_K - High sensitivity
- **Layers 29-31 (9%)**: Q8_0 - Critical final layers

### Conservative Mixed (6.03 BPW)
- **Layers 0-15 (50%)**: Q4_K - Early layers
- **Layers 16-25 (31%)**: Q6_K - Mid/late layers
- **Layers 26-31 (19%)**: Q8_0 - Critical final layers

## Methodology

1. **Activation Profiling**: Analyzed layer-wise activation magnitudes across diverse prompts
2. **Sensitivity Scoring**: Computed sensitivity scores (0-1) for each layer based on activation variance
3. **Constraint Mapping**: 
   - Sensitivity < 0.3 → Allow Q2_K, Q4_K, Q6_K, Q8_0
   - Sensitivity 0.3-0.6 → Allow Q4_K, Q6_K, Q8_0
   - Sensitivity 0.6-0.85 → Allow Q6_K, Q8_0
   - Sensitivity > 0.85 → Only Q8_0
4. **Mixed Quantization**: Used llama-quantize with per-tensor quantization mappings

## Comparison with Uniform Quantization

| Uniform Model | Size | BPW | vs Closest Mixed |
|---------------|------|-----|------------------|
| Q2_K | 2.87 GB | ~2.5 | Aggressive is +15% larger but likely higher quality |
| Q4_K_M | 4.07 GB | ~4.5 | Aggressive is -19%, Balanced is +15% |
| Q6_K | 5.53 GB | ~6.0 | Balanced is -15%, Conservative is -8% |
| Q8_0 | 7.17 GB | ~8.0 | Conservative is -29% |

## Next Steps

### 1. Benchmark Memory Usage
Run `scripts/13_benchmark_gguf_variants.py` on cluster to measure:
- Actual RAM usage at inference time
- GPU memory usage (if applicable)
- Load time

### 2. Measure Quality
Evaluate perplexity on validation set:
- Compare mixed models vs uniform baselines
- Test at similar size points (e.g., balanced_mixed @ 4.7GB vs Q4_K_M @ 4.07GB)

### 3. Quality-Size Analysis
- Plot Pareto frontier: perplexity vs model size
- **Hypothesis**: Mixed-precision models should achieve better perplexity than uniform quant at same size
- **Why**: Allocates more bits to sensitive layers, fewer to insensitive ones

### 4. Expected Results
If hypothesis holds:
- `balanced_mixed` (4.7GB) should outperform `Q4_K_M` (4.07GB) in perplexity
- `conservative_mixed` (5.1GB) should match or exceed `Q6_K` (5.53GB) quality at smaller size

## Files on Cluster

```
/workspace/gguf_models/mixed/
├── mistral-7b-aggressive_mixed.gguf    (3.3 GB)
├── mistral-7b-balanced_mixed.gguf       (4.7 GB)
├── mistral-7b-conservative_mixed.gguf   (5.1 GB)
├── aggressive_mixed.log                 (quantization log)
├── balanced_mixed.log                   (quantization log)
└── conservative_mixed.log               (quantization log)
```

## Downloading Results (Optional)

To download models to local machine:
```bash
oc cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-aggressive_mixed.gguf ./outputs/
oc cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-balanced_mixed.gguf ./outputs/
oc cp user-sbowerma/mixtral-dev:/workspace/gguf_models/mixed/mistral-7b-conservative_mixed.gguf ./outputs/
```

## Research Contribution

This demonstrates **activation-guided layer-wise quantization** - a novel approach that:
1. Profiles activation patterns to identify layer sensitivity
2. Constrains quantization choices based on sensitivity scores  
3. Creates mixed-precision models optimized for the quality-size tradeoff

The key insight: **Not all layers need the same precision**. By allocating bits based on activation analysis, we can potentially improve quality at fixed model size compared to uniform quantization.
