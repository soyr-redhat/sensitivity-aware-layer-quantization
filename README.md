# Activation-Guided Layer-Wise Quantization

Research demonstrating that layer sensitivity analysis enables better model compression than uniform quantization.

## Abstract

Not all layers in a transformer architecture require the same numerical precision. By profiling activation patterns and allocating quantization bits based on layer sensitivity, we achieve superior quality-size tradeoffs compared to uniform quantization:

- **Early layers** (low activation variance) - Aggressive quantization (Q4_K)
- **Middle layers** (medium variance) - Balanced precision (Q6_K)  
- **Final layers** (high variance) - Maximum precision (Q8_0)

## Key Results

The **conservative_mixed** configuration achieved:
- **Best quality** of all tested models (1.1527 perplexity)
- **8% smaller** than uniform Q6_K quantization (5.08 GB vs 5.53 GB)
- **19% of layers** at full Q8_0 precision on critical final layers

These results validate that activation-guided quantization outperforms uniform quantization on the quality-size Pareto frontier.

## Experimental Results

| Model | Size (GB) | Perplexity | Quality Rating |
|-------|-----------|------------|----------------|
| **conservative_mixed** | **5.08** | **1.1527** | **Best** |
| balanced_mixed | 4.69 | 1.1568 | Excellent |
| Q8_0 (uniform) | 7.17 | 1.1933 | Very Good |
| Q6_K (uniform) | 5.53 | 1.1940 | Very Good |
| Q4_K_M (uniform) | 4.07 | 1.2035 | Good |
| Q2_K (uniform) | 2.87 | 1.2375 | Acceptable |
| aggressive_mixed | 3.22 | 1.9578 | Poor |

See [RESULTS.md](RESULTS.md) for complete analysis.

## Methodology

### Option A: Bayesian Optimization (Recommended)

Find optimal layer configurations through empirical search:

```bash
python scripts/02_optimize_layer_config.py \
  --base-model models/mistral-7b-f16.gguf \
  --test-data data/perplexity_test.txt \
  --target-size 5.0 \
  --max-evals 50 \
  --save-config outputs/tensor_configs/optimized.txt
```

This approach:
1. Tests different layer quantization configurations
2. Measures actual perplexity for each configuration
3. Uses mutation-based search to explore configuration space
4. Finds the configuration with best quality within size budget
5. Outputs the optimal tensor-type file

**Pros:** Finds empirically optimal configuration for your specific model  
**Cons:** Requires ~50+ model quantizations and perplexity tests (time-intensive)

### Option B: Heuristic Allocation (Fast Alternative)

Generate configurations based on activation sensitivity heuristics:

```bash
python scripts/02_generate_configs_heuristic.py --activation-stats stats.json
```

This approach:
1. Analyzes layer sensitivity from activation statistics
2. Allocates quantization levels based on sensitivity thresholds
3. Generates multiple strategies (conservative/balanced/aggressive)

**Pros:** Fast, no need to test configurations  
**Cons:** Heuristic-based, may not find true optimum

For Mistral-7B, empirical optimization produced:
- **Conservative:** 50% Q4_K + 31% Q6_K + 19% Q8_0 (best quality)
- **Balanced:** 66% Q4_K + 25% Q6_K + 9% Q8_0 (good alternative)
- **Aggressive:** 59% Q2_K + 31% Q4_K + 10% higher (failed)

### 3. Create Mixed-Precision Models
```bash
./scripts/03_create_mixed_models.sh
```
Uses llama.cpp's `llama-quantize` with tensor-type files to create GGUF models with per-layer precision control.

### 4. Benchmark Quality
```bash
./scripts/04_benchmark_perplexity.sh
```
Measures perplexity on WikiText-2 style text to validate quality improvements.

## Repository Structure

```
activation-guided-quantization/
├── README.md                              # This file
├── RESULTS.md                             # Detailed benchmark results (Mistral-7B)
├── scripts/
│   ├── 01_profile_activations.py         # (Optional) Collect activation stats
│   ├── 02_optimize_layer_config.py       # Bayesian optimization (recommended)
│   ├── 02_generate_configs_heuristic.py  # Heuristic-based configs (fast)
│   ├── 02_generate_tensor_configs_manual.py  # Manual hardcoded configs
│   ├── 03_create_mixed_models.sh         # Build GGUF models with llama-quantize
│   └── 04_benchmark_perplexity.sh        # Quality benchmarking
├── outputs/
│   └── tensor_configs/                   # Quantization mapping files
│       ├── conservative_mixed.txt        # Mistral-7B optimal config
│       ├── balanced_mixed.txt            # Mistral-7B alternative config
│       └── aggressive_mixed.txt          # Mistral-7B over-compression baseline
└── requirements.txt
```

## Reproduction

### Prerequisites
- llama.cpp compiled with quantization tools ([build instructions](https://github.com/ggerganov/llama.cpp))
- Python 3.8+ with NumPy
- Base model in F16 GGUF format
- Test dataset for perplexity measurement

### Quick Start: Use Pre-Optimized Configs (Mistral-7B)

If using Mistral-7B, use the pre-optimized configurations:

```bash
# 1. Create mixed-precision models
./scripts/03_create_mixed_models.sh

# 2. Benchmark quality
./scripts/04_benchmark_perplexity.sh
```

### Full Pipeline: Optimize for Your Model

To find optimal configurations for a different model:

```bash
# 1. Run Bayesian optimization to find best layer configuration
python scripts/02_optimize_layer_config.py \
  --base-model /path/to/your-model-f16.gguf \
  --test-data /path/to/test_data.txt \
  --target-size 5.0 \
  --max-evals 50 \
  --save-config outputs/tensor_configs/optimized.txt

# 2. Create model with optimized config
llama-quantize \
  --tensor-type-file outputs/tensor_configs/optimized.txt \
  /path/to/your-model-f16.gguf \
  /path/to/your-model-optimized.gguf \
  Q4_K

# 3. Benchmark result
llama-perplexity \
  -m /path/to/your-model-optimized.gguf \
  -f /path/to/test_data.txt
```

## Research Contribution

### Problem Statement
Traditional quantization applies uniform precision across all model layers, treating every layer equally. This approach is suboptimal because:
- Early layers process simple features and can tolerate aggressive compression
- Final layers make critical decisions and require higher precision

### Proposed Solution
Activation-guided layer-wise quantization addresses this by:
1. Profiling activation variance across layers to identify sensitivity patterns
2. Allocating quantization precision proportional to measured sensitivity
3. Using GGUF format to preserve per-layer precision in memory

### Demonstrated Impact
Our approach achieves superior quality-size tradeoffs compared to uniform quantization:
- Conservative mixed: 3.5% better quality while being 8% smaller than Q6_K
- Balanced mixed: 3.2% better quality while being 15% smaller than Q6_K
- Empirically validates that layer sensitivity is a critical factor in compression

## Technical Details

**Model tested:** Mistral-7B-Instruct-v0.2 (32 layers)  
**Format:** GGUF (llama.cpp's quantization-preserving format)  
**Quantization types:** Q2_K, Q4_K, Q6_K, Q8_0  
**Metric:** Perplexity on WikiText-2 (lower = better)  
**Tool:** llama-quantize with `--tensor-type-file` flag

### Layer Allocation Strategy (Mistral-7B)

**Conservative Mixed (Optimal):**
```
Layers 0-15  (50%): Q4_K  - Moderate compression on insensitive layers
Layers 16-25 (31%): Q6_K  - High precision on sensitive layers
Layers 26-31 (19%): Q8_0  - Full precision on critical final layers
```

**Result:** 1.1527 perplexity at 5.08 GB model size

### Generalization to Other Models

The quantization allocations above are optimized for Mistral-7B based on its activation patterns. **Different model architectures will require different layer allocations.**

To apply this methodology to other models:
1. Profile layer activations on your target model using `01_profile_activations.py`
2. Run `02_analyze_and_generate_configs.py` with your activation statistics
3. The script will compute per-layer sensitivity and generate optimal quantization configs
4. Benchmark the resulting models to validate quality improvements

The methodology is architecture-agnostic, but the specific layer-to-quantization mappings are model-dependent.

## Citation

If you use this work in your research:

```bibtex
@misc{activation-guided-quantization-2026,
  title={Activation-Guided Layer-Wise Quantization for Improved Model Compression},
  year={2026},
  note={Demonstrates mixed-precision quantization based on layer sensitivity analysis}
}
```

## Related Work

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Quantization tools and GGUF format
- [GPTQ](https://arxiv.org/abs/2210.17323) - Weight quantization for LLMs
- [AWQ](https://arxiv.org/abs/2306.00978) - Activation-aware quantization

## License

MIT License - See LICENSE file for details

---

**Status:** Research complete - Models created, benchmarked, and validated  
**Recommended configuration:** `conservative_mixed` (optimal quality-size tradeoff)
