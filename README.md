# SALQ: Systematic Adaptive Layer Quantization

Research demonstrating that systematic layer-wise quantization optimization enables better model compression than uniform quantization.

## Abstract

Not all layers in a transformer architecture require the same numerical precision. By systematically testing layer configurations and selecting quantization levels based on empirical performance, we achieve superior quality-size tradeoffs compared to uniform quantization:

- **Early layers** (low activation variance) - Aggressive quantization (Q4_K)
- **Middle layers** (medium variance) - Balanced precision (Q6_K)  
- **Final layers** (high variance) - Maximum precision (Q8_0)

## Key Concepts

### What is Quantization?

Quantization reduces model size by storing weights with fewer bits. Instead of 16-bit floating point (F16), we use lower precision formats:

- **Q8_0**: 8 bits per weight (~8.0 BPW) - Minimal quality loss, ~2x compression
- **Q6_K**: 6.5 bits per weight (~6.5 BPW) - Good balance, ~2.5x compression
- **Q4_K**: 4.5 bits per weight (~4.5 BPW) - Aggressive compression, ~3.5x compression
- **Q2_K**: 2.8 bits per weight (~2.8 BPW) - Extreme compression, significant quality loss

BPW = "bits per weight" - the average number of bits used to store each model parameter.

### What is Perplexity?

Perplexity measures how well a language model predicts text. **Lower perplexity = better model quality.**

Technically, perplexity is the exponentiated cross-entropy loss:
```
Perplexity = 2^(average_bits_per_token)
```

A perplexity of 1.15 means the model is roughly 1.15x "confused" compared to perfect prediction. For reference:
- **< 1.20**: Excellent quality, near-original performance
- **1.20 - 1.30**: Good quality, acceptable degradation
- **> 1.50**: Noticeable quality loss

We use perplexity on WikiText-2 style text to benchmark quantization quality.

### Why Layer-Wise Quantization?

Traditional uniform quantization applies the same precision to all layers. But not all layers are equally sensitive:

- **Early layers** process simple patterns (edges, basic features) → tolerate aggressive quantization
- **Final layers** make critical decisions → require higher precision

By allocating bits based on layer sensitivity, we achieve better quality at the same model size.

## Key Results

The **conservative_mixed** configuration achieved:
- **Best quality** of all tested models (1.1527 perplexity)
- **8% smaller** than uniform Q6_K quantization (5.08 GB vs 5.53 GB)
- **19% of layers** at full Q8_0 precision on critical final layers

These results validate that systematic adaptive quantization outperforms uniform quantization on the quality-size Pareto frontier.

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

See [Mistral-7B Results](docs/RESULTS_MISTRAL.md) and [Qwen2.5-3B Test](docs/RESULTS_QWEN.md) for complete analysis.

## Methodology

### Option A: Bayesian Optimization (Recommended)

Find optimal layer configurations through empirical search:

```bash
python scripts/02_optimize_layer_config.py \
  --base-model models/mistral-7b-f16.gguf \
  --test-data data/perplexity_test.txt \
  --target-size 5.0 \
  --max-evals 50 \
  --save-config configs/optimized.txt \
  --save-model mistral-7b-optimized
```

This approach:
1. Tests different layer quantization configurations
2. Measures actual perplexity for each configuration
3. Uses mutation-based search to explore configuration space
4. Finds the configuration with best quality within size budget
5. **Saves the best model** to `models/mistral-7b-optimized.gguf`
6. Outputs the optimal tensor-type configuration file

**Pros:** Finds empirically optimal configuration for your specific model  
**Cons:** Requires ~50+ model quantizations and perplexity tests (time-intensive)

### Option B: Heuristic Allocation (Fast Alternative)

Generate configurations based on activation sensitivity heuristics:

```bash
python scripts/alternatives/heuristic_configs.py --activation-stats stats.json
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

**Perplexity (Fast):**
```bash
./scripts/04_benchmark_perplexity.sh
```
Measures perplexity on WikiText-2 style text to validate quality improvements.

**Comprehensive Benchmarks (LM Eval Harness):**
```bash
./scripts/05_benchmark_lmeval.sh models/mistral-7b-f16.gguf models/mistral-7b-optimized.gguf
```
Runs industry-standard benchmarks (GSM8k, MMLU-Pro, IfEval, GPQA, Math) to measure quality retention across multiple tasks.

## Repository Structure

```
systematic-adaptive-layer-quantization/
├── README.md                          # This file
├── LICENSE                            # MIT license
├── requirements.txt                   # Python dependencies
├── docs/
│   ├── RESULTS_MISTRAL.md            # Mistral-7B benchmark results
│   ├── RESULTS_QWEN.md               # Qwen2.5-3B large-scale test results
│   └── TESTING.md                    # Test verification
├── scripts/
│   ├── 01_profile_activations.py     # (Optional) Activation profiling
│   ├── 02_optimize_layer_config.py   # Bayesian optimizer (recommended)
│   ├── 03_create_mixed_models.sh     # Build GGUF models
│   ├── 04_benchmark_perplexity.sh    # Perplexity benchmarking
│   ├── 05_benchmark_lmeval.sh/.py    # LM Eval Harness benchmarks
│   └── alternatives/
│       ├── heuristic_configs.py      # Fast heuristic-based configs
│       └── manual_configs.py         # Manual hardcoded configs
├── configs/
│   └── mistral-7b/                   # Pre-optimized configurations
│       ├── conservative_mixed.txt    # Optimal (PPL 1.1527)
│       ├── balanced_mixed.txt        # Alternative
│       └── aggressive_mixed.txt      # Baseline
├── deployment/
│   ├── README.md                     # OpenShift deployment guide
│   └── optimizer-job.yaml            # Kubernetes/OpenShift Job spec
└── models/                            # Generated models (gitignored)
    └── *.gguf                        # Optimized quantized models
```

## Reproduction

### Prerequisites
- llama.cpp compiled with quantization tools ([build instructions](https://github.com/ggerganov/llama.cpp))
- Python 3.8+ with NumPy
- Base model in F16 GGUF format
- Test dataset for perplexity measurement

### Quick Start (OpenShift with GPU)

Run the optimizer on OpenShift cluster with GPU acceleration:

```bash
# Deploy optimization job
oc apply -f deployment/optimizer-job.yaml

# Watch progress
oc logs -f job/layer-optimizer

# Retrieve optimized model when complete
POD=$(oc get pods -l job-name=layer-optimizer -o jsonpath='{.items[0].metadata.name}')
oc cp ${POD}:/workspace/models/qwen-3b-optimized.gguf ./models/
```

See [deployment/README.md](deployment/README.md) for full OpenShift deployment guide.

### Quick Start: Use Pre-Optimized Configs (Mistral-7B)

If using Mistral-7B locally, use the pre-optimized configurations:

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
  --save-config configs/optimized.txt

# 2. Create model with optimized config
llama-quantize \
  --tensor-type-file configs/optimized.txt \
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
Systematic adaptive layer-wise quantization addresses this by:
1. Empirically testing different layer quantization configurations
2. Measuring perplexity for each configuration to determine quality
3. Using Bayesian optimization to find the best layer-precision allocation
4. Preserving the optimal configuration using GGUF tensor-type files

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
@misc{salq-2026,
  title={SALQ: Systematic Adaptive Layer Quantization for Improved Model Compression},
  year={2026},
  note={Demonstrates empirically-optimized mixed-precision quantization for LLMs}
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
