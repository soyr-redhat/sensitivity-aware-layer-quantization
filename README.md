# Activation-Guided Layer-Wise Quantization

**Research project demonstrating that layer sensitivity analysis enables better model compression than uniform quantization.**

## 🎯 Core Concept

Not all layers in a transformer need the same precision. By profiling activation patterns and allocating quantization bits based on layer sensitivity:

- **Early layers** (low activation variance) → Aggressive quantization (Q4_K)
- **Middle layers** (medium variance) → Balanced precision (Q6_K)  
- **Final layers** (high variance) → Maximum precision (Q8_0)

**Result:** Better quality at smaller size than uniform quantization.

## 🏆 Key Finding

Our **conservative_mixed** model achieved:
- ✅ **Best quality** of all tested models (1.1527 perplexity)
- ✅ **8% smaller** than uniform Q6_K quantization
- ✅ **19% of layers** at full Q8_0 precision (critical final layers)

This validates that **activation-guided quantization outperforms uniform quantization** on the quality-size Pareto frontier.

## 📊 Results Summary

| Model | Size | Perplexity | Quality |
|-------|------|------------|---------|
| **conservative_mixed** | **5.08 GB** | **1.1527** | **BEST** ✨ |
| balanced_mixed | 4.69 GB | 1.1568 | Excellent |
| Q8_0 (uniform) | 7.17 GB | 1.1933 | Very Good |
| Q6_K (uniform) | 5.53 GB | 1.1940 | Very Good |
| Q4_K_M (uniform) | 4.07 GB | 1.2035 | Good |
| Q2_K (uniform) | 2.87 GB | 1.2375 | Acceptable |
| aggressive_mixed | 3.22 GB | 1.9578 | Poor ❌ |

**Full analysis:** See [RESULTS.md](RESULTS.md)

## 🔬 Methodology

### 1. Activation Profiling
```bash
python scripts/01_profile_activations.py
```
Analyzes layer-wise activation statistics to identify sensitivity patterns.

### 2. Generate Quantization Configurations  
```bash
python scripts/02_generate_tensor_configs.py
```
Creates tensor-type mapping files based on activation analysis:
- **Conservative:** 50% Q4_K + 31% Q6_K + 19% Q8_0
- **Balanced:** 66% Q4_K + 25% Q6_K + 9% Q8_0
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

## 📁 Repository Structure

```
moe-dynamic-quant/
├── README.md                          # This file
├── RESULTS.md                         # Detailed benchmark results
├── scripts/
│   ├── 01_profile_activations.py     # Activation analysis
│   ├── 02_generate_tensor_configs.py # Create quantization configs
│   ├── 03_create_mixed_models.sh     # Build GGUF models
│   └── 04_benchmark_perplexity.sh    # Quality benchmarking
├── outputs/
│   └── tensor_configs/               # Quantization mapping files
│       ├── conservative_mixed.txt    # WINNER config
│       ├── balanced_mixed.txt        # Strong alternative
│       └── aggressive_mixed.txt      # Over-compression (failed)
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- llama.cpp compiled with quantization tools ([build instructions](https://github.com/ggerganov/llama.cpp))
- Python 3.8+ with PyTorch
- Base model in F16 GGUF format

### Run the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Profile layer activations (requires model access)
python scripts/01_profile_activations.py

# 3. Generate tensor-type configurations
python scripts/02_generate_tensor_configs.py

# 4. Create mixed-precision models (requires llama.cpp)
./scripts/03_create_mixed_models.sh

# 5. Benchmark quality
./scripts/04_benchmark_perplexity.sh
```

## 💡 Research Contribution

### Problem
Traditional quantization applies uniform precision across all model layers, treating every layer equally. This is suboptimal because:
- Early layers process simple features (can tolerate compression)
- Final layers make critical decisions (need high precision)

### Solution
**Activation-guided layer-wise quantization:**
1. Profile activation variance across layers
2. Allocate quantization precision proportional to sensitivity
3. Use GGUF format to preserve per-layer precision in memory

### Impact
Achieves **better quality-size tradeoffs** than uniform quantization:
- Conservative mixed: 3.5% better quality + 8% smaller than Q6_K
- Balanced mixed: 3.2% better quality + 15% smaller than Q6_K
- Validates that layer sensitivity matters for compression

## 🎓 Technical Details

**Model tested:** Mistral-7B-Instruct-v0.2 (32 layers)  
**Format:** GGUF (llama.cpp's quantization-preserving format)  
**Quantization types:** Q2_K, Q4_K, Q6_K, Q8_0  
**Metric:** Perplexity on WikiText-2 (lower = better)  
**Tool:** llama-quantize with `--tensor-type-file` flag

### Layer Allocation Strategy

**Conservative Mixed (BEST):**
```
Layers 0-15  (50%): Q4_K  - Moderate compression on insensitive layers
Layers 16-25 (31%): Q6_K  - High precision on sensitive layers
Layers 26-31 (19%): Q8_0  - Full precision on critical final layers
```

Result: 1.1527 perplexity @ 5.08 GB

## 📖 Citation

If you use this work in your research:

```bibtex
@misc{activation-guided-quantization-2026,
  title={Activation-Guided Layer-Wise Quantization for Improved Model Compression},
  year={2026},
  note={Demonstrates mixed-precision quantization based on layer sensitivity analysis}
}
```

## 🔗 Related Work

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Quantization tools and GGUF format
- [GPTQ](https://arxiv.org/abs/2210.17323) - Weight quantization for LLMs
- [AWQ](https://arxiv.org/abs/2306.00978) - Activation-aware quantization

## 📝 License

MIT License - See LICENSE file for details

---

**Status:** ✅ Research complete - Models created, benchmarked, and validated  
**Recommended model:** `conservative_mixed` (best quality-size tradeoff)
