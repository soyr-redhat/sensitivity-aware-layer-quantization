# Comprehensive Benchmark Results

**Model:** [Model Name]  
**Quantization Method:** Activation-Guided Layer-Wise Quantization  
**Date:** [Date]

---

## Summary

[Brief description of the model and quantization approach]

**Key Results:**
- Average recovery: XX.XX%
- Model size reduction: XX%
- Maintained XX% performance across all benchmarks

---

## Benchmark Results

### Core Benchmarks

| Benchmark | Baseline (F16/Q8) | Quantized | Recovery (%) |
|-----------|-------------------|-----------|--------------|
| GSM8k (0-shot) | XX.XX | XX.XX | XX.XX |
| MMLU-Pro (0-shot) | XX.XX | XX.XX | XX.XX |
| IfEval (0-shot) | XX.XX | XX.XX | XX.XX |
| GPQA Diamond (0-shot) | XX.XX | XX.XX | XX.XX |
| Math 500 (0-shot) | XX.XX | XX.XX | XX.XX |

**Average Recovery:** XX.XX%

---

## Model Configuration

### Quantization Strategy

**Layer allocation:**
```
Layers X-Y (Z%): Q4_K  - [Description]
Layers X-Y (Z%): Q6_K  - [Description]
Layers X-Y (Z%): Q8_0  - [Description]
```

### Size Comparison

| Model | Size (GB) | BPW | Compression Ratio |
|-------|-----------|-----|-------------------|
| Baseline (F16) | X.XX | 16.0 | 1.0x |
| Baseline (Q8_0) | X.XX | 8.0 | 2.0x |
| **Optimized Mixed** | **X.XX** | **X.X** | **X.Xx** |
| Uniform Q6_K | X.XX | 6.5 | X.Xx |
| Uniform Q4_K | X.XX | 4.5 | X.Xx |

---

## Quality Analysis

### Recovery by Task Type

**Mathematical Reasoning (GSM8k, Math 500):**
- Average recovery: XX.XX%
- Analysis: [How well does the model maintain math abilities?]

**General Knowledge (MMLU-Pro):**
- Recovery: XX.XX%
- Analysis: [Performance on factual knowledge]

**Instruction Following (IfEval):**
- Recovery: XX.XX%
- Analysis: [Ability to follow complex instructions]

**Advanced Reasoning (GPQA, AIME):**
- Recovery: XX.XX%
- Analysis: [Performance on graduate-level reasoning]

---

## Methodology

### Evaluation Setup

**Hardware:**
- GPU: [GPU Model]
- RAM: [RAM Amount]
- Platform: [Platform]

**Evaluation Framework:**
- Tool: lm-evaluation-harness v[version]
- Batch size: [batch size]
- Context length: [context length]

**Baseline Model:**
- Source: [Model source/HF repo]
- Format: [GGUF F16 / Q8_0]
- Verification: [SHA256 or model card link]

### Quantization Process

1. **Optimization:**
   ```bash
   python scripts/02_optimize_layer_config.py \
     --base-model [model].gguf \
     --test-data [data].txt \
     --target-size [size] \
     --max-evals [evals]
   ```

2. **Model Creation:**
   ```bash
   llama-quantize \
     --tensor-type-file configs/optimized.txt \
     [base-model].gguf \
     [output-model].gguf \
     Q4_K
   ```

3. **Benchmarking:**
   ```bash
   ./scripts/05_benchmark_lmeval.sh \
     [baseline].gguf \
     [quantized].gguf
   ```

---

## Detailed Results

### Per-Task Breakdown

#### GSM8k (Grade School Math)
- **Baseline:** XX.XX / 100
- **Quantized:** XX.XX / 100
- **Recovery:** XX.XX%
- **Analysis:** [Task-specific insights]

#### MMLU-Pro (Multi-task Language Understanding)
- **Baseline:** XX.XX / 100
- **Quantized:** XX.XX / 100
- **Recovery:** XX.XX%
- **Analysis:** [Task-specific insights]

[Continue for each benchmark...]

---

## Conclusions

### Key Findings

1. **Quality Retention:** [Summary of quality retention]
2. **Size Efficiency:** [Summary of compression achieved]
3. **Layer Sensitivity:** [Insights about which layers needed precision]

### Recommendations

**Use this model when:**
- [Scenario 1]
- [Scenario 2]

**Consider alternatives when:**
- [Scenario 1]
- [Scenario 2]

---

## Reproduction

To reproduce these results:

```bash
# 1. Clone repository
git clone https://github.com/[username]/systematic-adaptive-layer-quantization
cd systematic-adaptive-layer-quantization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run benchmarks
./scripts/05_benchmark_lmeval.sh \
  models/baseline.gguf \
  models/optimized.gguf
```

---

## References

- Model Card: [HuggingFace link]
- Configuration: [Link to tensor-type file]
- Raw Results: [Link to JSON results]

---

**Generated with:** [Sensitivity-Aware Layer Quantization](https://github.com/[username]/systematic-adaptive-layer-quantization)
