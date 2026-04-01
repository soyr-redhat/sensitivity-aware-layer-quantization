# Comprehensive Benchmarking Guide

This guide explains how to run thorough benchmarks on your quantized models to match the rigor of production quantization projects like RedHatAI's.

## Overview

To properly validate quantized models, you need to test across diverse tasks:

- **Math reasoning:** GSM8k, Math 500
- **General knowledge:** MMLU-Pro
- **Instruction following:** IfEval
- **Advanced reasoning:** GPQA, AIME

**Recovery percentage** = (Quantized Score / Baseline Score) × 100

## Prerequisites

### 1. Install Dependencies

```bash
# Install lm-evaluation-harness
pip install lm-eval>=0.4.0

# Install llama-cpp-python for GGUF support
pip install llama-cpp-python
```

### 2. Verify Installation

```bash
lm_eval --help
```

## Running Benchmarks

### Method 1: Using Our Script (Recommended)

```bash
# Run comprehensive benchmarks
./scripts/05_benchmark_lmeval.sh \
  models/mistral-7b-f16.gguf \
  models/mistral-7b-conservative-mixed.gguf
```

### Method 2: Manual lm-eval Commands

#### Step 1: Benchmark Baseline Model

```bash
lm_eval \
  --model gguf \
  --model_args filename=models/mistral-7b-f16.gguf \
  --tasks gsm8k,mmlu_pro,ifeval,gpqa_diamond,math_500 \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path results/baseline/ \
  --log_samples
```

#### Step 2: Benchmark Quantized Model

```bash
lm_eval \
  --model gguf \
  --model_args filename=models/mistral-7b-conservative-mixed.gguf \
  --tasks gsm8k,mmlu_pro,ifeval,gpqa_diamond,math_500 \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path results/quantized/ \
  --log_samples
```

#### Step 3: Calculate Recovery Percentages

```python
import json

# Load results
with open('results/baseline/results.json') as f:
    baseline = json.load(f)
with open('results/quantized/results.json') as f:
    quantized = json.load(f)

# Calculate recovery for each task
for task in ['gsm8k', 'mmlu_pro', 'ifeval', 'gpqa_diamond', 'math_500']:
    baseline_score = baseline['results'][task]['acc']
    quantized_score = quantized['results'][task]['acc']
    recovery = (quantized_score / baseline_score) * 100
    print(f"{task}: {recovery:.2f}% recovery")
```

## Benchmark Tasks Explained

### GSM8k (Grade School Math)
- **Task ID:** `gsm8k`
- **Metric:** Accuracy
- **What it tests:** Basic math reasoning with word problems
- **Example:** "If a train travels 60 mph for 2 hours, how far does it go?"

### MMLU-Pro (Multi-task Language Understanding)
- **Task ID:** `mmlu_pro`
- **Metric:** Accuracy
- **What it tests:** General knowledge across 57 subjects
- **Subjects:** Science, history, law, medicine, etc.

### IfEval (Instruction Following)
- **Task ID:** `ifeval`
- **Metric:** Accuracy
- **What it tests:** Ability to follow complex instructions
- **Example:** "Write exactly 3 sentences. Use the word 'quantum' in each."

### GPQA Diamond (Graduate-level Q&A)
- **Task ID:** `gpqa_diamond`
- **Metric:** Accuracy
- **What it tests:** PhD-level science questions
- **Domains:** Physics, chemistry, biology

### Math 500
- **Task ID:** `math_500`
- **Metric:** Accuracy
- **What it tests:** Competition-level math problems
- **Difficulty:** High school to college competition level

## GGUF Model Configuration

### Option 1: Using gguf backend (recommended for llama.cpp models)

```bash
lm_eval \
  --model gguf \
  --model_args filename=path/to/model.gguf,n_ctx=2048,n_gpu_layers=-1 \
  --tasks gsm8k \
  --batch_size auto
```

**Model args:**
- `filename`: Path to GGUF file
- `n_ctx`: Context window size (default: 2048)
- `n_gpu_layers`: Layers to offload to GPU (-1 = all)
- `n_batch`: Batch size for prompt processing

### Option 2: Using Hugging Face backend (if model is on HF)

```bash
lm_eval \
  --model hf \
  --model_args pretrained=username/model-name \
  --tasks gsm8k \
  --batch_size auto
```

## Expected Results Format

### Good Quantization (99%+ Recovery)

| Benchmark | Baseline | Quantized | Recovery (%) |
|-----------|----------|-----------|--------------|
| GSM8k | 95.59 | 95.37 | **99.77** ✓ |
| MMLU-Pro | 86.96 | 86.62 | **99.61** ✓ |
| IfEval | 93.80 | 93.32 | **99.49** ✓ |

**Interpretation:** Excellent quantization - minimal quality loss

### Acceptable Quantization (95-99% Recovery)

| Benchmark | Baseline | Quantized | Recovery (%) |
|-----------|----------|-----------|--------------|
| GSM8k | 95.59 | 93.12 | **97.42** |
| MMLU-Pro | 86.96 | 84.20 | **96.83** |

**Interpretation:** Good quantization - small quality tradeoff for size

### Poor Quantization (<95% Recovery)

| Benchmark | Baseline | Quantized | Recovery (%) |
|-----------|----------|-----------|--------------|
| GSM8k | 95.59 | 85.23 | **89.17** ✗ |
| MMLU-Pro | 86.96 | 78.45 | **90.21** ✗ |

**Interpretation:** Too aggressive - rethink quantization strategy

## Troubleshooting

### Issue: "Model not found" Error

**Solution:** Ensure GGUF file exists and path is correct

```bash
ls -lh models/*.gguf
```

### Issue: Out of Memory

**Solution:** Reduce context window or batch size

```bash
lm_eval \
  --model gguf \
  --model_args filename=model.gguf,n_ctx=1024 \
  --batch_size 1
```

### Issue: Slow Evaluation

**Solution:** Use GPU acceleration

```bash
lm_eval \
  --model gguf \
  --model_args filename=model.gguf,n_gpu_layers=-1 \
  --device cuda
```

### Issue: Missing Task

**Solution:** Check available tasks

```bash
lm_eval --tasks list | grep -i math
```

## Best Practices

### 1. Always Use Same Settings for Baseline and Quantized

```bash
# Same context, same batch size, same few-shot
lm_eval --model gguf --model_args filename=baseline.gguf,n_ctx=2048 --tasks gsm8k --num_fewshot 0
lm_eval --model gguf --model_args filename=quantized.gguf,n_ctx=2048 --tasks gsm8k --num_fewshot 0
```

### 2. Run Multiple Times for Stability

Some benchmarks have randomness. Run 2-3 times and average:

```bash
for i in {1..3}; do
  lm_eval ... --output_path results/run_$i/
done
```

### 3. Save Full Results

Always use `--log_samples` to debug unexpected results:

```bash
lm_eval ... --log_samples --output_path results/
```

### 4. Test Task by Task

Don't run all tasks at once - easier to debug failures:

```bash
# Run individually
lm_eval ... --tasks gsm8k
lm_eval ... --tasks mmlu_pro
lm_eval ... --tasks ifeval
```

## Creating Publication-Quality Results

### 1. Run All Benchmarks

```bash
./scripts/05_benchmark_lmeval.sh baseline.gguf quantized.gguf
```

### 2. Document Configuration

```markdown
## Evaluation Setup

**Model:** Mistral-7B-Instruct-v0.2
**Quantization:** Conservative Mixed (50% Q4_K, 31% Q6_K, 19% Q8_0)
**Baseline:** F16 GGUF
**Framework:** lm-evaluation-harness v0.4.11
**Hardware:** NVIDIA H200 GPU
**Context:** 2048 tokens
**Few-shot:** 0-shot
```

### 3. Format Results Table

```markdown
| Benchmark | Baseline | Quantized | Recovery (%) |
|-----------|----------|-----------|--------------|
| GSM8k Platinum (0-shot) | 95.59 | 95.37 | 99.77 |
| MMLU-Pro (0-shot) | 86.96 | 86.62 | 99.61 |
| IfEval (0-shot) | 93.80 | 93.32 | 99.49 |
| AIME 2025 | 92.92 | 91.66 | 98.65 |
| GPQA diamond | 87.54 | 86.70 | 99.04 |
| Math 500 | 84.73 | 84.80 | 100.08 |

**Average Recovery: 99.60%**
```

### 4. Add to Model Card

Copy results to `MODEL_CARD_TEMPLATE.md` and fill in all placeholders.

## Next Steps

After running benchmarks:

1. ✅ Document results in `docs/RESULTS_[MODEL].md`
2. ✅ Update model card with benchmark scores
3. ✅ Create HuggingFace model card
4. ✅ Upload model to HuggingFace Hub
5. ✅ Share results with community

---

**See also:**
- [Benchmark Template](BENCHMARK_TEMPLATE.md)
- [Model Card Template](../MODEL_CARD_TEMPLATE.md)
- [lm-evaluation-harness docs](https://github.com/EleutherAI/lm-evaluation-harness)
