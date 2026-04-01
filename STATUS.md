# Project Status: Layer-wise Quantization for Mistral-7B

## Current Phase: Mixed-Precision GGUF Creation (Ready for Cluster Execution)

### ✅ Completed Work

1. **Activation Profiling & Sensitivity Analysis**
   - Analyzed layer-wise activation patterns on Mistral-7B
   - Computed sensitivity scores for all 32 layers
   - Results: Layers 0-18 = low sensitivity, 19-28 = medium, 29-30 = high, 31 = critical
   - Location: `/workspace/outputs/analysis/layer_sensitivity_scores.json`

2. **Constraint Generation**
   - Mapped sensitivity scores to allowed quantization levels per layer
   - Reduced search space from 18 quintillion to 64.9 quadrillion (284× reduction)
   - Constraint rules:
     - Sensitivity < 0.3: Allow Q2_K, Q4_K_M, Q6_K, Q8_0
     - Sensitivity 0.3-0.6: Allow Q4_K_M, Q6_K, Q8_0
     - Sensitivity 0.6-0.85: Allow Q6_K, Q8_0
     - Sensitivity > 0.85: Only Q8_0

3. **Baseline GGUF Models**
   - Created uniform quantization baselines on cluster:
     - Q2_K: 2.87GB disk, 3.34GB RAM
     - Q4_K_M: 4.07GB disk, 8.26GB RAM
     - Q6_K: 5.53GB disk, 11.10GB RAM
     - Q8_0: 7.17GB disk, 15.15GB RAM
   - Verified GGUF provides REAL memory savings (unlike HuggingFace)
   - Location: `/workspace/gguf_models/`

4. **llama.cpp Setup**
   - Compiled llama.cpp on cluster with llama-quantize tool
   - Verified support for --tensor-type-file flag
   - Binary location: `/workspace/llama.cpp/build/bin/llama-quantize`

5. **Mixed-Precision Configuration Design**
   - Created 3 activation-guided configs:
     
     **aggressive_mixed**: Max compression, preserve critical layers
     - 19 layers @ Q2_K (0-18)
     - 10 layers @ Q4_K_M (19-28)
     - 2 layers @ Q6_K (29-30)
     - 1 layer @ Q8_0 (31)
     - Estimated size: ~3.5-4GB
     
     **balanced_mixed**: Quality-size balance
     - 21 layers @ Q4_K_M (0-20)
     - 8 layers @ Q6_K (21-28)
     - 3 layers @ Q8_0 (29-31)
     - Estimated size: ~4.5-5GB
     
     **conservative_mixed**: Prioritize quality
     - 16 layers @ Q4_K_M (0-15)
     - 10 layers @ Q6_K (16-25)
     - 6 layers @ Q8_0 (26-31)
     - Estimated size: ~5-6GB

6. **Tensor-Type Mapping Files**
   - Generated 3 tensor-type files for llama-quantize
   - Each file maps 224 tensors (7 tensors × 32 layers, excluding norms)
   - Format: `blk.{layer}.{tensor_name}={quant_type}`
   - Files are clean (no comments, no blank lines)
   - Location: `outputs/tensor_types/*.txt`

7. **Cluster Deployment Scripts**
   - Created quantization script for cluster execution
   - Created transfer instructions and helper scripts
   - Location: `scripts/14b_create_mixed_gguf_cluster.sh`

### 🔄 Current Task: Transfer Files to Cluster

**What needs to happen:**
1. Transfer tensor-type files to cluster: `outputs/tensor_types/*.txt` → `/workspace/outputs/tensor_types/`
2. Transfer quantization script: `scripts/14b_create_mixed_gguf_cluster.sh` → `/workspace/create_mixed_gguf.sh`
3. Run quantization script on cluster
4. Wait for 3 mixed-precision GGUF files to be created (~15-45 minutes total)

**See:** `CLUSTER_QUANTIZATION_STEPS.md` for detailed instructions

### 📋 Next Steps After GGUF Creation

1. **Benchmark Mixed-Precision Models**
   - Use script: `scripts/13_benchmark_gguf_variants.py`
   - Measure actual RAM usage for each mixed config
   - Calculate perplexity/quality metrics
   - Compare against uniform baselines

2. **Analyze Results**
   - Plot quality vs size tradeoff (Pareto frontier)
   - Compare mixed-precision to uniform quantization at same size
   - Validate hypothesis: Activation-guided layer selection improves quality at fixed size

3. **Write Up Findings**
   - Document methodology: activation profiling → constraint generation → mixed quantization
   - Show experimental results: memory savings + quality comparisons
   - Research contribution: Using activation sensitivity to guide layer-wise quantization decisions

### 📊 Expected Outcomes

**Hypothesis:** 
Mixed-precision models guided by activation sensitivity will achieve better quality than uniform quantization at the same model size.

**Example prediction:**
- `balanced_mixed` (~4.5GB) should outperform uniform Q4_K_M (4.07GB) in perplexity
- Because it allocates more bits to sensitive layers and fewer to insensitive layers

**Validation:**
Compare perplexity scores:
- Uniform Q4_K_M vs balanced_mixed (similar size)
- Uniform Q6_K vs conservative_mixed (similar size)
- Plot all configs on quality-size scatter plot

### 🛠️ Technical Stack

- **Model:** Mistral-7B-Instruct-v0.2
- **Quantization:** llama.cpp (GGUF format)
- **Analysis:** PyTorch + HuggingFace Transformers
- **Profiling:** Activation capture during forward pass
- **Deployment:** Kubernetes cluster with GPU

### 📁 Key Files

**Analysis & Configuration:**
- `src/quantization/constraints.py` - Sensitivity scoring & constraint generation
- `outputs/tensor_types/*.txt` - Per-tensor quantization mappings

**Scripts:**
- `scripts/09_analyze_activations_and_constraints.py` - Activation profiling
- `scripts/15_generate_tensor_type_files.py` - Mapping file generator
- `scripts/14b_create_mixed_gguf_cluster.sh` - Cluster quantization script
- `scripts/13_benchmark_gguf_variants.py` - Benchmarking script

**Documentation:**
- `CLUSTER_QUANTIZATION_STEPS.md` - Detailed cluster instructions
- `STATUS.md` - This file

### 🎯 Research Goal

Demonstrate that **activation-guided layer-wise quantization** can improve the quality-size tradeoff compared to uniform quantization, providing a practical method for optimizing LLM deployment in resource-constrained environments.
