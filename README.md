# MoE Dynamic Quantization

Research project exploring dynamic per-prompt quantization for Mixture-of-Experts models.

## Core Idea

Use a lightweight "eagle" predictor model to forecast which experts will activate, then:
- Keep predicted-active experts in high precision (fp16/bf16)
- Quantize inactive experts aggressively (int4/int8)
- Reduce memory bandwidth while maintaining quality

## Project Structure

```
moe-dynamic-quant/
├── src/
│   ├── config.py           # Configuration management
│   ├── model/
│   │   ├── loader.py       # Model loading utilities
│   │   └── routing.py      # Router extraction and logging
│   ├── analysis/
│   │   ├── patterns.py     # Routing pattern analysis
│   │   └── visualize.py    # Visualization tools
│   └── data/
│       └── dataset.py      # Dataset handling
├── scripts/
│   ├── 01_analyze_routing.py    # Phase 1: Analyze routing patterns
│   ├── 02_train_predictor.py    # Phase 2: Train eagle model (TODO)
│   └── 03_benchmark.py          # Phase 3: Benchmark (TODO)
├── configs/
│   └── default.yaml
└── outputs/                     # Routing logs, plots, checkpoints
```

## Phases

### Phase 1: Routing Analysis (Current)
Analyze Mixtral-8x7B routing patterns to determine predictability:
- Log expert activations across diverse prompts
- Measure routing stability and entropy
- Identify patterns

### Phase 2: Eagle Predictor (Next)
Build lightweight model to predict expert routing:
- Train on routing logs from Phase 1
- Optimize for speed vs accuracy tradeoff

### Phase 3: Dynamic Quantization (Future)
Implement actual precision switching system

## Getting Started

```bash
# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run routing analysis
python scripts/01_analyze_routing.py --num-samples 100
```

### Alternative: Using pip
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/01_analyze_routing.py
```
