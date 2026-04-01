
## Optimizer Testing

Tested end-to-end on Qwen2.5-3B (36 layers):

**Test Command:**
```bash
python scripts/02_optimize_layer_config.py \
  --base-model qwen2.5-3b-q8.gguf \
  --test-data test_perplexity_large.txt \
  --num-layers 36 \
  --max-evals 5 \
  --target-size 2.5
```

**Results:**
- ✅ Generated layer configs
- ✅ Quantized models with llama-quantize  
- ✅ Measured perplexity successfully
- ✅ Found optimal config (36×Q2_K, PPL=3.69, 2.28GB)

**Fixes applied:**
- Added `--allow-requantize` for pre-quantized base models
- Fixed perplexity extraction from stderr
- Optimized for faster testing

The optimizer works correctly end-to-end!
