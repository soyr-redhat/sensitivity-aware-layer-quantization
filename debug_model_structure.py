#!/usr/bin/env python3
"""Quick script to inspect Mixtral model structure"""

import torch
from transformers import AutoModelForCausalLM

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    cache_dir="/workspace/model_cache",
    device_map="cpu",  # Just load structure, don't need GPU
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print(f"\nModel type: {type(model)}")
print(f"Has 'model' attr: {hasattr(model, 'model')}")

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layer = model.model.layers[0]
    print(f"\nFirst layer type: {type(layer)}")
    print(f"\nFirst layer attributes:")
    for attr in dir(layer):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Check for MoE-related attributes
    print(f"\nChecking for MoE:")
    for name in ['block_sparse_moe', 'moe', 'mlp', 'feed_forward']:
        if hasattr(layer, name):
            obj = getattr(layer, name)
            print(f"  Found '{name}': {type(obj)}")
            print(f"    Attributes: {[a for a in dir(obj) if not a.startswith('_')][:10]}")
