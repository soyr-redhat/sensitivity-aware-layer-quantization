#!/usr/bin/env python3
"""Check all Mixtral layers for MoE"""
import torch
from transformers import AutoModelForCausalLM

print("Loading model structure (CPU only)...")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    cache_dir="/workspace/model_cache",
    device_map="cpu",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print(f"\nNumber of layers: {len(model.model.layers)}")

# Check each layer
for i, layer in enumerate(model.model.layers[:5]):  # Check first 5
    print(f"\nLayer {i}:")
    print(f"  Type: {type(layer).__name__}")
    # Check all attributes
    attrs = [a for a in dir(layer) if not a.startswith('_')]
    print(f"  Non-private attributes: {attrs[:15]}")

    # Specifically check for MoE-related
    if hasattr(layer, 'block_sparse_moe'):
        print(f"  ✓ Has block_sparse_moe!")
    if hasattr(layer, 'mlp'):
        print(f"  Has mlp: {type(layer.mlp).__name__}")
