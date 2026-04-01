#!/usr/bin/env python3
"""Generate tensor-type files for llama-quantize per-layer quantization.

This creates the mapping files that tell llama-quantize which tensors
to quantize to which levels based on our activation analysis.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Tensor names for Mistral architecture (per layer)
LAYER_TENSOR_NAMES = [
    'attn_q.weight',
    'attn_k.weight',
    'attn_v.weight',
    'attn_output.weight',
    'ffn_gate.weight',
    'ffn_up.weight',
    'ffn_down.weight',
    'attn_norm.weight',
    'ffn_norm.weight',
]


# Predefined mixed-precision configurations
# Note: llama-quantize expects lowercase type names in tensor-type files
CONFIGS = {
    'aggressive_mixed': {
        'description': 'Max compression: Q2_K early, Q4_K_M mid, Q6_K/Q8_0 late',
        'layers': (
            ['q2_k'] * 19 +      # Layers 0-18: Low sensitivity
            ['q4_k_m'] * 10 +    # Layers 19-28: Medium sensitivity
            ['q6_k'] * 2 +       # Layers 29-30: High sensitivity
            ['q8_0'] * 1         # Layer 31: Critical
        )
    },
    'balanced_mixed': {
        'description': 'Balanced: Q4_K_M early/mid, Q6_K/Q8_0 late',
        'layers': (
            ['q4_k_m'] * 21 +    # Layers 0-20
            ['q6_k'] * 8 +       # Layers 21-28
            ['q8_0'] * 3         # Layers 29-31
        )
    },
    'conservative_mixed': {
        'description': 'Prioritize quality: Q4_K_M early, Q6_K mid, Q8_0 late',
        'layers': (
            ['q4_k_m'] * 16 +    # Layers 0-15
            ['q6_k'] * 10 +      # Layers 16-25
            ['q8_0'] * 6         # Layers 26-31
        )
    },
}


def generate_tensor_type_file(
    layer_config: list,
    output_path: str,
    num_layers: int = 32
):
    """Generate a tensor-type file for llama-quantize.

    Args:
        layer_config: List of quant types per layer (32 elements)
        output_path: Output path for tensor-type file
        num_layers: Number of layers in model
    """
    print(f"\nGenerating: {output_path}")

    # Analyze config
    quant_counts = {}
    for quant in layer_config:
        quant_counts[quant] = quant_counts.get(quant, 0) + 1

    print(f"  Config: {', '.join(f'{v}×{k}' for k, v in sorted(quant_counts.items()))}")

    with open(output_path, 'w') as f:
        # Write tensor mappings for each layer
        for layer_idx in range(num_layers):
            if layer_idx >= len(layer_config):
                print(f"  WARNING: No config for layer {layer_idx}, using Q8_0")
                quant_type = 'Q8_0'
            else:
                quant_type = layer_config[layer_idx]

            # Map each tensor in this layer
            for tensor_name in LAYER_TENSOR_NAMES:
                # Skip norm weights - keep as F32 for stability
                if 'norm' in tensor_name:
                    continue

                full_name = f"blk.{layer_idx}.{tensor_name}"
                f.write(f"{full_name}={quant_type}\n")

    print(f"  ✓ Created with {sum(1 for _ in open(output_path) if '=' in _)} tensor mappings")


def main():
    """Generate tensor-type files for all configs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate tensor-type files for mixed-precision GGUF"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/gguf_models/tensor_types',
        help='Output directory for tensor-type files'
    )
    args = parser.parse_args()

    print("="*80)
    print("GENERATING TENSOR-TYPE FILES")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate file for each config
    for config_name, config_spec in CONFIGS.items():
        print(f"\n{config_name}:")
        print(f"  {config_spec['description']}")

        output_path = os.path.join(
            args.output_dir,
            f'{config_name}.txt'
        )

        generate_tensor_type_file(
            layer_config=config_spec['layers'],
            output_path=output_path
        )

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nGenerated files in: {args.output_dir}")
    print("\nNext step: Use with llama-quantize:")
    print(f"  llama-quantize --tensor-type-file <file.txt> input.gguf output.gguf Q4_K_M")


if __name__ == '__main__':
    main()
