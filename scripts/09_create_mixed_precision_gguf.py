#!/usr/bin/env python3
"""Create mixed-precision GGUF models based on activation profiling.

This script creates custom GGUF models where different layers have different
quantization levels based on our activation analysis.

Novel approach:
- Standard GGUF quantization is uniform across all layers
- We use activation profiling to determine which layers need higher precision
- Combine layers from different GGUF files (Q2_K, Q4_K_M, Q8_0)
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gguf
from src.quantization import LayerQuantConfig, QuantizationStrategy


def map_strategy_to_gguf_quant(strategy: QuantizationStrategy) -> str:
    """Map our quantization strategy to GGUF quant level."""
    mapping = {
        QuantizationStrategy.BF16: 'Q8_0',  # Closest to full precision
        QuantizationStrategy.FP16: 'Q8_0',
        QuantizationStrategy.INT8: 'Q4_K_M',  # Medium precision
        QuantizationStrategy.INT4: 'Q2_K',   # Low precision
        QuantizationStrategy.NONE: 'Q8_0',
    }
    return mapping.get(strategy, 'Q4_K_M')


def create_layer_to_quant_map(
    layer_configs: List[LayerQuantConfig]
) -> Dict[int, str]:
    """Create a mapping of layer index to GGUF quantization type.

    Args:
        layer_configs: Our layer quantization configs

    Returns:
        Dict mapping layer_idx -> GGUF quant type (Q8_0, Q4_K_M, Q2_K)
    """
    layer_map = {}

    for config in layer_configs:
        # Use the more conservative (higher precision) of attention and MLP
        strategies = [config.attention_quant, config.mlp_quant]

        # Map to GGUF quant type
        quant_types = [map_strategy_to_gguf_quant(s) for s in strategies]

        # Choose the higher precision one
        # Q8_0 > Q4_K_M > Q2_K
        quant_order = ['Q2_K', 'Q4_K_M', 'Q8_0']
        chosen_quant = max(quant_types, key=lambda q: quant_order.index(q))

        layer_map[config.layer_idx] = chosen_quant

    return layer_map


def extract_layer_tensors(
    reader: gguf.GGUFReader,
    layer_idx: int
) -> Dict[str, any]:
    """Extract all tensors for a specific layer.

    Args:
        reader: GGUF reader
        layer_idx: Layer index

    Returns:
        Dict of tensor name -> tensor data
    """
    layer_tensors = {}
    layer_prefix = f'blk.{layer_idx}.'

    for tensor in reader.tensors:
        if tensor.name.startswith(layer_prefix):
            layer_tensors[tensor.name] = tensor

    return layer_tensors


def analyze_gguf_files(gguf_paths: Dict[str, str]) -> dict:
    """Analyze available GGUF files.

    Args:
        gguf_paths: Dict of quant_type -> file path

    Returns:
        Analysis results
    """
    print("\n" + "="*80)
    print("ANALYZING GGUF FILES")
    print("="*80)

    analysis = {}

    for quant_type, path in gguf_paths.items():
        if not os.path.exists(path):
            print(f"\n⚠️  {quant_type}: File not found: {path}")
            continue

        print(f"\n=== {quant_type} ===")
        reader = gguf.GGUFReader(path)

        # Count layers
        layer_nums = set()
        for tensor in reader.tensors:
            if 'blk.' in tensor.name:
                layer_num = tensor.name.split('blk.')[1].split('.')[0]
                if layer_num.isdigit():
                    layer_nums.add(int(layer_num))

        num_layers = len(layer_nums)
        file_size_gb = os.path.getsize(path) / (1024**3)

        print(f"  Layers: {num_layers}")
        print(f"  Tensors: {len(reader.tensors)}")
        print(f"  File size: {file_size_gb:.2f} GB")

        analysis[quant_type] = {
            'path': path,
            'num_layers': num_layers,
            'num_tensors': len(reader.tensors),
            'size_gb': file_size_gb,
            'reader': reader
        }

    return analysis


def create_mixed_precision_gguf(
    source_gguf_paths: Dict[str, str],
    layer_map: Dict[int, str],
    output_path: str,
    metadata: Dict = None
):
    """Create a mixed-precision GGUF by combining layers from different files.

    Args:
        source_gguf_paths: Dict of quant_type -> source GGUF path
        layer_map: Dict of layer_idx -> desired quant_type
        output_path: Path to save output GGUF
        metadata: Optional metadata to include
    """
    print("\n" + "="*80)
    print("CREATING MIXED-PRECISION GGUF")
    print("="*80)

    # Analyze source files
    analysis = analyze_gguf_files(source_gguf_paths)

    # Count layers per quantization type
    quant_counts = {}
    for layer_idx, quant_type in sorted(layer_map.items()):
        quant_counts[quant_type] = quant_counts.get(quant_type, 0) + 1

    print("\n=== TARGET CONFIGURATION ===")
    for quant_type, count in sorted(quant_counts.items()):
        print(f"  {quant_type}: {count} layers")

    # Show layer distribution
    print("\n=== LAYER DISTRIBUTION ===")
    current_quant = None
    start_layer = None

    for layer_idx in sorted(layer_map.keys()):
        quant_type = layer_map[layer_idx]

        if quant_type != current_quant:
            if current_quant is not None:
                print(f"  Layers {start_layer:2d}-{layer_idx-1:2d}: {current_quant}")
            current_quant = quant_type
            start_layer = layer_idx

    # Print last range
    if current_quant is not None:
        print(f"  Layers {start_layer:2d}-{max(layer_map.keys()):2d}: {current_quant}")

    print("\n⚠️  Note: Actual GGUF merging requires binary manipulation")
    print("This is a research prototype - showing the strategy")
    print("\nNext steps:")
    print("1. Use llama.cpp tools to requantize specific layers")
    print("2. Or implement full GGUF writer to merge layers")

    # Save the layer map for reference
    layer_map_path = output_path.replace('.gguf', '_layer_map.json')
    with open(layer_map_path, 'w') as f:
        json.dump({
            'layer_map': {str(k): v for k, v in layer_map.items()},
            'quant_counts': quant_counts,
            'metadata': metadata or {}
        }, f, indent=2)

    print(f"\nSaved layer map to: {layer_map_path}")


def main():
    """Main mixed-precision GGUF creation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create mixed-precision GGUF from activation analysis"
    )
    parser.add_argument(
        '--activation-analysis',
        type=str,
        required=True,
        help='Path to activation analysis JSON'
    )
    parser.add_argument(
        '--q2k-gguf',
        type=str,
        help='Path to Q2_K GGUF (for low-importance layers)'
    )
    parser.add_argument(
        '--q4km-gguf',
        type=str,
        help='Path to Q4_K_M GGUF (for medium-importance layers)'
    )
    parser.add_argument(
        '--q8-gguf',
        type=str,
        help='Path to Q8_0 GGUF (for high-importance layers)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/workspace/outputs/mixed_precision.gguf',
        help='Output path for mixed GGUF'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['activation_based', 'manual_gradient'],
        default='activation_based',
        help='Quantization strategy to use'
    )
    args = parser.parse_args()

    # Load activation analysis
    print("Loading activation analysis...")
    with open(args.activation_analysis, 'r') as f:
        activation_analysis = json.load(f)

    num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])
    print(f"Found {num_layers} layers")

    # Create layer configs based on strategy
    from src.quantization import (
        create_layer_configs_from_activation_profile,
        create_manual_config
    )

    if args.strategy == 'activation_based':
        layer_configs = create_layer_configs_from_activation_profile(
            activation_analysis=activation_analysis,
            num_layers=num_layers,
            aggressive_threshold=0.3,
            conservative_threshold=0.7
        )
    else:  # manual_gradient
        layer_configs = create_manual_config(
            num_layers=num_layers,
            early_quant=QuantizationStrategy.INT4,
            mid_quant=QuantizationStrategy.INT8,
            late_quant=QuantizationStrategy.BF16,
            early_cutoff=10,
            late_cutoff=21
        )

    # Create layer -> quant type mapping
    layer_map = create_layer_to_quant_map(layer_configs)

    # Prepare source GGUF paths
    source_ggufs = {}
    if args.q2k_gguf:
        source_ggufs['Q2_K'] = args.q2k_gguf
    if args.q4km_gguf:
        source_ggufs['Q4_K_M'] = args.q4km_gguf
    if args.q8_gguf:
        source_ggufs['Q8_0'] = args.q8_gguf

    if not source_ggufs:
        print("Error: No source GGUF files provided")
        sys.exit(1)

    # Create mixed-precision GGUF
    create_mixed_precision_gguf(
        source_gguf_paths=source_ggufs,
        layer_map=layer_map,
        output_path=args.output,
        metadata={
            'strategy': args.strategy,
            'source': 'activation_profiling'
        }
    )


if __name__ == '__main__':
    main()
