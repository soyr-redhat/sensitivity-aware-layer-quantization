#!/usr/bin/env python3
"""Assemble custom layer-wise GGUF by copying layers from different sources.

Since real quantization requires C++ GGML, we "frankenstein" by copying
already-quantized layers from different source files based on our
activation analysis.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gguf


def analyze_layer_config(layer_config):
    """Analyze and print layer configuration."""
    print("\n" + "="*80)
    print("LAYER CONFIGURATION ANALYSIS")
    print("="*80)

    # Count by quant type
    quant_counts = {}
    for quant in layer_config:
        quant_counts[quant] = quant_counts.get(quant, 0) + 1

    print("\nQuantization distribution:")
    for quant_type in sorted(quant_counts.keys()):
        count = quant_counts[quant_type]
        pct = (count / len(layer_config)) * 100
        print(f"  {quant_type}: {count} layers ({pct:.1f}%)")

    # Show ranges
    print("\nLayer ranges:")
    current_type = None
    start_layer = 0

    for layer_idx, quant_type in enumerate(layer_config):
        if quant_type != current_type:
            if current_type is not None:
                print(f"  Layers {start_layer:2d}-{layer_idx-1:2d}: {current_type}")
            current_type = quant_type
            start_layer = layer_idx

    if current_type is not None:
        print(f"  Layers {start_layer:2d}-{len(layer_config)-1:2d}: {current_type}")

    return quant_counts


def assemble_custom_gguf(
    source_files: dict,
    layer_config: list,
    output_path: str
):
    """Assemble custom GGUF by copying tensors from sources.

    Args:
        source_files: Dict of quant_type -> gguf_path
        layer_config: List of 32 quant types (one per layer)
        output_path: Output path
    """
    print("\n" + "="*80)
    print("ASSEMBLING CUSTOM GGUF")
    print("="*80)
    print(f"Output: {output_path}")

    # Analyze config
    quant_counts = analyze_layer_config(layer_config)

    # Load source readers
    print("\n" + "="*80)
    print("LOADING SOURCE FILES")
    print("="*80)

    readers = {}
    for quant_type, path in source_files.items():
        if not os.path.exists(path):
            print(f"  ✗ {quant_type}: NOT FOUND - {path}")
            continue

        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  ✓ {quant_type}: {os.path.basename(path)} ({size_gb:.2f} GB)")
        readers[quant_type] = gguf.GGUFReader(path)

    # Verify we have all needed sources
    needed_types = set(layer_config)
    missing_types = needed_types - set(readers.keys())

    if missing_types:
        print(f"\n⚠️  WARNING: Missing source files for: {missing_types}")
        print("Cannot proceed without all source files.")
        return None

    # Create writer
    print("\n" + "="*80)
    print("COPYING TENSORS")
    print("="*80)

    writer = gguf.GGUFWriter(output_path, arch="llama")

    # Copy metadata from first source
    first_reader = list(readers.values())[0]

    print("\nCopying metadata from first source...")

    # Copy KV metadata
    for field_name, field in first_reader.fields.items():
        try:
            # Get field value from parts
            value = field.parts[field.data_offset] if field.parts else None

            # Add to writer based on type
            # Note: This is simplified, may need more robust handling
            if value is not None:
                writer.add_key(field_name)
                writer.add_val(value, field.types[0] if field.types else gguf.GGUFValueType.STRING)
        except Exception as e:
            # Skip problematic fields
            pass

    # Copy tensors
    print("\nCopying layer tensors...")
    tensors_copied = 0

    # Non-layer tensors (embeddings, output, etc.) - copy from first source
    print("  Non-layer tensors (embeddings, norms, etc.)...")
    for tensor in first_reader.tensors:
        if not tensor.name.startswith('blk.'):
            writer.add_tensor(tensor.name, tensor.data, tensor.tensor_type)
            tensors_copied += 1

    # Layer tensors - copy from appropriate source based on config
    print("  Layer-specific tensors...")
    for layer_idx, quant_type in enumerate(layer_config):
        reader = readers[quant_type]
        layer_prefix = f'blk.{layer_idx}.'

        # Find and copy all tensors for this layer
        layer_tensor_count = 0
        for tensor in reader.tensors:
            if tensor.name.startswith(layer_prefix):
                writer.add_tensor(tensor.name, tensor.data, tensor.tensor_type)
                tensors_copied += 1
                layer_tensor_count += 1

        if layer_idx % 8 == 0 or layer_idx == len(layer_config) - 1:
            print(f"    Layer {layer_idx}: {layer_tensor_count} tensors from {quant_type}")

    # Write the file
    print(f"\n✍️  Writing GGUF file...")
    print(f"  Total tensors to write: {tensors_copied}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()

    # Check output
    if os.path.exists(output_path):
        output_size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"\n✓ SUCCESS!")
        print(f"  Created: {output_path}")
        print(f"  Size: {output_size_gb:.2f} GB")
        return output_path
    else:
        print(f"\n✗ FAILED - Output file not created")
        return None


# Predefined configurations based on activation analysis
CONFIGS = {
    'aggressive_mixed': {
        'description': 'Max compression: Q2_K early, Q4_K_M mid, Q6_K/Q8_0 late',
        'layers': (
            ['Q2_K'] * 19 +      # Layers 0-18: Low sensitivity
            ['Q4_K_M'] * 10 +    # Layers 19-28: Medium sensitivity
            ['Q6_K'] * 2 +       # Layers 29-30: High sensitivity
            ['Q8_0'] * 1         # Layer 31: Critical
        )
    },
    'balanced_mixed': {
        'description': 'Balanced: Q4_K_M early/mid, Q6_K/Q8_0 late',
        'layers': (
            ['Q4_K_M'] * 21 +    # Layers 0-20
            ['Q6_K'] * 8 +       # Layers 21-28
            ['Q8_0'] * 3         # Layers 29-31
        )
    },
    'conservative_mixed': {
        'description': 'Prioritize quality: Q4_K_M early, Q6_K mid, Q8_0 late',
        'layers': (
            ['Q4_K_M'] * 16 +    # Layers 0-15
            ['Q6_K'] * 10 +      # Layers 16-25
            ['Q8_0'] * 6         # Layers 26-31
        )
    },
    'gradient': {
        'description': 'Linear gradient: Q2_K → Q4_K_M → Q6_K → Q8_0',
        'layers': (
            ['Q2_K'] * 8 +       # Layers 0-7
            ['Q4_K_M'] * 8 +     # Layers 8-15
            ['Q6_K'] * 8 +       # Layers 16-23
            ['Q8_0'] * 8         # Layers 24-31
        )
    },
    'early_aggressive': {
        'description': 'Compress early, preserve late: Q2_K early, Q8_0 late',
        'layers': (
            ['Q2_K'] * 21 +      # Layers 0-20
            ['Q8_0'] * 11        # Layers 21-31
        )
    },
}


def main():
    """Create custom GGUF variants."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble custom layer-wise GGUF files"
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='/workspace/gguf_models',
        help='Directory containing source GGUF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/gguf_models/custom',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=list(CONFIGS.keys()) + ['all'],
        default='all',
        help='Which config to create (or "all")'
    )
    args = parser.parse_args()

    # Define source files
    source_files = {
        'Q2_K': os.path.join(args.source_dir, 'mistral-7b-q2k.gguf'),
        'Q4_K_M': os.path.join(args.source_dir, 'mistral-7b-q4km.gguf'),
        'Q6_K': os.path.join(args.source_dir, 'mistral-7b-q6k.gguf'),
        'Q8_0': os.path.join(args.source_dir, 'mistral-7b-q8.gguf'),
    }

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which configs to create
    if args.config == 'all':
        configs_to_create = CONFIGS.items()
    else:
        configs_to_create = [(args.config, CONFIGS[args.config])]

    # Create each config
    results = {}
    for config_name, config_spec in configs_to_create:
        print("\n" + "="*80)
        print(f"CONFIG: {config_name}")
        print(f"  {config_spec['description']}")
        print("="*80)

        output_path = os.path.join(
            args.output_dir,
            f'mistral-7b-{config_name}.gguf'
        )

        result_path = assemble_custom_gguf(
            source_files=source_files,
            layer_config=config_spec['layers'],
            output_path=output_path
        )

        results[config_name] = result_path

    # Summary
    print("\n" + "="*80)
    print("ASSEMBLY COMPLETE")
    print("="*80)

    print("\nCreated custom GGUF files:")
    for config_name, path in results.items():
        if path:
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"  ✓ {config_name}: {size_gb:.2f} GB")
        else:
            print(f"  ✗ {config_name}: FAILED")

    print("\nNext step: Benchmark these variants with llama.cpp!")


if __name__ == '__main__':
    main()
