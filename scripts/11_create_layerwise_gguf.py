#!/usr/bin/env python3
"""Create layer-wise quantized GGUF files using custom quantization.

This script takes a base F16 GGUF and creates variants with different
quantization levels per layer based on our activation analysis.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import gguf
from gguf import GGMLQuantizationType


# Map our quant names to GGML types
QUANT_TYPE_MAP = {
    'Q2_K': GGMLQuantizationType.Q2_K,
    'Q4_K_M': GGMLQuantizationType.Q4_K_M,
    'Q6_K': GGMLQuantizationType.Q6_K,
    'Q8_0': GGMLQuantizationType.Q8_0,
    'F16': GGMLQuantizationType.F16,
    'F32': GGMLQuantizationType.F32,
}


def quantize_tensor(data, from_type, to_type):
    """Quantize a tensor (placeholder - actual quantization is complex).

    For now, we'll use pre-quantized source files and copy layers.
    Real quantization requires the C++ ggml library.

    Args:
        data: Tensor data
        from_type: Source type
        to_type: Target type

    Returns:
        Quantized tensor data
    """
    # This is a placeholder - real quantization needs ggml C++ library
    # For our purposes, we'll copy from pre-quantized source files
    return data


def create_layerwise_gguf_from_sources(
    source_files: dict,
    layer_config: list,
    output_path: str,
    model_name: str = "mistral-7b-custom"
):
    """Create a custom GGUF by copying layers from different source files.

    Args:
        source_files: Dict mapping quant_type -> source GGUF path
        layer_config: List of quant types per layer (32 elements)
        output_path: Output GGUF path
        model_name: Model name for metadata
    """
    print(f"\n{'='*80}")
    print(f"CREATING CUSTOM LAYER-WISE GGUF")
    print(f"{'='*80}")
    print(f"Output: {output_path}")

    # Load all source readers
    readers = {}
    print("\nLoading source files...")
    for quant_type, path in source_files.items():
        if os.path.exists(path):
            print(f"  {quant_type}: {path}")
            readers[quant_type] = gguf.GGUFReader(path)
        else:
            print(f"  WARNING: {quant_type} file not found: {path}")

    if not readers:
        raise ValueError("No source files found!")

    # Analyze config
    print("\n" + "="*80)
    print("LAYER CONFIGURATION")
    print("="*80)

    quant_counts = {}
    for quant in layer_config:
        quant_counts[quant] = quant_counts.get(quant, 0) + 1

    print("\nQuantization distribution:")
    for quant_type, count in sorted(quant_counts.items()):
        print(f"  {quant_type}: {count} layers")

    # Show layer ranges
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

    # Create writer
    print("\n" + "="*80)
    print("ASSEMBLING GGUF FILE")
    print("="*80)

    writer = gguf.GGUFWriter(output_path, arch="llama")

    # Copy metadata from first available source
    first_reader = list(readers.values())[0]

    print("\nCopying metadata...")
    for field_name, field in first_reader.fields.items():
        if field_name.startswith('general.') or field_name.startswith('llama.') or field_name.startswith('tokenizer.'):
            try:
                # Get the field value
                parts = field.parts
                types = field.types

                # Add to writer (simplified - may need more complex handling)
                # writer.add_* methods depend on field type
                pass
            except Exception as e:
                print(f"  Warning: Could not copy field {field_name}: {e}")

    # Set model name
    writer.add_name(model_name)

    # Copy tensors layer by layer
    print("\nCopying tensors...")
    tensors_copied = 0

    for layer_idx, quant_type in enumerate(layer_config):
        if quant_type not in readers:
            print(f"  WARNING: No source for {quant_type}, skipping layer {layer_idx}")
            continue

        reader = readers[quant_type]
        layer_prefix = f'blk.{layer_idx}.'

        # Find all tensors for this layer
        layer_tensors = []
        for tensor in reader.tensors:
            if tensor.name.startswith(layer_prefix):
                layer_tensors.append(tensor)

        # Copy tensors
        for tensor in layer_tensors:
            writer.add_tensor(
                tensor.name,
                tensor.data,
                tensor.tensor_type
            )
            tensors_copied += 1

        if layer_idx % 8 == 0:
            print(f"  Processed layer {layer_idx}...")

    # Copy non-layer tensors (embeddings, output, etc.) from first source
    print("\nCopying non-layer tensors...")
    for tensor in first_reader.tensors:
        if not tensor.name.startswith('blk.'):
            writer.add_tensor(
                tensor.name,
                tensor.data,
                tensor.tensor_type
            )
            tensors_copied += 1

    # Write file
    print(f"\nWriting GGUF file...")
    print(f"  Total tensors: {tensors_copied}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    output_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"\n✓ Created: {output_path}")
    print(f"  Size: {output_size_gb:.2f} GB")

    return output_path


def download_source_gguf_files(output_dir: str = '/workspace/gguf_models'):
    """Download missing source GGUF files from HuggingFace.

    Args:
        output_dir: Directory to save files
    """
    print("\nChecking for source GGUF files...")

    # Files we need
    needed_files = {
        'Q2_K': 'mistral-7b-instruct-v0.2.Q2_K.gguf',
        'Q6_K': 'mistral-7b-instruct-v0.2.Q6_K.gguf',
        'Q8_0': 'mistral-7b-instruct-v0.2.Q8_0.gguf',
    }

    repo_id = 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'

    download_urls = {
        'Q2_K': f'https://huggingface.co/{repo_id}/resolve/main/{needed_files["Q2_K"]}',
        'Q6_K': f'https://huggingface.co/{repo_id}/resolve/main/{needed_files["Q6_K"]}',
        'Q8_0': f'https://huggingface.co/{repo_id}/resolve/main/{needed_files["Q8_0"]}',
    }

    for quant_type, filename in needed_files.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"  ✓ {quant_type}: {filename} ({size_gb:.2f} GB)")
        else:
            print(f"  ✗ {quant_type}: {filename} - MISSING")
            print(f"    Download from: {download_urls[quant_type]}")

    print("\nNOTE: You can download missing files with:")
    print("  cd /workspace/gguf_models")
    for quant_type, url in download_urls.items():
        filename = needed_files[quant_type]
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"  wget -O {filename} '{url}'")


def main():
    """Create layer-wise GGUF variants."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create layer-wise quantized GGUF files"
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
        help='Output directory for custom GGUF files'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        required=True,
        help='Name of the configuration (e.g., aggressive_mixed)'
    )
    parser.add_argument(
        '--layer-config',
        type=str,
        required=True,
        help='JSON array of quantization levels per layer'
    )
    args = parser.parse_args()

    # Parse layer config
    layer_config = json.loads(args.layer_config)

    if len(layer_config) != 32:
        raise ValueError(f"Expected 32 layers, got {len(layer_config)}")

    # Define source files
    source_files = {
        'F16': os.path.join(args.source_dir, 'mistral-7b-f16.gguf'),
        'Q2_K': os.path.join(args.source_dir, 'mistral-7b-instruct-v0.2.Q2_K.gguf'),
        'Q4_K_M': os.path.join(args.source_dir, 'mistral-7b-q4km.gguf'),
        'Q6_K': os.path.join(args.source_dir, 'mistral-7b-instruct-v0.2.Q6_K.gguf'),
        'Q8_0': os.path.join(args.source_dir, 'mistral-7b-instruct-v0.2.Q8_0.gguf'),
    }

    # Check which files exist
    download_source_gguf_files(args.source_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(
        args.output_dir,
        f'mistral-7b-{args.config_name}.gguf'
    )

    # Create custom GGUF
    create_layerwise_gguf_from_sources(
        source_files=source_files,
        layer_config=layer_config,
        output_path=output_path,
        model_name=f'mistral-7b-{args.config_name}'
    )

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
