#!/usr/bin/env python3
"""Explore GGUF file structure to understand layer-wise quantization.

This script examines GGUF files to understand:
1. How layers are stored
2. What quantization info is available per layer
3. Whether we can selectively extract/merge layers
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gguf


def explore_gguf(gguf_path: str):
    """Explore a GGUF file structure."""
    print(f"Exploring: {gguf_path}")
    print("=" * 80)

    reader = gguf.GGUFReader(gguf_path)

    # Print metadata
    print("\n=== METADATA ===")
    print(f"Alignment: {reader.alignment}")
    print(f"Byte order: {reader.byte_order}")
    print(f"Tensor count: {len(reader.tensors)}")
    print(f"Field count: {len(reader.fields)}")

    # Print fields
    print("\n=== FIELDS (first 10) ===")
    for i, (name, field) in enumerate(reader.fields.items()):
        if i >= 10:
            break
        print(f"{name}: {type(field)}")

    # Print tensor info
    print("\n=== TENSORS (first 20) ===")
    print(f"{'Name':<60} {'Shape':<20} {'Type':<15} {'Size (MB)':<10}")
    print("-" * 110)

    for i, tensor in enumerate(reader.tensors[:20]):
        size_mb = tensor.n_bytes / (1024**2)
        print(f"{tensor.name:<60} {str(tensor.shape):<20} {str(tensor.tensor_type):<15} {size_mb:>8.2f}")

    if len(reader.tensors) > 20:
        print(f"\n... and {len(reader.tensors) - 20} more tensors")

    # Analyze layer structure
    print("\n=== LAYER ANALYSIS ===")
    layer_tensors = {}
    for tensor in reader.tensors:
        # Extract layer number from tensor name
        if 'blk.' in tensor.name:
            layer_num = tensor.name.split('blk.')[1].split('.')[0]
            if layer_num.isdigit():
                layer_num = int(layer_num)
                if layer_num not in layer_tensors:
                    layer_tensors[layer_num] = []
                layer_tensors[layer_num].append({
                    'name': tensor.name,
                    'type': tensor.tensor_type,
                    'size_mb': tensor.n_bytes / (1024**2)
                })

    print(f"Found {len(layer_tensors)} layers")

    # Show layer 0 structure as example
    if 0 in layer_tensors:
        print("\nLayer 0 tensors (example):")
        total_size = 0
        for t in layer_tensors[0]:
            print(f"  {t['name']:<60} {str(t['type']):<15} {t['size_mb']:>8.2f} MB")
            total_size += t['size_mb']
        print(f"  Total layer 0 size: {total_size:.2f} MB")

    # Calculate size per layer
    print("\n=== SIZE PER LAYER ===")
    print(f"{'Layer':<10} {'Size (MB)':<12} {'Num Tensors':<12}")
    print("-" * 40)
    for layer_num in sorted(layer_tensors.keys())[:5]:
        total_size = sum(t['size_mb'] for t in layer_tensors[layer_num])
        num_tensors = len(layer_tensors[layer_num])
        print(f"{layer_num:<10} {total_size:>10.2f}  {num_tensors:>10}")

    if len(layer_tensors) > 5:
        print("...")
        # Show last 2 layers
        for layer_num in sorted(layer_tensors.keys())[-2:]:
            total_size = sum(t['size_mb'] for t in layer_tensors[layer_num])
            num_tensors = len(layer_tensors[layer_num])
            print(f"{layer_num:<10} {total_size:>10.2f}  {num_tensors:>10}")

    # Check quantization types
    print("\n=== QUANTIZATION TYPES ===")
    quant_types = {}
    for tensor in reader.tensors:
        tensor_type = str(tensor.tensor_type)
        quant_types[tensor_type] = quant_types.get(tensor_type, 0) + 1

    for qtype, count in sorted(quant_types.items()):
        print(f"{qtype:<30} {count:>6} tensors")


def main():
    """Main exploration."""
    import argparse

    parser = argparse.ArgumentParser(description="Explore GGUF file structure")
    parser.add_argument('gguf_file', type=str, help='Path to GGUF file')
    args = parser.parse_args()

    if not os.path.exists(args.gguf_file):
        print(f"Error: File not found: {args.gguf_file}")
        sys.exit(1)

    explore_gguf(args.gguf_file)


if __name__ == '__main__':
    main()
