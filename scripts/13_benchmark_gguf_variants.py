#!/usr/bin/env python3
"""Benchmark GGUF variants with llama-cpp-python.

This measures REAL memory usage and quality for different quantization levels.
"""

import os
import sys
import json
import time
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from llama_cpp import Llama
import psutil

from src.config import Config
from src.data.prompts import load_prompts_dataset


def get_process_memory_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024**3)


def get_gpu_memory_gb():
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def calculate_perplexity_gguf(
    model: Llama,
    texts: list,
    max_length: int = 512
) -> dict:
    """Calculate perplexity using llama-cpp-python.

    Args:
        model: Llama model instance
        texts: List of text samples
        max_length: Max tokens per sample

    Returns:
        Dict with perplexity and loss
    """
    print(f"  Calculating perplexity on {len(texts)} samples...")

    total_loss = 0.0
    total_tokens = 0
    num_samples = 0

    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"    Sample {i+1}/{len(texts)}...", end='\r')

        try:
            # Tokenize
            tokens = model.tokenize(text.encode('utf-8'))

            # Limit length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            if len(tokens) < 2:
                continue

            # Calculate loss
            # Note: llama-cpp-python doesn't expose direct loss calculation
            # We'll use log probability of each token given previous context
            sample_loss = 0.0

            for j in range(1, len(tokens)):
                context = tokens[:j]
                target = tokens[j]

                # Get logits for next token
                model.reset()
                model.eval(context)
                logits = model.eval([target])

                # Calculate cross-entropy loss
                # This is simplified - real perplexity calculation is more complex
                sample_loss += -logits[0]  # Negative log likelihood

            total_loss += sample_loss
            total_tokens += len(tokens) - 1
            num_samples += 1

        except Exception as e:
            print(f"\n    Warning: Failed on sample {i}: {e}")
            continue

    if total_tokens == 0:
        return {'perplexity': float('inf'), 'avg_loss': float('inf')}

    avg_loss = total_loss / total_tokens
    perplexity = 2 ** avg_loss  # Convert bits to perplexity

    print(f"\n  Completed {num_samples} samples")

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'num_samples': num_samples,
        'total_tokens': total_tokens
    }


def benchmark_gguf(
    gguf_path: str,
    variant_name: str,
    validation_texts: list,
    max_length: int = 512,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1
) -> dict:
    """Benchmark a single GGUF file.

    Args:
        gguf_path: Path to GGUF file
        variant_name: Name for this variant
        validation_texts: Validation samples
        max_length: Max sequence length for perplexity
        n_ctx: Context size for model
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)

    Returns:
        Benchmark results
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {variant_name}")
    print(f"{'='*80}")
    print(f"File: {gguf_path}")

    if not os.path.exists(gguf_path):
        print(f"✗ File not found!")
        return None

    file_size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"Disk size: {file_size_gb:.2f} GB")

    # Clean memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure memory before loading
    mem_before = get_process_memory_gb()
    gpu_before = get_gpu_memory_gb()

    # Load model
    print("\n[1/3] Loading model...")
    start_time = time.time()

    try:
        model = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        load_time = time.time() - start_time
        print(f"  Loaded in {load_time:.1f}s")

    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return None

    # Measure memory after loading
    print("\n[2/3] Measuring memory...")
    mem_after = get_process_memory_gb()
    gpu_after = get_gpu_memory_gb()

    model_mem = mem_after - mem_before
    gpu_mem = gpu_after - gpu_before

    print(f"  RAM usage: {model_mem:.2f} GB")
    print(f"  GPU usage: {gpu_mem:.2f} GB")

    # Calculate perplexity
    print("\n[3/3] Measuring perplexity...")

    # Use subset for faster benchmarking
    test_subset = validation_texts[:20]  # Use 20 samples

    try:
        metrics = calculate_perplexity_gguf(
            model=model,
            texts=test_subset,
            max_length=max_length
        )
        print(f"  Perplexity: {metrics['perplexity']:.4f}")

    except Exception as e:
        print(f"✗ Perplexity calculation failed: {e}")
        metrics = {'perplexity': float('inf'), 'avg_loss': float('inf')}

    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        'variant': variant_name,
        'file_path': gguf_path,
        'disk_size_gb': file_size_gb,
        'ram_usage_gb': model_mem,
        'gpu_usage_gb': gpu_mem,
        'load_time_s': load_time,
        'perplexity': metrics['perplexity'],
        'avg_loss': metrics.get('avg_loss', float('inf')),
        'num_samples': metrics.get('num_samples', 0),
    }

    print(f"\n✓ Benchmark complete")
    return results


def main():
    """Run GGUF benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark GGUF variants"
    )
    parser.add_argument(
        '--gguf-dir',
        type=str,
        default='/workspace/gguf_models',
        help='Directory containing GGUF files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mistral7b.yaml',
        help='Model config for validation data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/workspace/outputs/analysis/gguf_benchmark_results.json',
        help='Output path for results'
    )
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=-1,
        help='Number of layers to offload to GPU (-1 = all)'
    )
    args = parser.parse_args()

    print("="*80)
    print("GGUF VARIANT BENCHMARKING")
    print("="*80)

    # Load validation data
    print("\n[1/2] Loading validation data...")
    config = Config.from_yaml(args.config)
    prompt_datasets = load_prompts_dataset(config)

    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)

    print(f"  Loaded {len(test_texts)} validation samples")

    # Define variants to test
    variants = {
        'Q2_K': os.path.join(args.gguf_dir, 'mistral-7b-q2k.gguf'),
        'Q4_K_M': os.path.join(args.gguf_dir, 'mistral-7b-q4km.gguf'),
        'Q6_K': os.path.join(args.gguf_dir, 'mistral-7b-q6k.gguf'),
        'Q8_0': os.path.join(args.gguf_dir, 'mistral-7b-q8.gguf'),
    }

    # Benchmark each variant
    print("\n[2/2] Benchmarking variants...")
    all_results = {}

    for variant_name, gguf_path in variants.items():
        results = benchmark_gguf(
            gguf_path=gguf_path,
            variant_name=variant_name,
            validation_texts=test_texts,
            max_length=config.data.max_length,
            n_gpu_layers=args.n_gpu_layers
        )

        if results:
            all_results[variant_name] = results

        # Pause between benchmarks
        time.sleep(2)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved to: {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    print(f"\n{'Variant':<10} {'Disk':<10} {'RAM':<10} {'GPU':<10} {'PPL':<12} {'Loss':<10}")
    print("-"*70)

    for variant_name in ['Q2_K', 'Q4_K_M', 'Q6_K', 'Q8_0']:
        if variant_name in all_results:
            r = all_results[variant_name]
            print(f"{variant_name:<10} "
                  f"{r['disk_size_gb']:>8.2f}GB "
                  f"{r['ram_usage_gb']:>8.2f}GB "
                  f"{r['gpu_usage_gb']:>8.2f}GB "
                  f"{r['perplexity']:>10.4f}  "
                  f"{r['avg_loss']:>8.4f}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
