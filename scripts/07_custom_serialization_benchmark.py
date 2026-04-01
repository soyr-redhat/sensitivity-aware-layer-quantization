#!/usr/bin/env python3
"""Benchmark with custom serialization format.

This script creates layer-wise quantized models using custom serialization
that properly saves INT8 weights, then loads and benchmarks them to measure
actual memory savings.
"""

import os
import sys
import json
from pathlib import Path
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from src.data.prompts import load_prompts_dataset
from src.config import Config
from src.model.loader import load_tokenizer
from src.quantization import (
    QuantizationStrategy,
    create_manual_config,
    create_layer_configs_from_activation_profile,
    get_model_memory_footprint,
)
from src.quantization.serialization import (
    save_layerwise_quantized_checkpoint,
    load_quantized_checkpoint,
    estimate_serialized_size
)
from src.evaluation import calculate_perplexity


def load_activation_analysis(analysis_path: str) -> dict:
    """Load activation analysis results."""
    with open(analysis_path, 'r') as f:
        return json.load(f)


def benchmark_serialized_checkpoint(
    checkpoint_path: str,
    tokenizer,
    test_texts: list,
    baseline_ppl: float,
    baseline_memory: float,
    max_length: int = 512
) -> dict:
    """Benchmark a serialized checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        tokenizer: Tokenizer
        test_texts: Test samples
        baseline_ppl: Baseline perplexity
        baseline_memory: Baseline memory
        max_length: Max sequence length

    Returns:
        Benchmark results
    """
    strategy_name = os.path.basename(checkpoint_path)

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {strategy_name}")
    print(f"{'='*80}")

    # Clean memory before loading
    gc.collect()
    torch.cuda.empty_cache()

    # Check disk size
    print("\n[1/4] Checking disk size...")
    disk_info = estimate_serialized_size(checkpoint_path)
    print(f"  Disk size: {disk_info['weights_size_gb']:.2f} GB")

    # Load checkpoint
    print("\n[2/4] Loading checkpoint...")
    model = load_quantized_checkpoint(
        checkpoint_path=checkpoint_path,
        device_map="auto"
    )

    # Measure memory
    print("\n[3/4] Measuring memory...")
    memory = get_model_memory_footprint(model)
    gpu_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    print(f"  Model memory: {memory['total_memory_gb']:.2f} GB")
    print(f"  GPU allocated: {gpu_allocated:.2f} GB")

    # Measure perplexity
    print("\n[4/4] Measuring perplexity...")
    metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    print(f"  Perplexity: {metrics['perplexity']:.4f}")

    # Calculate comparisons
    ppl_delta = ((metrics['perplexity'] / baseline_ppl) - 1) * 100
    mem_savings = ((baseline_memory - memory['total_memory_gb']) / baseline_memory) * 100
    compression = baseline_memory / memory['total_memory_gb']

    print(f"\n  vs Baseline:")
    print(f"    Perplexity: {ppl_delta:+.2f}%")
    print(f"    Memory: {mem_savings:+.1f}%")
    print(f"    Compression: {compression:.2f}x")

    results = {
        'strategy': strategy_name,
        'checkpoint_path': checkpoint_path,
        'disk_size_gb': disk_info['weights_size_gb'],
        'model_memory_gb': memory['total_memory_gb'],
        'gpu_allocated_gb': gpu_allocated,
        'perplexity': metrics['perplexity'],
        'ppl_delta_pct': ppl_delta,
        'memory_savings_pct': mem_savings,
        'compression_ratio': compression,
    }

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    """Main benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark with custom serialization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mistral7b.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--activation-analysis',
        type=str,
        default='/workspace/outputs/analysis/activation_analysis.json',
        help='Path to activation analysis'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/workspace/outputs/serialized_checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--skip-creation',
        action='store_true',
        help='Skip checkpoint creation (use existing)'
    )
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    print("="*80)
    print("CUSTOM SERIALIZATION BENCHMARK")
    print("="*80)

    # Load test data
    print("\n[1/6] Loading test data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)
    print(f"  Loaded {len(test_texts)} test samples")

    # Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer = load_tokenizer(config.model)

    # Load activation analysis
    print("\n[3/6] Loading activation analysis...")
    activation_analysis = load_activation_analysis(args.activation_analysis)
    num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])
    print(f"  Found {num_layers} layers")

    # Use baseline from previous benchmark
    baseline_ppl = 10.7742
    baseline_memory = 13.50
    print(f"\n  Baseline: {baseline_ppl:.4f} PPL @ {baseline_memory:.2f} GB")

    # Create layer-wise configs
    strategies = {
        'manual_gradient': create_manual_config(
            num_layers=num_layers,
            early_quant=QuantizationStrategy.INT4,
            mid_quant=QuantizationStrategy.INT8,
            late_quant=QuantizationStrategy.BF16,
            early_cutoff=10,
            late_cutoff=21
        ),
        'activation_based': create_layer_configs_from_activation_profile(
            activation_analysis=activation_analysis,
            num_layers=num_layers,
            aggressive_threshold=0.3,
            conservative_threshold=0.7
        ),
    }

    checkpoint_paths = {}

    # Create checkpoints if needed
    if not args.skip_creation:
        print("\n[4/6] Creating serialized checkpoints...")

        for strategy_name, layer_configs in strategies.items():
            checkpoint_path = save_layerwise_quantized_checkpoint(
                base_model_name=config.model.name,
                cache_dir=config.model.cache_dir,
                configs=layer_configs,
                output_dir=args.checkpoint_dir,
                strategy_name=strategy_name,
                device_map=config.model.device_map
            )
            checkpoint_paths[strategy_name] = checkpoint_path
    else:
        print("\n[4/6] Using existing checkpoints...")
        for strategy_name in strategies.keys():
            checkpoint_path = os.path.join(args.checkpoint_dir, strategy_name)
            if os.path.exists(checkpoint_path):
                checkpoint_paths[strategy_name] = checkpoint_path
                print(f"  Found: {checkpoint_path}")
            else:
                print(f"  Warning: {checkpoint_path} not found")

    # Benchmark each checkpoint
    print("\n[5/6] Benchmarking checkpoints...")

    all_results = {
        'baseline_bf16': {
            'strategy': 'baseline_bf16',
            'perplexity': baseline_ppl,
            'model_memory_gb': baseline_memory,
            'ppl_delta_pct': 0.0,
            'memory_savings_pct': 0.0,
            'compression_ratio': 1.0,
        }
    }

    for strategy_name, checkpoint_path in checkpoint_paths.items():
        results = benchmark_serialized_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            test_texts=test_texts,
            baseline_ppl=baseline_ppl,
            baseline_memory=baseline_memory,
            max_length=config.data.max_length
        )
        all_results[strategy_name] = results

    # Save results
    print("\n[6/6] Saving results...")
    output_dir = config.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'serialized_benchmark_results.json')
    with open(results_file, 'w') as f:
        serializable = {
            k: {k2: v2 for k2, v2 in v.items() if isinstance(v2, (int, float, str, bool, type(None)))}
            for k, v in all_results.items()
        }
        json.dump(serializable, f, indent=2)

    print(f"Saved results to: {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("SERIALIZED CHECKPOINT RESULTS")
    print("="*80)

    print(f"\n{'Strategy':<25} {'PPL':<10} {'PPL Δ%':<10} {'Memory':<12} {'Disk':<12} {'Savings':<10} {'Ratio':<8}")
    print("-"*100)

    for name, data in all_results.items():
        ppl = data['perplexity']
        ppl_delta = data.get('ppl_delta_pct', 0.0)
        mem = data['model_memory_gb']
        disk = data.get('disk_size_gb', mem)
        savings = data.get('memory_savings_pct', 0.0)
        ratio = data.get('compression_ratio', 1.0)

        print(f"{name:<25} {ppl:>8.4f}  {ppl_delta:>8.2f}%  "
              f"{mem:>10.2f}GB  {disk:>10.2f}GB  {savings:>8.1f}%  {ratio:>6.2f}x")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
