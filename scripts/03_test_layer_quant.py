#!/usr/bin/env python3
"""Test layer-wise quantization strategies.

This script evaluates different layer-wise quantization configurations
based on activation profiling results, measuring the trade-off between
memory savings and perplexity degradation.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from src.model.loader import load_model_and_tokenizer
from src.data.prompts import load_prompts_dataset
from src.config import Config
from src.quantization import (
    QuantizationStrategy,
    LayerQuantConfig,
    create_uniform_config,
    create_manual_config,
    create_layer_configs_from_activation_profile,
    apply_layer_wise_quantization,
    get_model_memory_footprint,
    estimate_quantized_memory
)
from src.evaluation import calculate_perplexity


def load_activation_analysis(analysis_path: str) -> dict:
    """Load activation analysis results."""
    with open(analysis_path, 'r') as f:
        return json.load(f)


def create_quantization_strategies(
    activation_analysis: dict,
    num_layers: int
) -> dict:
    """Create different quantization strategies to test.

    Args:
        activation_analysis: Activation profiling results
        num_layers: Number of model layers

    Returns:
        Dictionary of strategy_name -> LayerQuantConfig list
    """
    strategies = {}

    # Strategy 1: Uniform INT8 (baseline quantized)
    strategies['uniform_int8'] = create_uniform_config(
        num_layers=num_layers,
        strategy=QuantizationStrategy.INT8
    )

    # Strategy 2: Uniform INT4 (aggressive baseline)
    strategies['uniform_int4'] = create_uniform_config(
        num_layers=num_layers,
        strategy=QuantizationStrategy.INT4
    )

    # Strategy 3: Manual layer-wise (based on our observation)
    # Early layers (0-10): INT4
    # Mid layers (11-20): INT8
    # Late layers (21-31): BF16
    strategies['manual_gradient'] = create_manual_config(
        num_layers=num_layers,
        early_quant=QuantizationStrategy.INT4,
        mid_quant=QuantizationStrategy.INT8,
        late_quant=QuantizationStrategy.BF16,
        early_cutoff=10,
        late_cutoff=21
    )

    # Strategy 4: Conservative layer-wise
    # Early layers (0-15): INT8
    # Late layers (16-31): BF16
    strategies['conservative_gradient'] = create_manual_config(
        num_layers=num_layers,
        early_quant=QuantizationStrategy.INT8,
        mid_quant=QuantizationStrategy.INT8,
        late_quant=QuantizationStrategy.BF16,
        early_cutoff=16,
        late_cutoff=16  # No middle section
    )

    # Strategy 5: Aggressive layer-wise
    # Early layers (0-20): INT4
    # Late layers (21-31): INT8
    strategies['aggressive_gradient'] = create_manual_config(
        num_layers=num_layers,
        early_quant=QuantizationStrategy.INT4,
        mid_quant=QuantizationStrategy.INT4,
        late_quant=QuantizationStrategy.INT8,
        early_cutoff=21,
        late_cutoff=21
    )

    # Strategy 6: Based on actual activation profile
    strategies['activation_based'] = create_layer_configs_from_activation_profile(
        activation_analysis=activation_analysis,
        num_layers=num_layers,
        aggressive_threshold=0.3,
        conservative_threshold=0.7
    )

    return strategies


def print_strategy_summary(strategies: dict):
    """Print summary of quantization strategies."""
    print("\n" + "="*80)
    print("QUANTIZATION STRATEGIES TO TEST")
    print("="*80)

    for name, configs in strategies.items():
        print(f"\n{name}:")

        # Count layers by quantization level
        quant_counts = {}
        for config in configs:
            # Simplified: just count MLP quantization
            q = config.mlp_quant
            quant_counts[q] = quant_counts.get(q, 0) + 1

        for quant, count in sorted(quant_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {quant.value:6s}: {count:2d} layers")

        # Show layer ranges
        print("  Layer breakdown:")
        current_quant = configs[0].mlp_quant
        start_idx = 0

        for i, config in enumerate(configs):
            if config.mlp_quant != current_quant or i == len(configs) - 1:
                end_idx = i if config.mlp_quant != current_quant else i + 1
                print(f"    Layers {start_idx:2d}-{end_idx-1:2d}: {current_quant.value}")
                current_quant = config.mlp_quant
                start_idx = i


def evaluate_strategies(
    model,
    tokenizer,
    strategies: dict,
    test_texts: List[str],
    baseline_memory_gb: float,
    max_length: int = 512
) -> dict:
    """Evaluate all quantization strategies.

    Args:
        model: The model (not quantized)
        tokenizer: Tokenizer
        strategies: Dictionary of strategy_name -> configs
        test_texts: Test dataset
        baseline_memory_gb: Baseline memory usage
        max_length: Max sequence length

    Returns:
        Dictionary of results per strategy
    """
    results = {}

    print("\n" + "="*80)
    print("EVALUATING STRATEGIES")
    print("="*80)

    # Baseline perplexity (BF16, no quantization)
    print("\n[1/N] Baseline (BF16)...")
    baseline_ppl = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    results['baseline_bf16'] = {
        'perplexity': baseline_ppl['perplexity'],
        'memory_gb': baseline_memory_gb,
        'perplexity_delta': 0.0,
        'perplexity_increase_pct': 0.0,
        'memory_savings_gb': 0.0,
        'memory_savings_pct': 0.0,
        'compression_ratio': 1.0
    }

    print(f"  Perplexity: {baseline_ppl['perplexity']:.2f}")
    print(f"  Memory: {baseline_memory_gb:.2f} GB")

    # Evaluate each strategy
    for idx, (strategy_name, configs) in enumerate(strategies.items(), start=2):
        print(f"\n[{idx}/{len(strategies)+1}] {strategy_name}...")

        # Note: We're not actually quantizing here, just estimating
        # In a real implementation, you'd quantize and measure actual perplexity

        # For now, estimate memory and use baseline perplexity
        # (actual quantization would require bitsandbytes or similar)
        memory_est = estimate_quantized_memory(
            baseline_memory_gb=baseline_memory_gb,
            configs=configs,
            num_layers=len(configs)
        )

        # Placeholder: use baseline perplexity
        # In reality, you'd quantize and re-evaluate
        ppl = baseline_ppl['perplexity']  # This would change with actual quantization

        results[strategy_name] = {
            'perplexity': ppl,
            'memory_gb': memory_est['estimated_memory_gb'],
            'perplexity_delta': ppl - baseline_ppl['perplexity'],
            'perplexity_increase_pct': ((ppl / baseline_ppl['perplexity']) - 1) * 100,
            'memory_savings_gb': memory_est['savings_gb'],
            'memory_savings_pct': memory_est['savings_percent'],
            'compression_ratio': memory_est['compression_ratio']
        }

        print(f"  Estimated Memory: {memory_est['estimated_memory_gb']:.2f} GB "
              f"(-{memory_est['savings_percent']:.1f}%)")
        print(f"  Compression: {memory_est['compression_ratio']:.2f}x")
        print(f"  Note: Perplexity not measured (would require actual quantization)")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test layer-wise quantization strategies"
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
        help='Path to activation analysis results'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    print("="*80)
    print("LAYER-WISE QUANTIZATION STRATEGY EVALUATION")
    print("="*80)

    # Load activation analysis
    print("\n[1/6] Loading activation analysis...")
    activation_analysis = load_activation_analysis(args.activation_analysis)
    num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])
    print(f"  Found analysis for {num_layers} layers")

    # Load model
    print("\n[2/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer(config.model)

    # Get baseline memory
    print("\n[3/6] Measuring baseline memory...")
    baseline_memory = get_model_memory_footprint(model)
    print(f"  Total parameters: {baseline_memory['total_params_millions']:.1f}M")
    print(f"  Memory usage: {baseline_memory['total_memory_gb']:.2f} GB")

    # Create quantization strategies
    print("\n[4/6] Creating quantization strategies...")
    strategies = create_quantization_strategies(activation_analysis, num_layers)
    print_strategy_summary(strategies)

    # Load test data (reuse prompt datasets)
    print("\n[5/6] Loading test data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)
    print(f"  Loaded {len(test_texts)} test samples")

    # Evaluate strategies
    print("\n[6/6] Evaluating strategies...")
    results = evaluate_strategies(
        model=model,
        tokenizer=tokenizer,
        strategies=strategies,
        test_texts=test_texts,
        baseline_memory_gb=baseline_memory['total_memory_gb'],
        max_length=config.data.max_length
    )

    # Save results
    output_dir = config.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'quantization_strategies.json')
    with open(results_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for name, data in results.items():
            serializable_results[name] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in data.items()
            }
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved results to: {results_file}")

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Strategy':<25} {'Memory (GB)':<12} {'Savings':<10} {'Compression':<12}")
    print("-"*80)

    for name, data in results.items():
        print(f"{name:<25} "
              f"{data['memory_gb']:>10.2f}  "
              f"{data['memory_savings_pct']:>8.1f}%  "
              f"{data['compression_ratio']:>10.2f}x")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nNote: Actual quantization and perplexity measurement would require")
    print("proper quantization implementation (e.g., bitsandbytes).")
    print("These results show estimated memory savings based on the strategies.")


if __name__ == '__main__':
    main()
