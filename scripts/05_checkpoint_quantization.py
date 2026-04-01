#!/usr/bin/env python3
"""Create and evaluate quantized checkpoints with proper memory management.

This script creates separate checkpoints for each quantization strategy,
then loads and evaluates each one independently to get accurate memory measurements.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from src.model.loader import load_tokenizer
from src.data.prompts import load_prompts_dataset
from src.config import Config
from src.quantization import (
    QuantizationStrategy,
    create_manual_config,
    create_layer_configs_from_activation_profile,
    get_model_memory_footprint,
)
from src.quantization.checkpoint import (
    create_quantized_checkpoint_strategy,
    load_quantized_model_from_checkpoint,
    estimate_checkpoint_size
)
from src.evaluation import calculate_perplexity


def load_activation_analysis(analysis_path: str) -> dict:
    """Load activation analysis results."""
    with open(analysis_path, 'r') as f:
        return json.load(f)


def evaluate_checkpoint(
    checkpoint_path: str,
    tokenizer,
    test_texts: List[str],
    baseline_memory: float,
    baseline_ppl: float,
    max_length: int = 512
) -> Dict:
    """Evaluate a quantized checkpoint.

    Args:
        checkpoint_path: Path to the quantized checkpoint
        tokenizer: Tokenizer
        test_texts: Test samples
        baseline_memory: Baseline memory for comparison
        baseline_ppl: Baseline perplexity for comparison
        max_length: Max sequence length

    Returns:
        Evaluation results
    """
    strategy_name = os.path.basename(checkpoint_path)

    print(f"\n{'='*80}")
    print(f"EVALUATING: {strategy_name}")
    print(f"{'='*80}")

    # Free all memory before loading
    gc.collect()
    torch.cuda.empty_cache()

    # Load the checkpoint
    print("\n[1/3] Loading checkpoint...")
    model = load_quantized_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device_map="auto"
    )

    # Measure memory
    print("\n[2/3] Measuring memory...")
    memory = get_model_memory_footprint(model)
    print(f"  Memory: {memory['total_memory_gb']:.2f} GB")

    # Check disk size
    disk_size = estimate_checkpoint_size(checkpoint_path)
    print(f"  Disk size: {disk_size['total_size_gb']:.2f} GB")

    # Measure perplexity
    print("\n[3/3] Measuring perplexity...")
    metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    ppl = metrics['perplexity']
    ppl_increase = ((ppl / baseline_ppl) - 1) * 100
    mem_savings = ((baseline_memory - memory['total_memory_gb']) / baseline_memory) * 100
    compression = baseline_memory / memory['total_memory_gb']

    print(f"\n  Perplexity: {ppl:.2f} ({ppl_increase:+.2f}%)")
    print(f"  Memory savings: {mem_savings:.1f}%")
    print(f"  Compression: {compression:.2f}x")

    results = {
        'strategy': strategy_name,
        'checkpoint_path': checkpoint_path,
        'perplexity': ppl,
        'perplexity_delta': ppl - baseline_ppl,
        'perplexity_increase_pct': ppl_increase,
        'memory_gb': memory['total_memory_gb'],
        'memory_savings_gb': baseline_memory - memory['total_memory_gb'],
        'memory_savings_pct': mem_savings,
        'compression_ratio': compression,
        'disk_size_gb': disk_size['total_size_gb'],
        'metrics': metrics,
        'memory_stats': memory
    }

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and evaluate quantized checkpoints"
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
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/workspace/outputs/checkpoints',
        help='Directory to save quantized checkpoints'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline measurement (use previous results)'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    print("="*80)
    print("LAYER-WISE QUANTIZATION WITH CHECKPOINT EVALUATION")
    print("="*80)

    # Load test data
    print("\n[1/7] Loading test data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)
    print(f"  Loaded {len(test_texts)} test samples")

    # Load tokenizer
    print("\n[2/7] Loading tokenizer...")
    tokenizer = load_tokenizer(config.model)

    # Load activation analysis
    print("\n[3/7] Loading activation analysis...")
    activation_analysis = load_activation_analysis(args.activation_analysis)
    num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])
    print(f"  Found analysis for {num_layers} layers")

    # Measure baseline if needed
    if not args.skip_baseline:
        print("\n[4/7] Measuring baseline...")
        from transformers import AutoModelForCausalLM

        model_baseline = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            cache_dir=config.model.cache_dir,
            device_map=config.model.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model_baseline.eval()

        baseline_memory = get_model_memory_footprint(model_baseline)
        print(f"  Memory: {baseline_memory['total_memory_gb']:.2f} GB")

        baseline_metrics = calculate_perplexity(
            model=model_baseline,
            tokenizer=tokenizer,
            texts=test_texts,
            max_length=config.data.max_length
        )
        baseline_ppl = baseline_metrics['perplexity']
        print(f"  Perplexity: {baseline_ppl:.2f}")

        baseline_results = {
            'strategy': 'baseline_bf16',
            'perplexity': baseline_ppl,
            'memory_gb': baseline_memory['total_memory_gb'],
            'perplexity_delta': 0.0,
            'perplexity_increase_pct': 0.0,
            'memory_savings_pct': 0.0,
            'compression_ratio': 1.0
        }

        # Clean up
        del model_baseline
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[4/7] Using previous baseline...")
        baseline_ppl = 10.77  # From previous run
        baseline_memory = {'total_memory_gb': 13.50}
        baseline_results = {
            'strategy': 'baseline_bf16',
            'perplexity': baseline_ppl,
            'memory_gb': 13.50,
            'perplexity_delta': 0.0,
            'perplexity_increase_pct': 0.0,
            'memory_savings_pct': 0.0,
            'compression_ratio': 1.0
        }
        print(f"  Baseline: {baseline_ppl:.2f} PPL @ {baseline_memory['total_memory_gb']:.2f} GB")

    all_results = {'baseline_bf16': baseline_results}

    # Create quantization strategies
    print("\n[5/7] Creating quantization strategies...")
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
        'conservative': create_manual_config(
            num_layers=num_layers,
            early_quant=QuantizationStrategy.INT8,
            mid_quant=QuantizationStrategy.INT8,
            late_quant=QuantizationStrategy.BF16,
            early_cutoff=16,
            late_cutoff=16
        ),
    }

    # Create checkpoints for each strategy
    print("\n[6/7] Creating quantized checkpoints...")
    checkpoint_paths = {}

    for strategy_name, layer_configs in strategies.items():
        checkpoint_path = create_quantized_checkpoint_strategy(
            base_model_name=config.model.name,
            cache_dir=config.model.cache_dir,
            configs=layer_configs,
            output_dir=args.checkpoint_dir,
            strategy_name=strategy_name,
            device_map=config.model.device_map
        )
        checkpoint_paths[strategy_name] = checkpoint_path

    # Evaluate each checkpoint
    print("\n[7/7] Evaluating checkpoints...")

    for strategy_name, checkpoint_path in checkpoint_paths.items():
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            test_texts=test_texts,
            baseline_memory=baseline_memory['total_memory_gb'],
            baseline_ppl=baseline_ppl,
            max_length=config.data.max_length
        )
        all_results[strategy_name] = results

    # Save results
    output_dir = config.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'checkpoint_quantization_results.json')
    with open(results_file, 'w') as f:
        # Serialize results
        serializable_results = {}
        for name, data in all_results.items():
            serializable_results[name] = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    continue  # Skip nested structures
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_results[name][k] = v
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved results to: {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Strategy':<25} {'PPL':<8} {'PPL Δ%':<10} {'Memory':<10} {'Savings':<10} {'Ratio':<8}")
    print("-"*80)

    for name, data in all_results.items():
        ppl = data.get('perplexity', 0)
        ppl_delta = data.get('perplexity_increase_pct', 0)
        mem = data.get('memory_gb', 0)
        savings = data.get('memory_savings_pct', 0)
        ratio = data.get('compression_ratio', 1)

        print(f"{name:<25} {ppl:>6.2f}  {ppl_delta:>8.2f}%  "
              f"{mem:>8.2f}GB  {savings:>8.1f}%  {ratio:>6.2f}x")

    print("\n" + "="*80)
    print("CHECKPOINT EVALUATION COMPLETE")
    print("="*80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
