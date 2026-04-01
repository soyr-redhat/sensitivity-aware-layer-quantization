#!/usr/bin/env python3
"""Measure actual quantization performance with real perplexity.

This script applies actual quantization to the model and measures
the perplexity impact compared to baseline.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import gc
from src.model.loader import load_tokenizer
from src.data.prompts import load_prompts_dataset
from src.config import Config
from src.quantization import (
    QuantizationStrategy,
    create_uniform_config,
    create_manual_config,
    create_layer_configs_from_activation_profile,
    get_model_memory_footprint,
)
from src.quantization.apply_quant import (
    load_quantized_model,
    apply_layer_wise_quantization_real
)
from src.evaluation import calculate_perplexity


def load_activation_analysis(analysis_path: str) -> dict:
    """Load activation analysis results."""
    with open(analysis_path, 'r') as f:
        return json.load(f)


def test_baseline(model, tokenizer, test_texts, max_length=512):
    """Test baseline model (BF16)."""
    print("\n" + "="*80)
    print("BASELINE: BF16 (No Quantization)")
    print("="*80)

    memory = get_model_memory_footprint(model)
    print(f"\nMemory: {memory['total_memory_gb']:.2f} GB")
    print(f"Parameters: {memory['total_params_millions']:.1f}M")

    print("\nCalculating perplexity...")
    metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Avg Loss: {metrics['avg_loss']:.4f}")

    return {
        'strategy': 'baseline_bf16',
        'perplexity': metrics['perplexity'],
        'memory_gb': memory['total_memory_gb'],
        'memory_params_m': memory['total_params_millions'],
        'metrics': metrics,
        'memory': memory
    }


def test_uniform_quantization(
    model_name: str,
    cache_dir: str,
    tokenizer,
    test_texts: List[str],
    baseline_memory: float,
    baseline_ppl: float,
    max_length: int = 512
) -> Dict:
    """Test uniform quantization strategies (INT8, INT4)."""
    results = {}

    # Test INT8
    print("\n" + "="*80)
    print("UNIFORM INT8 QUANTIZATION")
    print("="*80)

    try:
        print("\nLoading INT8 quantized model...")
        model_int8 = load_quantized_model(
            model_name=model_name,
            cache_dir=cache_dir,
            quantization_config={'load_in_8bit': True}
        )

        memory_int8 = get_model_memory_footprint(model_int8)
        print(f"Memory: {memory_int8['total_memory_gb']:.2f} GB")

        print("Calculating perplexity...")
        metrics_int8 = calculate_perplexity(
            model=model_int8,
            tokenizer=tokenizer,
            texts=test_texts,
            max_length=max_length
        )

        print(f"Perplexity: {metrics_int8['perplexity']:.2f} "
              f"(+{((metrics_int8['perplexity']/baseline_ppl - 1)*100):.2f}%)")

        results['uniform_int8'] = {
            'strategy': 'uniform_int8',
            'perplexity': metrics_int8['perplexity'],
            'perplexity_delta': metrics_int8['perplexity'] - baseline_ppl,
            'perplexity_increase_pct': (metrics_int8['perplexity']/baseline_ppl - 1) * 100,
            'memory_gb': memory_int8['total_memory_gb'],
            'memory_savings_pct': ((baseline_memory - memory_int8['total_memory_gb'])/baseline_memory) * 100,
            'compression_ratio': baseline_memory / memory_int8['total_memory_gb']
        }

        # Clean up
        del model_int8
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing INT8: {e}")
        results['uniform_int8'] = {'error': str(e)}

    # Test INT4
    print("\n" + "="*80)
    print("UNIFORM INT4 QUANTIZATION")
    print("="*80)

    try:
        print("\nLoading INT4 quantized model...")
        model_int4 = load_quantized_model(
            model_name=model_name,
            cache_dir=cache_dir,
            quantization_config={'load_in_4bit': True}
        )

        memory_int4 = get_model_memory_footprint(model_int4)
        print(f"Memory: {memory_int4['total_memory_gb']:.2f} GB")

        print("Calculating perplexity...")
        metrics_int4 = calculate_perplexity(
            model=model_int4,
            tokenizer=tokenizer,
            texts=test_texts,
            max_length=max_length
        )

        print(f"Perplexity: {metrics_int4['perplexity']:.2f} "
              f"(+{((metrics_int4['perplexity']/baseline_ppl - 1)*100):.2f}%)")

        results['uniform_int4'] = {
            'strategy': 'uniform_int4',
            'perplexity': metrics_int4['perplexity'],
            'perplexity_delta': metrics_int4['perplexity'] - baseline_ppl,
            'perplexity_increase_pct': (metrics_int4['perplexity']/baseline_ppl - 1) * 100,
            'memory_gb': memory_int4['total_memory_gb'],
            'memory_savings_pct': ((baseline_memory - memory_int4['total_memory_gb'])/baseline_memory) * 100,
            'compression_ratio': baseline_memory / memory_int4['total_memory_gb']
        }

        # Clean up
        del model_int4
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing INT4: {e}")
        results['uniform_int4'] = {'error': str(e)}

    return results


def test_layer_wise_quantization(
    model,
    tokenizer,
    layer_configs: List,
    strategy_name: str,
    test_texts: List[str],
    baseline_memory: float,
    baseline_ppl: float,
    max_length: int = 512
) -> Dict:
    """Test a layer-wise quantization strategy."""
    print("\n" + "="*80)
    print(f"LAYER-WISE: {strategy_name}")
    print("="*80)

    try:
        # Apply quantization
        apply_layer_wise_quantization_real(
            model=model,
            configs=layer_configs,
            verbose=True
        )

        # Measure memory
        memory = get_model_memory_footprint(model)
        print(f"\nMemory after quantization: {memory['total_memory_gb']:.2f} GB")

        # Measure perplexity
        print("Calculating perplexity...")
        metrics = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            max_length=max_length
        )

        print(f"Perplexity: {metrics['perplexity']:.2f} "
              f"(+{((metrics['perplexity']/baseline_ppl - 1)*100):.2f}%)")

        return {
            'strategy': strategy_name,
            'perplexity': metrics['perplexity'],
            'perplexity_delta': metrics['perplexity'] - baseline_ppl,
            'perplexity_increase_pct': (metrics['perplexity']/baseline_ppl - 1) * 100,
            'memory_gb': memory['total_memory_gb'],
            'memory_savings_pct': ((baseline_memory - memory['total_memory_gb'])/baseline_memory) * 100,
            'compression_ratio': baseline_memory / memory['total_memory_gb']
        }

    except Exception as e:
        print(f"Error testing {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'strategy': strategy_name, 'error': str(e)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure real quantization performance"
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
        '--test-layer-wise',
        action='store_true',
        help='Test layer-wise strategies (requires manual quantization)'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    print("="*80)
    print("REAL QUANTIZATION PERFORMANCE MEASUREMENT")
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

    # Test baseline
    print("\n[3/6] Testing baseline model...")
    from transformers import AutoModelForCausalLM

    model_baseline = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        device_map=config.model.device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model_baseline.eval()

    baseline_results = test_baseline(
        model=model_baseline,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )

    baseline_memory = baseline_results['memory_gb']
    baseline_ppl = baseline_results['perplexity']

    all_results = {'baseline_bf16': baseline_results}

    # Clean up baseline model
    del model_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # Test uniform quantization
    print("\n[4/6] Testing uniform quantization...")
    uniform_results = test_uniform_quantization(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        tokenizer=tokenizer,
        test_texts=test_texts,
        baseline_memory=baseline_memory,
        baseline_ppl=baseline_ppl,
        max_length=config.data.max_length
    )
    all_results.update(uniform_results)

    # Test layer-wise strategies if requested
    if args.test_layer_wise:
        print("\n[5/6] Testing layer-wise quantization...")
        print("Note: Layer-wise quantization is experimental")

        # Load activation analysis
        activation_analysis = load_activation_analysis(args.activation_analysis)
        num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])

        # Create strategies
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
            )
        }

        # Test each strategy
        for strategy_name, layer_configs in strategies.items():
            # Reload model for each test
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                cache_dir=config.model.cache_dir,
                device_map=config.model.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model.eval()

            result = test_layer_wise_quantization(
                model=model,
                tokenizer=tokenizer,
                layer_configs=layer_configs,
                strategy_name=strategy_name,
                test_texts=test_texts,
                baseline_memory=baseline_memory,
                baseline_ppl=baseline_ppl,
                max_length=config.data.max_length
            )
            all_results[strategy_name] = result

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    print("\n[6/6] Saving results...")
    output_dir = config.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'real_quantization_results.json')
    with open(results_file, 'w') as f:
        # Serialize results
        serializable_results = {}
        for name, data in all_results.items():
            serializable_results[name] = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    continue  # Skip nested dicts
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_results[name][k] = v
        json.dump(serializable_results, f, indent=2)

    print(f"Saved results to: {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Strategy':<25} {'PPL':<8} {'PPL Δ%':<10} {'Memory':<10} {'Savings':<10} {'Ratio':<8}")
    print("-"*80)

    for name, data in all_results.items():
        if 'error' in data:
            print(f"{name:<25} ERROR: {data['error']}")
            continue

        ppl = data.get('perplexity', 0)
        ppl_delta = data.get('perplexity_increase_pct', 0)
        mem = data.get('memory_gb', 0)
        savings = data.get('memory_savings_pct', 0)
        ratio = data.get('compression_ratio', 1)

        print(f"{name:<25} {ppl:>6.2f}  {ppl_delta:>8.2f}%  "
              f"{mem:>8.2f}GB  {savings:>8.1f}%  {ratio:>6.2f}x")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
