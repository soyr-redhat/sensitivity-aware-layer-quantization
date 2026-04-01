#!/usr/bin/env python3
"""Final benchmark: Compare baseline, uniform, and layer-wise quantization.

This script properly benchmarks all approaches by measuring actual runtime
memory and perplexity for each configuration.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.data.prompts import load_prompts_dataset
from src.config import Config
from src.quantization import (
    QuantizationStrategy,
    create_manual_config,
    create_layer_configs_from_activation_profile,
    get_model_memory_footprint,
)
from src.quantization.apply_quant import apply_layer_wise_quantization_real
from src.evaluation import calculate_perplexity


def load_activation_analysis(analysis_path: str) -> dict:
    """Load activation analysis results."""
    with open(analysis_path, 'r') as f:
        return json.load(f)


def get_gpu_memory_allocated() -> float:
    """Get actual GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_gpu_memory_reserved() -> float:
    """Get GPU memory reserved in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024**3)
    return 0.0


def benchmark_configuration(
    config_name: str,
    model,
    tokenizer,
    test_texts: List[str],
    max_length: int = 512
) -> Dict:
    """Benchmark a single configuration.

    Args:
        config_name: Name of the configuration
        model: The model to benchmark
        tokenizer: Tokenizer
        test_texts: Test samples
        max_length: Max sequence length

    Returns:
        Benchmark results
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {config_name}")
    print(f"{'='*80}")

    # Measure memory
    print("\n[1/2] Measuring memory usage...")

    # Model footprint
    model_memory = get_model_memory_footprint(model)

    # GPU memory
    gpu_allocated = get_gpu_memory_allocated()
    gpu_reserved = get_gpu_memory_reserved()

    print(f"  Model memory: {model_memory['total_memory_gb']:.2f} GB")
    print(f"  GPU allocated: {gpu_allocated:.2f} GB")
    print(f"  GPU reserved: {gpu_reserved:.2f} GB")
    print(f"  Parameters: {model_memory['total_params_millions']:.1f}M")

    # Measure perplexity
    print("\n[2/2] Measuring perplexity...")
    metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    print(f"  Perplexity: {metrics['perplexity']:.4f}")
    print(f"  Avg Loss: {metrics['avg_loss']:.4f}")

    return {
        'config_name': config_name,
        'perplexity': metrics['perplexity'],
        'avg_loss': metrics['avg_loss'],
        'model_memory_gb': model_memory['total_memory_gb'],
        'gpu_allocated_gb': gpu_allocated,
        'gpu_reserved_gb': gpu_reserved,
        'total_params_m': model_memory['total_params_millions'],
        'metrics': metrics,
        'memory_stats': model_memory
    }


def load_baseline_model(model_name: str, cache_dir: str, device_map: str = "auto"):
    """Load baseline BF16 model."""
    print("\n" + "="*80)
    print("Loading BASELINE (BF16)")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    return model


def load_uniform_int8_model(model_name: str, cache_dir: str, device_map: str = "auto"):
    """Load uniformly quantized INT8 model."""
    print("\n" + "="*80)
    print("Loading UNIFORM INT8")
    print("="*80)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.eval()
    return model


def load_uniform_int4_model(model_name: str, cache_dir: str, device_map: str = "auto"):
    """Load uniformly quantized INT4 model."""
    print("\n" + "="*80)
    print("Loading UNIFORM INT4")
    print("="*80)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.eval()
    return model


def load_and_quantize_layerwise(
    model_name: str,
    cache_dir: str,
    layer_configs: List,
    strategy_name: str,
    device_map: str = "auto"
):
    """Load model and apply layer-wise quantization."""
    print("\n" + "="*80)
    print(f"Loading and applying LAYER-WISE: {strategy_name}")
    print("="*80)

    # Load BF16 model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # Apply layer-wise quantization
    apply_layer_wise_quantization_real(
        model=model,
        configs=layer_configs,
        verbose=True
    )

    return model


def main():
    """Main benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Final benchmark of quantization strategies"
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
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    print("="*80)
    print("FINAL QUANTIZATION BENCHMARK")
    print("="*80)
    print("\nThis benchmark measures actual runtime memory and perplexity")
    print("for different quantization strategies.\n")

    # Load test data
    print("[1/8] Loading test data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)
    print(f"  Loaded {len(test_texts)} test samples")

    # Load tokenizer
    print("\n[2/8] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load activation analysis for layer-wise strategies
    print("\n[3/8] Loading activation analysis...")
    activation_analysis = load_activation_analysis(args.activation_analysis)
    num_layers = len(activation_analysis['prompt_type_signatures']['code']['mlp_profile'])
    print(f"  Found {num_layers} layers")

    # Create layer-wise configs
    layer_configs = {
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

    all_results = {}

    # Benchmark 1: Baseline (BF16)
    print("\n[4/8] Benchmarking baseline (BF16)...")
    model = load_baseline_model(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        device_map=config.model.device_map
    )
    baseline_results = benchmark_configuration(
        config_name="baseline_bf16",
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )
    all_results['baseline_bf16'] = baseline_results

    baseline_ppl = baseline_results['perplexity']
    baseline_memory = baseline_results['model_memory_gb']

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark 2: Uniform INT8
    print("\n[5/8] Benchmarking uniform INT8...")
    model = load_uniform_int8_model(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        device_map=config.model.device_map
    )
    int8_results = benchmark_configuration(
        config_name="uniform_int8",
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )
    all_results['uniform_int8'] = int8_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark 3: Uniform INT4
    print("\n[6/8] Benchmarking uniform INT4...")
    model = load_uniform_int4_model(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        device_map=config.model.device_map
    )
    int4_results = benchmark_configuration(
        config_name="uniform_int4",
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )
    all_results['uniform_int4'] = int4_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark 4: Manual gradient layer-wise
    print("\n[7/8] Benchmarking manual_gradient layer-wise...")
    model = load_and_quantize_layerwise(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        layer_configs=layer_configs['manual_gradient'],
        strategy_name='manual_gradient',
        device_map=config.model.device_map
    )
    manual_results = benchmark_configuration(
        config_name="layerwise_manual",
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )
    all_results['layerwise_manual'] = manual_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark 5: Activation-based layer-wise
    print("\n[8/8] Benchmarking activation_based layer-wise...")
    model = load_and_quantize_layerwise(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        layer_configs=layer_configs['activation_based'],
        strategy_name='activation_based',
        device_map=config.model.device_map
    )
    activation_results = benchmark_configuration(
        config_name="layerwise_activation",
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=config.data.max_length
    )
    all_results['layerwise_activation'] = activation_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate comparative metrics
    for name, results in all_results.items():
        if name == 'baseline_bf16':
            results['ppl_delta_pct'] = 0.0
            results['memory_savings_pct'] = 0.0
            results['compression_ratio'] = 1.0
        else:
            ppl_delta = ((results['perplexity'] / baseline_ppl) - 1) * 100
            mem_savings = ((baseline_memory - results['model_memory_gb']) / baseline_memory) * 100
            compression = baseline_memory / results['model_memory_gb']

            results['ppl_delta_pct'] = ppl_delta
            results['memory_savings_pct'] = mem_savings
            results['compression_ratio'] = compression

    # Save results
    output_dir = config.analysis.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'final_benchmark_results.json')
    with open(results_file, 'w') as f:
        serializable = {}
        for name, data in all_results.items():
            serializable[name] = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        json.dump(serializable, f, indent=2)

    print(f"\n\nSaved results to: {results_file}")

    # Print comprehensive summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    print(f"\n{'Strategy':<25} {'PPL':<10} {'PPL Δ%':<10} {'Memory':<12} {'GPU Alloc':<12} {'Savings':<10} {'Ratio':<8}")
    print("-"*100)

    for name, data in all_results.items():
        ppl = data['perplexity']
        ppl_delta = data.get('ppl_delta_pct', 0.0)
        mem = data['model_memory_gb']
        gpu = data['gpu_allocated_gb']
        savings = data.get('memory_savings_pct', 0.0)
        ratio = data.get('compression_ratio', 1.0)

        print(f"{name:<25} {ppl:>8.4f}  {ppl_delta:>8.2f}%  "
              f"{mem:>10.2f}GB  {gpu:>10.2f}GB  {savings:>8.1f}%  {ratio:>6.2f}x")

    # Print quality analysis
    print("\n" + "="*80)
    print("QUALITY ANALYSIS")
    print("="*80)
    print("\nPerplexity changes from baseline:")
    for name, data in all_results.items():
        if name != 'baseline_bf16':
            ppl_delta = data.get('ppl_delta_pct', 0.0)
            status = "✓" if abs(ppl_delta) < 1.0 else "!"
            print(f"  {status} {name:<25} {ppl_delta:+.2f}%")

    # Print memory analysis
    print("\n" + "="*80)
    print("MEMORY ANALYSIS")
    print("="*80)
    print("\nMemory savings from baseline:")
    for name, data in all_results.items():
        if name != 'baseline_bf16':
            savings = data.get('memory_savings_pct', 0.0)
            ratio = data.get('compression_ratio', 1.0)
            print(f"  {name:<25} {savings:>6.1f}% savings  ({ratio:.2f}x compression)")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
