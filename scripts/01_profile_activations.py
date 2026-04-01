#!/usr/bin/env python3
"""Profile activation patterns for different prompt types.

This script analyzes how different types of prompts (code, math, creative, etc.)
activate different parts of the model, which informs prompt-aware quantization
strategies.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from src.model.loader import load_model_and_tokenizer
from src.model.activations import collect_activations
from src.data.prompts import load_prompts_dataset
from src.config import Config


def profile_activations(config):
    """Profile activation patterns for different prompt types."""

    print("=" * 80)
    print("PHASE 1: ACTIVATION PROFILING")
    print("=" * 80)

    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model)

    # Load prompt datasets
    print("\n[2/5] Loading prompt datasets...")
    prompt_datasets = load_prompts_dataset(config)
    print(f"Loaded {len(prompt_datasets)} prompt types")
    print(f"Total prompts: {sum(len(p) for p in prompt_datasets.values())}")

    # Create output directories
    os.makedirs(config.activation_analysis.output_dir, exist_ok=True)
    os.makedirs(config.analysis.output_dir, exist_ok=True)

    # Collect activation statistics for each prompt type
    print("\n[3/5] Collecting activation statistics...")
    activation_results = {}

    for prompt_type, prompts in prompt_datasets.items():
        print(f"\nProcessing prompt type: {prompt_type}")
        print(f"  Samples: {len(prompts)}")

        # Collect activations
        results = collect_activations(
            model=model,
            tokenizer=tokenizer,
            texts=prompts,
            prompt_type=prompt_type,
            max_length=config.data.max_length
        )

        activation_results[prompt_type] = results

        # Save individual results
        output_file = os.path.join(
            config.activation_analysis.output_dir,
            f'activations_{prompt_type}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {output_file}")

    # Analyze cross-prompt patterns
    print("\n[4/5] Analyzing cross-prompt patterns...")
    analysis = analyze_activation_patterns(activation_results)

    # Save analysis
    analysis_file = os.path.join(
        config.analysis.output_dir,
        'activation_analysis.json'
    )
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_file}")

    # Generate report
    print("\n[5/5] Generating summary report...")
    generate_summary_report(activation_results, analysis)

    print("\n" + "=" * 80)
    print("ACTIVATION PROFILING COMPLETE")
    print("=" * 80)


def analyze_activation_patterns(activation_results):
    """Analyze activation patterns across different prompt types.

    Args:
        activation_results: Dictionary of activation statistics per prompt type

    Returns:
        Analysis results
    """
    analysis = {
        'per_layer_comparison': {},
        'prompt_type_signatures': {},
        'layer_importance_variance': {}
    }

    # Get number of layers from first result
    first_result = next(iter(activation_results.values()))
    num_layers = first_result['num_layers']

    # Compare activation levels across prompt types for each layer
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        layer_comparison = {}

        for prompt_type, results in activation_results.items():
            if layer_key in results['activation_stats']['mlp']:
                mlp_stats = results['activation_stats']['mlp'][layer_key]
                attn_stats = results['activation_stats']['attention'][layer_key]

                layer_comparison[prompt_type] = {
                    'mlp_avg_mean': mlp_stats['avg_mean'],
                    'mlp_avg_max': mlp_stats['avg_max'],
                    'attn_avg_mean': attn_stats['avg_mean'],
                    'attn_avg_max': attn_stats['avg_max'],
                }

        analysis['per_layer_comparison'][layer_key] = layer_comparison

        # Calculate variance across prompt types for this layer
        mlp_means = [v['mlp_avg_mean'] for v in layer_comparison.values()]
        attn_means = [v['attn_avg_mean'] for v in layer_comparison.values()]

        analysis['layer_importance_variance'][layer_key] = {
            'mlp_variance': float(np.var(mlp_means)),
            'attn_variance': float(np.var(attn_means)),
            'mlp_range': float(max(mlp_means) - min(mlp_means)),
            'attn_range': float(max(attn_means) - min(attn_means)),
        }

    # Create prompt type signatures (which layers are most active)
    for prompt_type, results in activation_results.items():
        mlp_activations = []
        attn_activations = []

        for layer_idx in range(num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in results['activation_stats']['mlp']:
                mlp_activations.append(
                    results['activation_stats']['mlp'][layer_key]['avg_mean']
                )
                attn_activations.append(
                    results['activation_stats']['attention'][layer_key]['avg_mean']
                )

        analysis['prompt_type_signatures'][prompt_type] = {
            'mlp_profile': mlp_activations,
            'attn_profile': attn_activations,
            'most_active_mlp_layers': sorted(
                range(len(mlp_activations)),
                key=lambda i: mlp_activations[i],
                reverse=True
            )[:5],  # Top 5 most active layers
            'least_active_mlp_layers': sorted(
                range(len(mlp_activations)),
                key=lambda i: mlp_activations[i]
            )[:5],  # Bottom 5 least active layers
        }

    return analysis


def generate_summary_report(activation_results, analysis):
    """Generate a human-readable summary report."""

    print("\n" + "=" * 80)
    print("ACTIVATION ANALYSIS SUMMARY")
    print("=" * 80)

    num_layers = next(iter(activation_results.values()))['num_layers']

    # Report on layer-wise variance
    print("\nLayer-wise Activation Variance:")
    print("(Higher variance = different prompt types activate layers differently)")
    print("-" * 80)

    high_variance_layers = []
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        variance = analysis['layer_importance_variance'][layer_key]

        if variance['mlp_variance'] > 0.1 or variance['attn_variance'] > 0.1:
            high_variance_layers.append((layer_idx, variance))

    if high_variance_layers:
        print(f"\nHigh variance layers (top 10):")
        for layer_idx, var in sorted(high_variance_layers,
                                     key=lambda x: x[1]['mlp_variance'],
                                     reverse=True)[:10]:
            print(f"  Layer {layer_idx:2d}: MLP var={var['mlp_variance']:.4f}, "
                  f"Attn var={var['attn_variance']:.4f}")
    else:
        print("\nNo high variance layers found (variance < 0.1)")
        print("This suggests prompts activate layers similarly across types.")

    # Report on prompt type signatures
    print("\n" + "-" * 80)
    print("Prompt Type Activation Signatures:")
    print("-" * 80)

    for prompt_type, signature in analysis['prompt_type_signatures'].items():
        print(f"\n{prompt_type.upper()}:")
        print(f"  Most active MLP layers: {signature['most_active_mlp_layers']}")
        print(f"  Least active MLP layers: {signature['least_active_mlp_layers']}")

    # Calculate similarity between prompt types
    print("\n" + "-" * 80)
    print("Prompt Type Similarity Analysis:")
    print("(Cosine similarity of activation profiles)")
    print("-" * 80)

    prompt_types = list(analysis['prompt_type_signatures'].keys())
    for i, type1 in enumerate(prompt_types):
        for type2 in prompt_types[i+1:]:
            profile1 = np.array(analysis['prompt_type_signatures'][type1]['mlp_profile'])
            profile2 = np.array(analysis['prompt_type_signatures'][type2]['mlp_profile'])

            # Cosine similarity
            similarity = np.dot(profile1, profile2) / (
                np.linalg.norm(profile1) * np.linalg.norm(profile2)
            )

            print(f"  {type1:10s} vs {type2:10s}: {similarity:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile model activations for different prompt types"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mistral7b.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Save config to output directory
    os.makedirs(config.analysis.output_dir, exist_ok=True)
    config_save_path = os.path.join(config.analysis.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    print(f"Saved config to: {config_save_path}")

    # Run profiling
    profile_activations(config)


if __name__ == '__main__':
    main()
