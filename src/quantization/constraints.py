#!/usr/bin/env python3
"""Activation-based constraint generation for layer-wise quantization.

This module analyzes activation statistics to determine which quantization
levels are safe for each layer.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


# GGUF quantization levels (in order of precision)
GGUF_QUANT_LEVELS = ['Q2_K', 'Q4_K_M', 'Q6_K', 'Q8_0']

# Mapping for fallback when GGUF not available
QUANT_LEVEL_MAPPING = {
    'Q2_K': 'INT4',    # ~2 bits
    'Q4_K_M': 'INT4',  # ~4.5 bits
    'Q6_K': 'INT8',    # ~6.5 bits
    'Q8_0': 'INT8',    # 8 bits
}


def calculate_layer_sensitivity(
    activation_analysis: dict,
    prompt_type: str = 'code'
) -> List[float]:
    """Calculate sensitivity score for each layer.

    Higher score = more sensitive = needs more precision.

    Args:
        activation_analysis: Activation analysis results
        prompt_type: Which prompt type to use for analysis

    Returns:
        List of sensitivity scores (0.0 to 1.0) per layer
    """
    signatures = activation_analysis['prompt_type_signatures'][prompt_type]

    attn_profile = signatures['attn_profile']
    mlp_profile = signatures['mlp_profile']

    num_layers = len(attn_profile)
    sensitivity_scores = []

    for layer_idx in range(num_layers):
        # Our activation analysis stores simple mean values per layer
        attn_activation = attn_profile[layer_idx]
        mlp_activation = mlp_profile[layer_idx]

        # Multiple factors contribute to sensitivity:
        # 1. Activation magnitude (higher = more important)
        activation_magnitude = (attn_activation + mlp_activation) / 2

        # 2. Later layers typically more important (higher weights)
        layer_position_weight = (layer_idx + 1) / num_layers

        # Combine factors
        # Weight activation magnitude heavily (0.8) and position moderately (0.2)
        sensitivity = (
            0.8 * activation_magnitude +
            0.2 * layer_position_weight * max(attn_activation, mlp_activation)
        )

        sensitivity_scores.append(sensitivity)

    # Normalize to 0-1 range
    if max(sensitivity_scores) > 0:
        sensitivity_scores = [
            s / max(sensitivity_scores) for s in sensitivity_scores
        ]

    return sensitivity_scores


def get_layer_constraints(
    sensitivity_score: float,
    aggressive_threshold: float = 0.3,
    moderate_threshold: float = 0.6,
    conservative_threshold: float = 0.85
) -> List[str]:
    """Get allowed quantization levels for a layer based on sensitivity.

    Args:
        sensitivity_score: Layer sensitivity (0.0 to 1.0)
        aggressive_threshold: Below this = allow aggressive quantization
        moderate_threshold: Below this = allow moderate quantization
        conservative_threshold: Below this = allow Q6_K

    Returns:
        List of allowed GGUF quantization levels
    """
    if sensitivity_score < aggressive_threshold:
        # Low sensitivity: can use any quantization
        return ['Q2_K', 'Q4_K_M', 'Q6_K', 'Q8_0']

    elif sensitivity_score < moderate_threshold:
        # Medium sensitivity: avoid aggressive quantization
        return ['Q4_K_M', 'Q6_K', 'Q8_0']

    elif sensitivity_score < conservative_threshold:
        # High sensitivity: conservative quantization only
        return ['Q6_K', 'Q8_0']

    else:
        # Critical layer: minimal quantization
        return ['Q8_0']


def create_constraint_matrix(
    activation_analysis: dict,
    prompt_type: str = 'code',
    aggressive_threshold: float = 0.3,
    moderate_threshold: float = 0.6,
    conservative_threshold: float = 0.85,
    verbose: bool = True
) -> Tuple[List[List[str]], List[float]]:
    """Create per-layer quantization constraints from activation analysis.

    Args:
        activation_analysis: Activation analysis results
        prompt_type: Which prompt type to analyze
        aggressive_threshold: Threshold for aggressive quantization
        moderate_threshold: Threshold for moderate quantization
        conservative_threshold: Threshold for conservative quantization
        verbose: Print analysis

    Returns:
        Tuple of (constraint_matrix, sensitivity_scores)
        - constraint_matrix: List of allowed quant levels per layer
        - sensitivity_scores: Sensitivity score per layer
    """
    # Calculate sensitivity for each layer
    sensitivity_scores = calculate_layer_sensitivity(
        activation_analysis,
        prompt_type
    )

    num_layers = len(sensitivity_scores)

    # Create constraints for each layer
    constraint_matrix = []
    for layer_idx, sensitivity in enumerate(sensitivity_scores):
        constraints = get_layer_constraints(
            sensitivity,
            aggressive_threshold,
            moderate_threshold,
            conservative_threshold
        )
        constraint_matrix.append(constraints)

    if verbose:
        print("\n" + "="*80)
        print("ACTIVATION-BASED QUANTIZATION CONSTRAINTS")
        print("="*80)
        print(f"\nPrompt type: {prompt_type}")
        print(f"Thresholds: aggressive={aggressive_threshold}, "
              f"moderate={moderate_threshold}, "
              f"conservative={conservative_threshold}")

        print(f"\n{'Layer':<8} {'Sensitivity':<12} {'Allowed Quantizations':<30} {'# Options':<10}")
        print("-"*70)

        for layer_idx, (sensitivity, constraints) in enumerate(
            zip(sensitivity_scores, constraint_matrix)
        ):
            constraint_str = ', '.join(constraints)
            print(f"{layer_idx:<8} {sensitivity:>10.4f}  "
                  f"{constraint_str:<30} {len(constraints):<10}")

        # Print summary statistics
        print("\n" + "="*80)
        print("CONSTRAINT SUMMARY")
        print("="*80)

        constraint_counts = {}
        for constraints in constraint_matrix:
            num_options = len(constraints)
            constraint_counts[num_options] = constraint_counts.get(num_options, 0) + 1

        print(f"\nLayers by flexibility:")
        for num_options in sorted(constraint_counts.keys(), reverse=True):
            count = constraint_counts[num_options]
            print(f"  {num_options} options: {count} layers")

        # Calculate search space size
        search_space_size = 1
        for constraints in constraint_matrix:
            search_space_size *= len(constraints)

        print(f"\nSearch space size: {search_space_size:,} configurations")
        print(f"  (vs unconstrained: {len(GGUF_QUANT_LEVELS)**num_layers:,})")
        print(f"  Reduction: {len(GGUF_QUANT_LEVELS)**num_layers / search_space_size:.1f}x")

    return constraint_matrix, sensitivity_scores


def save_constraints(
    constraint_matrix: List[List[str]],
    sensitivity_scores: List[float],
    output_path: str,
    metadata: dict = None
):
    """Save constraint matrix and sensitivity scores.

    Args:
        constraint_matrix: Per-layer constraints
        sensitivity_scores: Per-layer sensitivity scores
        output_path: Path to save JSON
        metadata: Optional metadata
    """
    output = {
        'constraint_matrix': constraint_matrix,
        'sensitivity_scores': sensitivity_scores,
        'num_layers': len(constraint_matrix),
        'quant_levels': GGUF_QUANT_LEVELS,
        'metadata': metadata or {}
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved constraints to: {output_path}")


def load_constraints(constraint_path: str) -> Tuple[List[List[str]], List[float]]:
    """Load constraint matrix from file.

    Args:
        constraint_path: Path to constraint JSON

    Returns:
        Tuple of (constraint_matrix, sensitivity_scores)
    """
    with open(constraint_path, 'r') as f:
        data = json.load(f)

    return data['constraint_matrix'], data['sensitivity_scores']


if __name__ == '__main__':
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate quantization constraints from activation analysis"
    )
    parser.add_argument(
        '--activation-analysis',
        type=str,
        required=True,
        help='Path to activation analysis JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/analysis/quant_constraints.json',
        help='Output path for constraints'
    )
    parser.add_argument(
        '--prompt-type',
        type=str,
        default='code',
        choices=['code', 'math', 'creative', 'factual', 'reasoning'],
        help='Prompt type to analyze'
    )
    parser.add_argument(
        '--aggressive-threshold',
        type=float,
        default=0.3,
        help='Sensitivity threshold for aggressive quantization'
    )
    parser.add_argument(
        '--moderate-threshold',
        type=float,
        default=0.6,
        help='Sensitivity threshold for moderate quantization'
    )
    parser.add_argument(
        '--conservative-threshold',
        type=float,
        default=0.85,
        help='Sensitivity threshold for conservative quantization'
    )
    args = parser.parse_args()

    # Load activation analysis
    with open(args.activation_analysis, 'r') as f:
        activation_analysis = json.load(f)

    # Create constraints
    constraint_matrix, sensitivity_scores = create_constraint_matrix(
        activation_analysis,
        prompt_type=args.prompt_type,
        aggressive_threshold=args.aggressive_threshold,
        moderate_threshold=args.moderate_threshold,
        conservative_threshold=args.conservative_threshold,
        verbose=True
    )

    # Save
    save_constraints(
        constraint_matrix,
        sensitivity_scores,
        args.output,
        metadata={
            'prompt_type': args.prompt_type,
            'thresholds': {
                'aggressive': args.aggressive_threshold,
                'moderate': args.moderate_threshold,
                'conservative': args.conservative_threshold
            }
        }
    )
