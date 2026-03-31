"""Routing pattern analysis functions."""

import numpy as np
from scipy import stats
from typing import Dict, List
import json
from pathlib import Path


def compute_routing_entropy(routing_decisions: np.ndarray, num_experts: int = 8) -> float:
    """Compute entropy of expert routing distribution.

    Higher entropy means more uniform distribution across experts.
    Lower entropy means routing is concentrated on fewer experts.

    Args:
        routing_decisions: Array of expert selections [num_tokens, top_k]
        num_experts: Total number of experts

    Returns:
        Entropy value (in bits)
    """
    # Count how often each expert was selected
    expert_counts = np.bincount(routing_decisions.flatten(), minlength=num_experts)

    # Convert to probability distribution
    expert_probs = expert_counts / expert_counts.sum()

    # Compute entropy (using log2 for bits)
    # Filter out zeros to avoid log(0)
    expert_probs = expert_probs[expert_probs > 0]
    entropy = -np.sum(expert_probs * np.log2(expert_probs))

    return entropy


def analyze_routing_stability(routing_data: List[Dict]) -> Dict:
    """Analyze how stable routing is within sequences.

    Args:
        routing_data: List of routing data dictionaries

    Returns:
        Dictionary with stability metrics
    """
    stability_metrics = {
        'per_layer': {},
        'overall': {}
    }

    # Get number of layers from first sample
    num_layers = len([k for k in routing_data[0]['routing_decisions'].keys() if k.startswith('layer_')])

    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'

        # Track how often the same expert stays active across consecutive tokens
        consecutive_matches = []
        expert_switches = []

        for sample in routing_data:
            decisions = sample['routing_decisions'][layer_key]

            # For each position, check if any expert from prev position is in current
            for i in range(1, len(decisions)):
                prev_experts = set(decisions[i-1])
                curr_experts = set(decisions[i])

                # How many experts stayed the same?
                overlap = len(prev_experts & curr_experts)
                consecutive_matches.append(overlap)

                # Did any expert change?
                expert_switches.append(1 if overlap < 2 else 0)

        stability_metrics['per_layer'][layer_key] = {
            'avg_consecutive_overlap': np.mean(consecutive_matches),
            'switch_rate': np.mean(expert_switches),
            'stability_score': 1.0 - np.mean(expert_switches)
        }

    # Overall metrics
    all_scores = [m['stability_score'] for m in stability_metrics['per_layer'].values()]
    stability_metrics['overall'] = {
        'avg_stability': np.mean(all_scores),
        'min_stability': np.min(all_scores),
        'max_stability': np.max(all_scores)
    }

    return stability_metrics


def analyze_routing_patterns(routing_log_path: str, output_dir: str) -> Dict:
    """Analyze routing patterns from logged data.

    Args:
        routing_log_path: Path to routing log JSON file
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing routing patterns from: {routing_log_path}")

    # Load routing data
    with open(routing_log_path, 'r') as f:
        data = json.load(f)

    routing_data = data['routing_data']
    num_layers = data['num_layers']

    analysis = {
        'entropy_per_layer': {},
        'expert_usage_per_layer': {},
        'stability': analyze_routing_stability(routing_data),
        'num_samples': len(routing_data),
        'num_layers': num_layers
    }

    # Analyze each layer
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'

        # Collect all routing decisions for this layer
        all_decisions = []
        for sample in routing_data:
            all_decisions.append(sample['routing_decisions'][layer_key])

        all_decisions = np.concatenate(all_decisions, axis=0)

        # Compute entropy
        entropy = compute_routing_entropy(all_decisions)
        analysis['entropy_per_layer'][layer_key] = entropy

        # Expert usage distribution
        expert_counts = np.bincount(all_decisions.flatten(), minlength=8)
        expert_usage = expert_counts / expert_counts.sum()

        analysis['expert_usage_per_layer'][layer_key] = {
            'counts': expert_counts.tolist(),
            'percentages': (expert_usage * 100).tolist(),
            'max_usage_expert': int(np.argmax(expert_usage)),
            'min_usage_expert': int(np.argmin(expert_usage)),
            'usage_variance': float(np.var(expert_usage))
        }

    # Save analysis results
    output_path = Path(output_dir) / "routing_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to: {output_path}")

    # Print summary
    print("\n=== Routing Analysis Summary ===")
    print(f"Samples analyzed: {analysis['num_samples']}")
    print(f"Layers: {analysis['num_layers']}")
    print(f"\nAverage entropy per layer: {np.mean(list(analysis['entropy_per_layer'].values())):.3f} bits")
    print(f"Overall stability score: {analysis['stability']['overall']['avg_stability']:.3f}")

    return analysis
