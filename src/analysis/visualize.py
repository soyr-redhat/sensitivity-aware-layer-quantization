"""Visualization utilities for routing analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List


def plot_expert_usage(analysis: Dict, output_dir: str):
    """Plot expert usage distribution across layers.

    Args:
        analysis: Analysis dictionary from analyze_routing_patterns
        output_dir: Directory to save plots
    """
    num_layers = analysis['num_layers']
    expert_usage = analysis['expert_usage_per_layer']

    # Create figure with subplots
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    fig.suptitle('Expert Usage Distribution by Layer', fontsize=16)

    axes = axes.flatten()

    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        percentages = expert_usage[layer_key]['percentages']

        ax = axes[layer_idx]
        bars = ax.bar(range(8), percentages, color='steelblue')

        # Highlight most and least used experts
        max_expert = expert_usage[layer_key]['max_usage_expert']
        min_expert = expert_usage[layer_key]['min_usage_expert']
        bars[max_expert].set_color('darkgreen')
        bars[min_expert].set_color('darkred')

        ax.set_title(f'Layer {layer_idx}', fontsize=10)
        ax.set_xlabel('Expert ID', fontsize=8)
        ax.set_ylabel('Usage %', fontsize=8)
        ax.set_xticks(range(8))
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    output_path = Path(output_dir) / "expert_usage.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved expert usage plot to: {output_path}")
    plt.close()


def plot_routing_heatmap(routing_log_path: str, output_dir: str, num_samples: int = 10):
    """Plot heatmap of routing decisions for sample sequences.

    Args:
        routing_log_path: Path to routing log JSON
        output_dir: Directory to save plots
        num_samples: Number of samples to visualize
    """
    # Load routing data
    with open(routing_log_path, 'r') as f:
        data = json.load(f)

    routing_data = data['routing_data'][:num_samples]
    num_layers = data['num_layers']

    for sample_idx, sample in enumerate(routing_data):
        num_tokens = sample['num_tokens']

        # Create matrix: [num_layers, num_tokens]
        # Each cell shows which experts were selected (we'll show the first expert)
        routing_matrix = np.zeros((num_layers, num_tokens))

        for layer_idx in range(num_layers):
            layer_key = f'layer_{layer_idx}'
            decisions = sample['routing_decisions'][layer_key]

            # Take the first expert from top-2
            for token_idx in range(min(num_tokens, len(decisions))):
                routing_matrix[layer_idx, token_idx] = decisions[token_idx][0]

        # Plot heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(
            routing_matrix,
            cmap='tab10',
            vmin=0,
            vmax=7,
            cbar_kws={'label': 'Expert ID'},
            xticklabels=50,
            yticklabels=True
        )

        plt.title(f'Routing Decisions - Sample {sample_idx}\n(Showing primary expert per token)', fontsize=14)
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Layer', fontsize=12)

        output_path = Path(output_dir) / f"routing_heatmap_sample_{sample_idx}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved routing heatmap to: {output_path}")
        plt.close()


def plot_entropy_by_layer(analysis: Dict, output_dir: str):
    """Plot routing entropy across layers.

    Args:
        analysis: Analysis dictionary
        output_dir: Directory to save plots
    """
    entropy_values = [analysis['entropy_per_layer'][f'layer_{i}']
                     for i in range(analysis['num_layers'])]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(entropy_values)), entropy_values, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=np.mean(entropy_values), color='r', linestyle='--',
                label=f'Mean: {np.mean(entropy_values):.2f}')

    plt.title('Routing Entropy by Layer', fontsize=14)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = Path(output_dir) / "entropy_by_layer.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved entropy plot to: {output_path}")
    plt.close()
