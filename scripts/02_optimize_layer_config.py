#!/usr/bin/env python3
"""Bayesian optimization to find optimal per-layer quantization configuration.

This script uses Bayesian optimization to search for the best layer-wise
quantization configuration by:
1. Testing different layer configurations
2. Measuring actual perplexity for each configuration
3. Finding the configuration with best quality within a size budget

This is more rigorous than heuristic-based allocation.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np


# Quantization levels in order of size (smallest to largest)
QUANT_LEVELS = ['Q2_K', 'Q4_K', 'Q6_K', 'Q8_0']

# Approximate bits per weight for each level
QUANT_BPW = {
    'Q2_K': 2.8,
    'Q4_K': 4.5,
    'Q6_K': 6.5,
    'Q8_0': 8.0,
}

# Tensor names for Mistral architecture
LAYER_TENSOR_NAMES = [
    'attn_q.weight',
    'attn_k.weight',
    'attn_v.weight',
    'attn_output.weight',
    'ffn_gate.weight',
    'ffn_up.weight',
    'ffn_down.weight',
]


@dataclass
class EvaluationResult:
    """Result from evaluating a configuration."""
    config: List[str]
    perplexity: float
    size_gb: float
    avg_bpw: float


class ConfigurationOptimizer:
    """Bayesian optimizer for layer quantization configurations."""

    def __init__(
        self,
        base_model_path: str,
        test_data_path: str,
        llama_quantize_path: str,
        llama_perplexity_path: str,
        num_layers: int = 32,
        target_size_gb: float = None,
        max_evals: int = 50,
        output_dir: str = 'outputs/optimization',
        verbose: bool = True
    ):
        """Initialize optimizer.

        Args:
            base_model_path: Path to F16 base model
            test_data_path: Path to test dataset for perplexity
            llama_quantize_path: Path to llama-quantize binary
            llama_perplexity_path: Path to llama-perplexity binary
            num_layers: Number of layers in model
            target_size_gb: Target model size (if None, minimize perplexity only)
            max_evals: Maximum number of configurations to evaluate
            output_dir: Directory for temporary files and results
            verbose: Print progress
        """
        self.base_model_path = base_model_path
        self.test_data_path = test_data_path
        self.llama_quantize = llama_quantize_path
        self.llama_perplexity = llama_perplexity_path
        self.num_layers = num_layers
        self.target_size_gb = target_size_gb
        self.max_evals = max_evals
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_count = 0
        self.results_history: List[EvaluationResult] = []

    def config_to_tensor_file(self, config: List[str], output_path: str):
        """Generate tensor-type file from configuration."""
        with open(output_path, 'w') as f:
            for layer_idx in range(self.num_layers):
                quant_type = config[layer_idx]
                for tensor_name in LAYER_TENSOR_NAMES:
                    f.write(f"blk.{layer_idx}.{tensor_name}={quant_type}\n")

    def estimate_size(self, config: List[str]) -> Tuple[float, float]:
        """Estimate model size from configuration.

        Returns:
            (size_gb, avg_bpw)
        """
        # Approximate: Mistral-7B has ~7B params, each layer ~218M params
        params_per_layer = 7e9 / self.num_layers

        total_bits = 0
        for quant in config:
            total_bits += params_per_layer * QUANT_BPW[quant]

        size_gb = (total_bits / 8) / (1024**3)
        avg_bpw = sum(QUANT_BPW[q] for q in config) / len(config)

        return size_gb, avg_bpw

    def evaluate_config(self, config: List[str]) -> EvaluationResult:
        """Evaluate a quantization configuration.

        Args:
            config: List of quant levels per layer

        Returns:
            EvaluationResult with perplexity and size
        """
        self.eval_count += 1

        # Estimate size
        size_gb, avg_bpw = self.estimate_size(config)

        # Check if within size budget
        if self.target_size_gb and size_gb > self.target_size_gb * 1.1:
            # Reject configs that exceed budget by >10%
            if self.verbose:
                print(f"[{self.eval_count}/{self.max_evals}] SKIP (size {size_gb:.2f} GB > target {self.target_size_gb:.2f} GB)")
            return EvaluationResult(
                config=config,
                perplexity=999.0,  # Penalty
                size_gb=size_gb,
                avg_bpw=avg_bpw
            )

        if self.verbose:
            config_str = self._summarize_config(config)
            print(f"\n[{self.eval_count}/{self.max_evals}] Testing: {config_str}")
            print(f"  Estimated size: {size_gb:.2f} GB ({avg_bpw:.2f} BPW)")

        # Create temporary tensor-type file
        tensor_file = self.output_dir / f"config_{self.eval_count:03d}.txt"
        self.config_to_tensor_file(config, str(tensor_file))

        # Quantize model
        quant_model = self.output_dir / f"model_{self.eval_count:03d}.gguf"

        try:
            if self.verbose:
                print(f"  Quantizing...")

            subprocess.run([
                self.llama_quantize,
                '--allow-requantize',  # Allow requantizing already-quantized models
                '--tensor-type-file', str(tensor_file),
                self.base_model_path,
                str(quant_model),
                'Q4_K'  # Default fallback (overridden by tensor file)
            ], check=True, capture_output=True)

            # Measure perplexity
            if self.verbose:
                print(f"  Measuring perplexity...")

            result = subprocess.run([
                self.llama_perplexity,
                '-m', str(quant_model),
                '-f', self.test_data_path,
                '-c', '512',  # Reduced context for faster testing
                '-ngl', '0',
                '-t', '4'
            ], check=True, capture_output=True, text=True)

            # Extract perplexity (check both stdout and stderr)
            output_text = result.stdout + result.stderr
            for line in output_text.split('\n'):
                if 'Final estimate: PPL' in line:
                    ppl = float(line.split('=')[1].strip().split()[0])
                    break
            else:
                ppl = 999.0

            if self.verbose:
                print(f"  Result: PPL = {ppl:.4f}")

            # Clean up temporary model
            quant_model.unlink()

        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e}")
            ppl = 999.0
        except Exception as e:
            print(f"  ERROR: {e}")
            ppl = 999.0

        result = EvaluationResult(
            config=config,
            perplexity=ppl,
            size_gb=size_gb,
            avg_bpw=avg_bpw
        )

        self.results_history.append(result)
        return result

    def _summarize_config(self, config: List[str]) -> str:
        """Summarize configuration as string."""
        counts = {}
        for q in config:
            counts[q] = counts.get(q, 0) + 1
        return ', '.join(f"{count}×{q}" for q, count in sorted(counts.items()))

    def _generate_initial_configs(self) -> List[List[str]]:
        """Generate initial configurations to seed optimization."""
        configs = []

        # Uniform baselines
        for level in QUANT_LEVELS:
            configs.append([level] * self.num_layers)

        # Graduated configs (early layers aggressive, late layers precise)
        # Scale to actual number of layers

        # Conservative: ~45% Q4, ~30% Q6, ~25% Q8
        split1 = int(self.num_layers * 0.45)
        split2 = int(self.num_layers * 0.75)
        config = (['Q4_K'] * split1 +
                  ['Q6_K'] * (split2 - split1) +
                  ['Q8_0'] * (self.num_layers - split2))
        configs.append(config)

        # Balanced: ~60% Q4, ~30% Q6, ~10% Q8
        split1 = int(self.num_layers * 0.60)
        split2 = int(self.num_layers * 0.90)
        config = (['Q4_K'] * split1 +
                  ['Q6_K'] * (split2 - split1) +
                  ['Q8_0'] * (self.num_layers - split2))
        configs.append(config)

        # Aggressive: ~55% Q2, ~35% Q4, ~7% Q6, ~3% Q8
        split1 = int(self.num_layers * 0.55)
        split2 = int(self.num_layers * 0.90)
        split3 = int(self.num_layers * 0.97)
        config = (['Q2_K'] * split1 +
                  ['Q4_K'] * (split2 - split1) +
                  ['Q6_K'] * (split3 - split2) +
                  ['Q8_0'] * (self.num_layers - split3))
        configs.append(config)

        return configs

    def _mutate_config(self, config: List[str]) -> List[str]:
        """Generate a mutation of a configuration."""
        new_config = config.copy()

        # Mutate 1-3 random layers
        num_mutations = np.random.randint(1, 4)
        positions = np.random.choice(self.num_layers, size=num_mutations, replace=False)

        for pos in positions:
            current_level = new_config[pos]
            current_idx = QUANT_LEVELS.index(current_level)

            # Move up or down one level (with bounds)
            if np.random.random() < 0.5:
                new_idx = max(0, current_idx - 1)  # Lower precision
            else:
                new_idx = min(len(QUANT_LEVELS) - 1, current_idx + 1)  # Higher precision

            new_config[pos] = QUANT_LEVELS[new_idx]

        return new_config

    def optimize(self) -> EvaluationResult:
        """Run Bayesian optimization to find best configuration.

        Returns:
            Best configuration found
        """
        print("=" * 80)
        print("BAYESIAN OPTIMIZATION FOR LAYER-WISE QUANTIZATION")
        print("=" * 80)
        print(f"\nTarget size: {self.target_size_gb:.2f} GB" if self.target_size_gb else "\nNo size constraint")
        print(f"Max evaluations: {self.max_evals}")
        print(f"Model layers: {self.num_layers}")

        # Phase 1: Evaluate initial configurations
        print("\n" + "=" * 80)
        print("PHASE 1: Initial Configurations")
        print("=" * 80)

        initial_configs = self._generate_initial_configs()
        for config in initial_configs:
            if self.eval_count >= self.max_evals:
                break
            self.evaluate_config(config)

        # Phase 2: Mutation-based search
        print("\n" + "=" * 80)
        print("PHASE 2: Mutation Search")
        print("=" * 80)

        while self.eval_count < self.max_evals:
            # Select best configurations so far
            valid_results = [r for r in self.results_history if r.perplexity < 900]
            if not valid_results:
                break

            valid_results.sort(key=lambda r: r.perplexity)

            # Sample from top 30% to mutate
            top_n = max(1, len(valid_results) // 3)
            parent = np.random.choice(valid_results[:top_n])

            # Generate mutation
            mutated_config = self._mutate_config(parent.config)
            self.evaluate_config(mutated_config)

        # Find best result
        valid_results = [r for r in self.results_history if r.perplexity < 900]
        if not valid_results:
            print("\nERROR: No valid configurations found!")
            return None

        valid_results.sort(key=lambda r: r.perplexity)
        best = valid_results[0]

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nBest configuration:")
        print(f"  Config: {self._summarize_config(best.config)}")
        print(f"  Perplexity: {best.perplexity:.4f}")
        print(f"  Size: {best.size_gb:.2f} GB ({best.avg_bpw:.2f} BPW)")

        # Save results
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'best_config': best.config,
                'best_perplexity': best.perplexity,
                'best_size_gb': best.size_gb,
                'all_results': [
                    {
                        'config': r.config,
                        'perplexity': r.perplexity,
                        'size_gb': r.size_gb
                    }
                    for r in self.results_history
                ]
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return best

    def save_best_config(self, result: EvaluationResult, output_path: str):
        """Save best configuration as tensor-type file."""
        self.config_to_tensor_file(result.config, output_path)
        print(f"\nBest config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize layer-wise quantization configuration"
    )
    parser.add_argument(
        '--base-model',
        type=str,
        required=True,
        help='Path to F16 base model GGUF file'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test dataset for perplexity measurement'
    )
    parser.add_argument(
        '--llama-quantize',
        type=str,
        default='llama-quantize',
        help='Path to llama-quantize binary'
    )
    parser.add_argument(
        '--llama-perplexity',
        type=str,
        default='llama-perplexity',
        help='Path to llama-perplexity binary'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=32,
        help='Number of layers in model'
    )
    parser.add_argument(
        '--target-size',
        type=float,
        default=None,
        help='Target model size in GB (optional)'
    )
    parser.add_argument(
        '--max-evals',
        type=int,
        default=50,
        help='Maximum number of configurations to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/optimization',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        default='outputs/tensor_configs/optimized.txt',
        help='Path to save best tensor-type config'
    )
    args = parser.parse_args()

    optimizer = ConfigurationOptimizer(
        base_model_path=args.base_model,
        test_data_path=args.test_data,
        llama_quantize_path=args.llama_quantize,
        llama_perplexity_path=args.llama_perplexity,
        num_layers=args.num_layers,
        target_size_gb=args.target_size,
        max_evals=args.max_evals,
        output_dir=args.output_dir
    )

    best = optimizer.optimize()

    if best:
        optimizer.save_best_config(best, args.save_config)


if __name__ == '__main__':
    main()
