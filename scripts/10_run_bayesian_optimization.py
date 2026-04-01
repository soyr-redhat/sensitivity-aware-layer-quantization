#!/usr/bin/env python3
"""Run Bayesian optimization to find optimal layer-wise quantization configs.

This script uses constrained Bayesian optimization to search for configurations
that optimize the perplexity/size tradeoff.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

from src.config import Config
from src.data.prompts import load_prompts_dataset
from src.model.loader import load_tokenizer
from src.quantization.constraints import load_constraints
from src.quantization.optimizer import QuantizationEvaluator, create_initial_configs


def setup_search_space(constraint_matrix):
    """Create search space from constraints.

    Args:
        constraint_matrix: Allowed quants per layer

    Returns:
        List of Categorical dimensions
    """
    dimensions = []
    for layer_idx, constraints in enumerate(constraint_matrix):
        dim = Categorical(
            categories=constraints,
            name=f'layer_{layer_idx}'
        )
        dimensions.append(dim)

    return dimensions


def run_optimization(
    evaluator: QuantizationEvaluator,
    constraint_matrix,
    n_calls: int = 50,
    n_initial_points: int = 10,
    random_state: int = 42,
    output_dir: str = '/workspace/outputs/optimization'
):
    """Run Bayesian optimization.

    Args:
        evaluator: QuantizationEvaluator instance
        constraint_matrix: Per-layer constraints
        n_calls: Number of evaluations
        n_initial_points: Random samples before GP
        random_state: Random seed
        output_dir: Output directory

    Returns:
        Optimization result
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up search space
    print("\n" + "="*80)
    print("SETTING UP BAYESIAN OPTIMIZATION")
    print("="*80)

    dimensions = setup_search_space(constraint_matrix)
    num_layers = len(dimensions)

    print(f"\nSearch space:")
    print(f"  Dimensions: {num_layers}")
    print(f"  Total configs: {np.prod([len(d.categories) for d in dimensions]):,}")

    # Define objective function
    @use_named_args(dimensions)
    def objective(**params):
        """Objective: minimize perplexity."""
        # Extract config from params (ordered by layer)
        config = [params[f'layer_{i}'] for i in range(num_layers)]

        # Evaluate
        result = evaluator.evaluate(config)

        # Return perplexity (we want to minimize)
        return result.perplexity

    # Get initial points
    print("\nGenerating initial points...")
    initial_configs = create_initial_configs(num_layers, constraint_matrix)

    # Convert to the format skopt expects
    x0 = []
    for config in initial_configs[:n_initial_points]:
        x0.append(config)

    print(f"  Using {len(x0)} initial points")

    # Run optimization
    print("\n" + "="*80)
    print(f"RUNNING BAYESIAN OPTIMIZATION")
    print(f"  n_calls: {n_calls}")
    print(f"  n_initial_points: {n_initial_points}")
    print("="*80)

    start_time = time.time()

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        x0=x0,
        random_state=random_state,
        verbose=True,
        n_jobs=1  # Sequential to avoid OOM
    )

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Evaluations: {len(result.func_vals)}")
    print(f"  Best perplexity: {result.fun:.4f}")

    # Save results
    results_data = {
        'best_config': result.x,
        'best_perplexity': float(result.fun),
        'n_calls': n_calls,
        'total_time_seconds': total_time,
        'all_configs': [list(x) for x in result.x_iters],
        'all_perplexities': [float(y) for y in result.func_vals],
    }

    results_path = os.path.join(output_dir, 'optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nSaved results to: {results_path}")

    # Save evaluation history
    history_path = os.path.join(output_dir, 'evaluation_history.json')
    evaluator.save_history(history_path)

    return result


def analyze_pareto_frontier(
    evaluation_history,
    baseline_ppl: float,
    output_dir: str
):
    """Find Pareto optimal configurations.

    Args:
        evaluation_history: List of EvaluationResult objects
        baseline_ppl: Baseline perplexity for comparison
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("PARETO FRONTIER ANALYSIS")
    print("="*80)

    # Note: Since we can't measure real memory with HF, we'll find configs
    # that minimize perplexity at different config "sizes" (# of aggressive quants)

    # Calculate aggressiveness score for each config
    quant_scores = {'Q2_K': 1, 'Q4_K_M': 2, 'Q6_K': 3, 'Q8_0': 4}

    configs_with_scores = []
    for result in evaluation_history:
        # Calculate average quantization level
        avg_quant = np.mean([quant_scores[q] for q in result.config])

        configs_with_scores.append({
            'config': result.config,
            'perplexity': result.perplexity,
            'avg_quant_level': avg_quant,
            'ppl_delta_pct': ((result.perplexity / baseline_ppl) - 1) * 100,
            'eval_id': result.eval_id
        })

    # Sort by avg quant level
    configs_with_scores.sort(key=lambda x: x['avg_quant_level'])

    # Find Pareto frontier
    # (configs where no other config is both smaller AND better quality)
    pareto_frontier = []

    for i, cfg in enumerate(configs_with_scores):
        is_pareto = True
        for other_cfg in configs_with_scores:
            # Check if other config dominates this one
            if (other_cfg['avg_quant_level'] <= cfg['avg_quant_level'] and
                other_cfg['perplexity'] < cfg['perplexity'] and
                (other_cfg['avg_quant_level'] < cfg['avg_quant_level'] or
                 other_cfg['perplexity'] < cfg['perplexity'])):
                is_pareto = False
                break

        if is_pareto:
            pareto_frontier.append(cfg)

    print(f"\nFound {len(pareto_frontier)} Pareto optimal configurations:")
    print(f"\n{'#':<4} {'Eval ID':<10} {'Avg Quant':<12} {'PPL':<10} {'PPL Δ%':<10}")
    print("-"*50)

    for i, cfg in enumerate(pareto_frontier):
        print(f"{i+1:<4} {cfg['eval_id']:<10} "
              f"{cfg['avg_quant_level']:>10.2f}  "
              f"{cfg['perplexity']:>8.4f}  "
              f"{cfg['ppl_delta_pct']:>8.2f}%")

    # Save Pareto frontier
    pareto_path = os.path.join(output_dir, 'pareto_frontier.json')
    with open(pareto_path, 'w') as f:
        json.dump(pareto_frontier, f, indent=2)

    print(f"\nSaved Pareto frontier to: {pareto_path}")

    return pareto_frontier


def main():
    """Main optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for layer-wise quantization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mistral7b.yaml',
        help='Model config'
    )
    parser.add_argument(
        '--constraints',
        type=str,
        default='/workspace/outputs/analysis/quant_constraints.json',
        help='Quantization constraints'
    )
    parser.add_argument(
        '--n-calls',
        type=int,
        default=100,
        help='Number of optimization iterations'
    )
    parser.add_argument(
        '--n-initial',
        type=int,
        default=10,
        help='Number of random initial points'
    )
    parser.add_argument(
        '--num-val-samples',
        type=int,
        default=50,
        help='Number of validation samples for perplexity'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/outputs/optimization',
        help='Output directory'
    )
    parser.add_argument(
        '--baseline-ppl',
        type=float,
        default=10.7742,
        help='Baseline perplexity for comparison'
    )
    args = parser.parse_args()

    print("="*80)
    print("BAYESIAN OPTIMIZATION FOR LAYER-WISE QUANTIZATION")
    print("="*80)

    # Load config
    config = Config.from_yaml(args.config)

    # Load validation data
    print("\n[1/5] Loading validation data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)

    # Shuffle and take subset
    import random
    random.seed(42)
    random.shuffle(test_texts)
    test_texts = test_texts[:args.num_val_samples]
    print(f"  Using {len(test_texts)} validation samples")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = load_tokenizer(config.model)

    # Load constraints
    print("\n[3/5] Loading constraints...")
    constraint_matrix, sensitivity_scores = load_constraints(args.constraints)
    num_layers = len(constraint_matrix)
    print(f"  Loaded constraints for {num_layers} layers")

    # Create evaluator
    print("\n[4/5] Creating evaluator...")
    evaluator = QuantizationEvaluator(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        tokenizer=tokenizer,
        validation_texts=test_texts,
        max_length=config.data.max_length,
        device_map=config.model.device_map,
        verbose=True
    )

    # Run optimization
    print("\n[5/5] Running optimization...")
    result = run_optimization(
        evaluator=evaluator,
        constraint_matrix=constraint_matrix,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        output_dir=args.output_dir
    )

    # Analyze results
    pareto_frontier = analyze_pareto_frontier(
        evaluation_history=evaluator.evaluation_history,
        baseline_ppl=args.baseline_ppl,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Best config found: {evaluator._summarize_config(result.x)}")
    print(f"Best perplexity: {result.fun:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Review Pareto frontier configurations")
    print(f"  2. Convert top configs to GGUF format")
    print(f"  3. Validate real memory savings with llama.cpp")


if __name__ == '__main__':
    main()
