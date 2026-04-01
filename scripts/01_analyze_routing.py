#!/usr/bin/env python3
"""
Phase 1: Routing Analysis

This script analyzes routing patterns in Mixtral-8x7B by:
1. Loading the model and dataset
2. Collecting routing decisions across diverse prompts
3. Analyzing patterns (entropy, stability, expert usage)
4. Generating visualizations

Usage:
    python scripts/01_analyze_routing.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import Config
from src.model import load_model_and_tokenizer, collect_routing_decisions
from src.data import load_dataset_samples
from src.analysis import (
    analyze_routing_patterns,
    plot_expert_usage,
    plot_routing_heatmap,
    plot_entropy_by_layer
)


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE routing patterns")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override number of samples to analyze"
    )
    args = parser.parse_args()

    # Load configuration
    print("=" * 80)
    print("PHASE 1: ROUTING ANALYSIS")
    print("=" * 80)

    config = Config.from_yaml(args.config)

    # Override num_samples if provided
    if args.num_samples:
        config.data.num_samples = args.num_samples

    # Create output directories
    Path(config.routing_analysis.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.analysis.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.analysis.plot_dir).mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config.save_yaml(f"{config.analysis.output_dir}/config.yaml")

    # Step 1: Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model)

    # Step 2: Load dataset
    print("\n[2/5] Loading dataset samples...")
    texts = load_dataset_samples(config.data)

    # Step 3: Collect routing decisions
    print("\n[3/5] Collecting routing decisions...")
    print(f"Processing {len(texts)} samples...")

    routing_results = collect_routing_decisions(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=config.data.max_length
    )

    # Save routing log
    routing_log_path = f"{config.routing_analysis.output_dir}/routing_log.json"
    with open(routing_log_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(routing_results, f, indent=2, default=lambda x: x.tolist())

    print(f"Saved routing log to: {routing_log_path}")

    # Step 4: Analyze patterns
    print("\n[4/5] Analyzing routing patterns...")
    analysis = analyze_routing_patterns(
        routing_log_path=routing_log_path,
        output_dir=config.analysis.output_dir
    )

    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")

    plot_expert_usage(
        analysis=analysis,
        output_dir=config.analysis.plot_dir
    )

    plot_entropy_by_layer(
        analysis=analysis,
        output_dir=config.analysis.plot_dir
    )

    plot_routing_heatmap(
        routing_log_path=routing_log_path,
        output_dir=config.analysis.plot_dir,
        num_samples=min(10, len(texts))
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Routing log: {routing_log_path}")
    print(f"  - Analysis: {config.analysis.output_dir}/routing_analysis.json")
    print(f"  - Plots: {config.analysis.plot_dir}/")
    print("\nNext steps:")
    print("  - Review the plots to understand routing patterns")
    print("  - Check if routing is predictable enough for eagle model")
    print("  - Move to Phase 2: Training the predictor")


if __name__ == "__main__":
    main()
