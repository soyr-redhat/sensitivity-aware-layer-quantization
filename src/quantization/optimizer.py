#!/usr/bin/env python3
"""Bayesian optimization for layer-wise quantization.

This module implements the evaluation and optimization framework for finding
optimal per-layer quantization configurations.
"""

import os
import gc
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.quantization import (
    QuantizationStrategy,
    LayerQuantConfig,
    get_model_memory_footprint,
)
from src.quantization.apply_quant import apply_layer_wise_quantization_real
from src.evaluation import calculate_perplexity


# Mapping from GGUF quant levels to our QuantizationStrategy
GGUF_TO_STRATEGY = {
    'Q2_K': QuantizationStrategy.INT4,     # ~2 bits
    'Q4_K_M': QuantizationStrategy.INT4,   # ~4.5 bits (use INT4 as proxy)
    'Q6_K': QuantizationStrategy.INT8,     # ~6.5 bits
    'Q8_0': QuantizationStrategy.INT8,     # 8 bits
}


@dataclass
class EvaluationResult:
    """Results from evaluating a quantization configuration."""
    config: List[str]  # GGUF quant levels per layer
    perplexity: float
    model_size_gb: float
    eval_time_seconds: float
    eval_id: int


class QuantizationEvaluator:
    """Evaluates quantization configurations."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        tokenizer,
        validation_texts: List[str],
        max_length: int = 512,
        device_map: str = "auto",
        verbose: bool = True
    ):
        """Initialize evaluator.

        Args:
            model_name: HuggingFace model name
            cache_dir: Model cache directory
            tokenizer: Tokenizer instance
            validation_texts: Validation texts for perplexity
            max_length: Max sequence length
            device_map: Device mapping
            verbose: Print progress
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.validation_texts = validation_texts
        self.max_length = max_length
        self.device_map = device_map
        self.verbose = verbose

        self.eval_count = 0
        self.evaluation_history = []

    def config_to_layer_configs(
        self,
        gguf_config: List[str]
    ) -> List[LayerQuantConfig]:
        """Convert GGUF config to LayerQuantConfig list.

        Args:
            gguf_config: List of GGUF quant levels per layer

        Returns:
            List of LayerQuantConfig objects
        """
        layer_configs = []
        for layer_idx, gguf_quant in enumerate(gguf_config):
            strategy = GGUF_TO_STRATEGY[gguf_quant]
            layer_configs.append(
                LayerQuantConfig(
                    layer_idx=layer_idx,
                    attention_quant=strategy,
                    mlp_quant=strategy
                )
            )
        return layer_configs

    def evaluate(self, gguf_config: List[str]) -> EvaluationResult:
        """Evaluate a quantization configuration.

        Args:
            gguf_config: List of GGUF quant levels (one per layer)

        Returns:
            EvaluationResult with perplexity and size
        """
        start_time = time.time()
        self.eval_count += 1

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EVALUATION #{self.eval_count}")
            print(f"{'='*80}")
            config_summary = self._summarize_config(gguf_config)
            print(f"Config: {config_summary}")

        # Clean memory before loading
        gc.collect()
        torch.cuda.empty_cache()

        # Load fresh model
        if self.verbose:
            print("\n[1/4] Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()

        # Apply quantization
        if self.verbose:
            print("[2/4] Applying quantization...")

        layer_configs = self.config_to_layer_configs(gguf_config)
        apply_layer_wise_quantization_real(
            model=model,
            configs=layer_configs,
            verbose=False
        )

        # Measure memory
        if self.verbose:
            print("[3/4] Measuring memory...")

        memory_stats = get_model_memory_footprint(model)
        model_size_gb = memory_stats['total_memory_gb']

        if self.verbose:
            print(f"  Model size: {model_size_gb:.2f} GB")

        # Measure perplexity
        if self.verbose:
            print("[4/4] Measuring perplexity...")

        metrics = calculate_perplexity(
            model=model,
            tokenizer=self.tokenizer,
            texts=self.validation_texts,
            max_length=self.max_length
        )
        perplexity = metrics['perplexity']

        if self.verbose:
            print(f"  Perplexity: {perplexity:.4f}")

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        eval_time = time.time() - start_time

        if self.verbose:
            print(f"\n✓ Evaluation complete in {eval_time:.1f}s")

        # Create result
        result = EvaluationResult(
            config=gguf_config.copy(),
            perplexity=perplexity,
            model_size_gb=model_size_gb,
            eval_time_seconds=eval_time,
            eval_id=self.eval_count
        )

        self.evaluation_history.append(result)

        return result

    def _summarize_config(self, config: List[str]) -> str:
        """Create a compact summary of a config.

        Args:
            config: GGUF quant levels per layer

        Returns:
            Summary string
        """
        # Count each quant type
        counts = {}
        for quant in config:
            counts[quant] = counts.get(quant, 0) + 1

        parts = [f"{count}×{quant}" for quant, count in sorted(counts.items())]
        return ", ".join(parts)

    def save_history(self, output_path: str):
        """Save evaluation history.

        Args:
            output_path: Path to save JSON
        """
        history_data = []
        for result in self.evaluation_history:
            history_data.append({
                'eval_id': result.eval_id,
                'config': result.config,
                'config_summary': self._summarize_config(result.config),
                'perplexity': result.perplexity,
                'model_size_gb': result.model_size_gb,
                'eval_time_seconds': result.eval_time_seconds,
            })

        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"\nSaved {len(history_data)} evaluations to: {output_path}")


def create_initial_configs(
    num_layers: int,
    constraint_matrix: List[List[str]]
) -> List[List[str]]:
    """Create good initial configurations to seed optimization.

    Args:
        num_layers: Number of layers
        constraint_matrix: Allowed quants per layer

    Returns:
        List of initial configs to try
    """
    configs = []

    # Config 1: All most aggressive (smallest)
    aggressive = []
    for constraints in constraint_matrix:
        aggressive.append(constraints[0])  # First option (most aggressive)
    configs.append(aggressive)

    # Config 2: All most conservative (best quality)
    conservative = []
    for constraints in constraint_matrix:
        conservative.append(constraints[-1])  # Last option (most conservative)
    configs.append(conservative)

    # Config 3: Balanced (middle option when available)
    balanced = []
    for constraints in constraint_matrix:
        mid_idx = len(constraints) // 2
        balanced.append(constraints[mid_idx])
    configs.append(balanced)

    # Config 4: Gradient (aggressive early, conservative late)
    gradient = []
    for layer_idx, constraints in enumerate(constraint_matrix):
        # Linearly transition from aggressive to conservative
        progress = layer_idx / (num_layers - 1) if num_layers > 1 else 0
        idx = int(progress * (len(constraints) - 1))
        gradient.append(constraints[idx])
    configs.append(gradient)

    return configs


if __name__ == '__main__':
    """Test evaluation framework."""
    import argparse
    from src.config import Config
    from src.data.prompts import load_prompts_dataset
    from src.quantization.constraints import load_constraints

    parser = argparse.ArgumentParser(
        description="Test quantization evaluation framework"
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
        '--num-samples',
        type=int,
        default=20,
        help='Number of validation samples'
    )
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Load validation data
    print("Loading validation data...")
    prompt_datasets = load_prompts_dataset(config)
    test_texts = []
    for prompts in prompt_datasets.values():
        test_texts.extend(prompts)
    test_texts = test_texts[:args.num_samples]
    print(f"Using {len(test_texts)} validation samples")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load constraints
    print("\nLoading constraints...")
    constraint_matrix, _ = load_constraints(args.constraints)
    num_layers = len(constraint_matrix)
    print(f"Loaded constraints for {num_layers} layers")

    # Create evaluator
    evaluator = QuantizationEvaluator(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        tokenizer=tokenizer,
        validation_texts=test_texts,
        max_length=config.data.max_length,
        device_map=config.model.device_map,
        verbose=True
    )

    # Create initial configs
    print("\nGenerating initial test configs...")
    initial_configs = create_initial_configs(num_layers, constraint_matrix)

    print(f"Created {len(initial_configs)} initial configs:")
    for i, cfg in enumerate(initial_configs):
        print(f"  {i+1}. {evaluator._summarize_config(cfg)}")

    # Evaluate first config as a test
    print("\n" + "="*80)
    print("TESTING EVALUATION FRAMEWORK")
    print("="*80)
    print("\nEvaluating first config (most aggressive)...")

    result = evaluator.evaluate(initial_configs[0])

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nResult:")
    print(f"  Perplexity: {result.perplexity:.4f}")
    print(f"  Model size: {result.model_size_gb:.2f} GB")
    print(f"  Eval time: {result.eval_time_seconds:.1f}s")

    # Save history
    evaluator.save_history('/workspace/outputs/analysis/eval_test.json')
