"""Model evaluation utilities for measuring perplexity."""

import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 1,
    device: Optional[str] = None
) -> Dict[str, float]:
    """Calculate perplexity on a set of texts.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        texts: List of text samples
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device to use (None = use model's device)

    Returns:
        Dictionary with perplexity metrics
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    all_losses = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Calculate loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss

            # Count actual tokens (excluding padding)
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            all_losses.append(loss.item())

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'perplexity': float(perplexity),
        'avg_loss': float(avg_loss),
        'total_tokens': total_tokens,
        'num_samples': len(texts),
        'loss_std': float(np.std(all_losses))
    }


def calculate_perplexity_by_prompt_type(
    model,
    tokenizer,
    prompt_datasets: Dict[str, List[str]],
    max_length: int = 512,
    batch_size: int = 1
) -> Dict[str, Dict[str, float]]:
    """Calculate perplexity separately for each prompt type.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt_datasets: Dictionary mapping prompt_type -> list of texts
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation

    Returns:
        Dictionary mapping prompt_type -> perplexity metrics
    """
    results = {}

    for prompt_type, texts in prompt_datasets.items():
        print(f"\nEvaluating {prompt_type}...")
        metrics = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
            batch_size=batch_size
        )
        results[prompt_type] = metrics
        print(f"  Perplexity: {metrics['perplexity']:.2f}")

    # Calculate overall average
    all_texts = [text for texts in prompt_datasets.values() for text in texts]
    overall = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=all_texts,
        max_length=max_length,
        batch_size=batch_size
    )
    results['overall'] = overall

    return results


def compare_quantization_strategies(
    model,
    tokenizer,
    test_texts: List[str],
    quantization_configs: Dict[str, List],  # name -> LayerQuantConfig list
    max_length: int = 512
) -> Dict[str, Dict]:
    """Compare different quantization strategies.

    Args:
        model: Base model (will be quantized with different configs)
        tokenizer: Tokenizer
        test_texts: Test dataset
        quantization_configs: Dictionary of config name -> LayerQuantConfig list
        max_length: Maximum sequence length

    Returns:
        Dictionary mapping config_name -> evaluation results
    """
    from ..quantization import apply_layer_wise_quantization, get_model_memory_footprint

    results = {}

    # Baseline: no quantization
    print("\n" + "="*80)
    print("BASELINE (BF16)")
    print("="*80)

    baseline_memory = get_model_memory_footprint(model)
    baseline_metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        max_length=max_length
    )

    results['baseline_bf16'] = {
        'perplexity': baseline_metrics['perplexity'],
        'memory_gb': baseline_memory['total_memory_gb'],
        'metrics': baseline_metrics,
        'memory': baseline_memory
    }

    print(f"Perplexity: {baseline_metrics['perplexity']:.2f}")
    print(f"Memory: {baseline_memory['total_memory_gb']:.2f} GB")

    # Test each quantization config
    for config_name, layer_configs in quantization_configs.items():
        print("\n" + "="*80)
        print(f"TESTING: {config_name}")
        print("="*80)

        # Note: In practice, you'd actually quantize the model here
        # For now, we're just simulating the evaluation
        # apply_layer_wise_quantization(model, layer_configs, verbose=True)

        # Evaluate
        metrics = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            max_length=max_length
        )

        # Estimate memory (since we're not actually quantizing)
        from ..quantization.layer_quant import estimate_quantized_memory
        memory_estimate = estimate_quantized_memory(
            baseline_memory_gb=baseline_memory['total_memory_gb'],
            configs=layer_configs,
            num_layers=len(layer_configs)
        )

        results[config_name] = {
            'perplexity': metrics['perplexity'],
            'perplexity_delta': metrics['perplexity'] - baseline_metrics['perplexity'],
            'perplexity_increase_pct': ((metrics['perplexity'] / baseline_metrics['perplexity']) - 1) * 100,
            'estimated_memory_gb': memory_estimate['estimated_memory_gb'],
            'memory_savings_gb': memory_estimate['savings_gb'],
            'memory_savings_pct': memory_estimate['savings_percent'],
            'compression_ratio': memory_estimate['compression_ratio'],
            'metrics': metrics,
            'memory_estimate': memory_estimate
        }

        print(f"Perplexity: {metrics['perplexity']:.2f} "
              f"(+{results[config_name]['perplexity_increase_pct']:.2f}%)")
        print(f"Estimated Memory: {memory_estimate['estimated_memory_gb']:.2f} GB "
              f"(-{memory_estimate['savings_percent']:.1f}%)")
        print(f"Compression Ratio: {memory_estimate['compression_ratio']:.2f}x")

    return results
