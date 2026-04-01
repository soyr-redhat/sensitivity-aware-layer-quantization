"""Save and load quantized model checkpoints."""

import torch
import os
import json
from pathlib import Path
from typing import List, Dict
from .layer_quant import LayerQuantConfig, QuantizationStrategy


def save_quantized_model(
    model,
    configs: List[LayerQuantConfig],
    output_dir: str,
    model_name: str = "layer_quantized"
) -> str:
    """Save a quantized model to disk.

    Args:
        model: The quantized model
        configs: Layer quantization configs used
        output_dir: Directory to save the model
        model_name: Name for the saved model

    Returns:
        Path to saved model directory
    """
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving quantized model to: {save_path}")

    # Save the model using HuggingFace's save_pretrained
    # This will save the quantized weights
    model.save_pretrained(save_path, safe_serialization=True)

    # Save quantization config for reference
    config_dict = {
        'strategy': 'layer_wise',
        'num_layers': len(configs),
        'layers': [
            {
                'layer_idx': c.layer_idx,
                'attention_quant': c.attention_quant.value,
                'mlp_quant': c.mlp_quant.value
            }
            for c in configs
        ]
    }

    config_path = os.path.join(save_path, 'quantization_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"✓ Model saved successfully")
    return save_path


def load_quantized_model_from_checkpoint(
    checkpoint_path: str,
    device_map: str = "auto"
):
    """Load a quantized model from checkpoint.

    Args:
        checkpoint_path: Path to the saved model
        device_map: Device mapping strategy

    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM

    print(f"\nLoading quantized model from: {checkpoint_path}")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )

    model.eval()

    # Load quantization config for reference
    config_path = os.path.join(checkpoint_path, 'quantization_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            quant_config = json.load(f)
        print(f"✓ Loaded {quant_config.get('strategy', 'unknown')} quantization strategy")

    return model


def quantize_and_save(
    model,
    tokenizer,
    configs: List[LayerQuantConfig],
    output_dir: str,
    strategy_name: str,
    verbose: bool = True
):
    """Quantize a model and save it to disk (then free memory).

    Args:
        model: Model to quantize
        tokenizer: Tokenizer (will also be saved)
        configs: Layer quantization configs
        output_dir: Directory to save to
        strategy_name: Name for this quantization strategy
        verbose: Print progress

    Returns:
        Path to saved model
    """
    from .apply_quant import apply_layer_wise_quantization_real

    # Apply quantization in-place
    apply_layer_wise_quantization_real(
        model=model,
        configs=configs,
        verbose=verbose
    )

    # Save the quantized model
    save_path = save_quantized_model(
        model=model,
        configs=configs,
        output_dir=output_dir,
        model_name=strategy_name
    )

    # Save tokenizer too
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        if verbose:
            print("✓ Tokenizer saved")

    return save_path


def estimate_checkpoint_size(checkpoint_path: str) -> Dict[str, float]:
    """Estimate the size of a saved checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint

    Returns:
        Dictionary with size info
    """
    total_size = 0
    file_count = 0

    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
            file_count += 1

    return {
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024**2),
        'total_size_gb': total_size / (1024**3),
        'file_count': file_count,
        'checkpoint_path': checkpoint_path
    }


def create_quantized_checkpoint_strategy(
    base_model_name: str,
    cache_dir: str,
    configs: List[LayerQuantConfig],
    output_dir: str,
    strategy_name: str,
    device_map: str = "auto"
) -> str:
    """Create a quantized checkpoint by loading, quantizing, and saving.

    This is a clean workflow that:
    1. Loads base model
    2. Applies quantization
    3. Saves to disk
    4. Frees memory
    5. Returns path for later reloading

    Args:
        base_model_name: HuggingFace model name
        cache_dir: Cache directory for base model
        configs: Quantization configs
        output_dir: Where to save quantized checkpoint
        strategy_name: Name for this strategy
        device_map: Device mapping

    Returns:
        Path to saved checkpoint
    """
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*80}")
    print(f"Creating quantized checkpoint: {strategy_name}")
    print(f"{'='*80}")

    # Load base model
    print("\n[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply quantization and save
    print("\n[2/4] Applying quantization...")
    save_path = quantize_and_save(
        model=model,
        tokenizer=tokenizer,
        configs=configs,
        output_dir=output_dir,
        strategy_name=strategy_name,
        verbose=True
    )

    # Check checkpoint size
    print("\n[3/4] Checking checkpoint size...")
    size_info = estimate_checkpoint_size(save_path)
    print(f"  Checkpoint size: {size_info['total_size_gb']:.2f} GB")
    print(f"  Files: {size_info['file_count']}")

    # Clean up
    print("\n[4/4] Cleaning up memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ Memory freed")

    print(f"\n{'='*80}")
    print(f"Checkpoint ready at: {save_path}")
    print(f"{'='*80}\n")

    return save_path
