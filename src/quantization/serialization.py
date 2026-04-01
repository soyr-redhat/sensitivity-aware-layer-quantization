"""Custom serialization for layer-wise quantized models.

This module provides utilities to save and load layer-wise quantized models
in a format that preserves the quantization and reduces disk/memory usage.
"""

import torch
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from safetensors.torch import save_file, load_file
from .layer_quant import LayerQuantConfig, QuantizationStrategy


def extract_quantized_state_dict(model) -> Dict[str, torch.Tensor]:
    """Extract quantized weights from a model with bitsandbytes layers.

    Args:
        model: Model with quantized layers

    Returns:
        State dict with quantized weights
    """
    state_dict = {}

    for name, param in model.named_parameters():
        # Check if this is a bitsandbytes quantized parameter
        if hasattr(param, 'CB') and param.CB is not None:
            # Store the quantized data
            state_dict[name] = param.data.to(torch.int8)
            # Also store the quantization metadata if it exists
            if hasattr(param, 'SCB') and param.SCB is not None:
                state_dict[f'{name}.SCB'] = param.SCB
        else:
            # Regular parameter (BF16 or other)
            state_dict[name] = param.data

    # Also save buffers (non-parameter tensors)
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer

    return state_dict


def save_quantized_model(
    model,
    save_dir: str,
    configs: List[LayerQuantConfig],
    metadata: Optional[Dict] = None
):
    """Save a quantized model with proper serialization.

    Args:
        model: The quantized model
        save_dir: Directory to save to
        configs: Layer quantization configs
        metadata: Optional metadata to save
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving quantized model to: {save_dir}")

    # Extract state dict
    print("  Extracting quantized weights...")
    state_dict = extract_quantized_state_dict(model)

    # Save using safetensors (efficient binary format)
    weights_path = os.path.join(save_dir, 'model.safetensors')
    print(f"  Saving weights to {weights_path}...")
    save_file(state_dict, weights_path)

    # Calculate size
    total_size = sum(
        param.numel() * param.element_size()
        for param in state_dict.values()
    )
    print(f"  Saved {total_size / (1024**3):.2f} GB")

    # Save quantization config
    config_dict = {
        'strategy': 'layer_wise_custom',
        'num_layers': len(configs),
        'layers': [
            {
                'layer_idx': c.layer_idx,
                'attention_quant': c.attention_quant.value,
                'mlp_quant': c.mlp_quant.value
            }
            for c in configs
        ],
        'metadata': metadata or {}
    }

    config_path = os.path.join(save_dir, 'quantization_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save model config (for reconstruction)
    if hasattr(model, 'config'):
        model.config.save_pretrained(save_dir)

    print("✓ Model saved successfully")


def create_quantized_model_from_weights(
    model_name: str,
    weights_path: str,
    quant_config_path: str,
    device_map: str = "auto"
):
    """Create a quantized model by loading custom-serialized weights.

    Args:
        model_name: Base model name/path
        weights_path: Path to saved quantized weights
        quant_config_path: Path to quantization config
        device_map: Device mapping

    Returns:
        Model with quantized weights loaded
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"\nLoading quantized model from custom format...")

    # Load quantization config
    with open(quant_config_path, 'r') as f:
        quant_config = json.load(f)

    print(f"  Strategy: {quant_config['strategy']}")
    print(f"  Layers: {quant_config['num_layers']}")

    # Load the base model structure (empty)
    print("  Loading model structure...")

    # First try to load config from saved path
    config_dir = os.path.dirname(weights_path)
    if os.path.exists(os.path.join(config_dir, 'config.json')):
        config = AutoConfig.from_pretrained(config_dir)
    else:
        config = AutoConfig.from_pretrained(model_name)

    # Create model with empty weights
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16
    )

    # Load quantized weights
    print("  Loading quantized weights...")
    state_dict = load_file(weights_path)

    # Load weights into model (with proper conversion)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        print(f"  Warning: {len(incompatible.missing_keys)} missing keys")
    if incompatible.unexpected_keys:
        print(f"  Warning: {len(incompatible.unexpected_keys)} unexpected keys")

    # Move to device
    if device_map == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = model.to(device_map)

    model.eval()

    print("✓ Model loaded successfully")
    return model


def save_layerwise_quantized_checkpoint(
    base_model_name: str,
    cache_dir: str,
    configs: List[LayerQuantConfig],
    output_dir: str,
    strategy_name: str,
    device_map: str = "auto"
) -> str:
    """Create a layer-wise quantized checkpoint with custom serialization.

    This loads the model, quantizes it, extracts the quantized weights,
    and saves them in a format that can be loaded without the original
    BF16 weights.

    Args:
        base_model_name: HuggingFace model name
        cache_dir: Cache directory
        configs: Layer quantization configs
        output_dir: Output directory
        strategy_name: Name for this strategy
        device_map: Device mapping

    Returns:
        Path to saved checkpoint
    """
    import gc
    from transformers import AutoModelForCausalLM
    from .apply_quant import apply_layer_wise_quantization_real

    print(f"\n{'='*80}")
    print(f"Creating custom quantized checkpoint: {strategy_name}")
    print(f"{'='*80}")

    save_path = os.path.join(output_dir, strategy_name)

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

    # Apply quantization
    print("\n[2/4] Applying layer-wise quantization...")
    apply_layer_wise_quantization_real(
        model=model,
        configs=configs,
        verbose=True
    )

    # Save with custom serialization
    print("\n[3/4] Saving with custom serialization...")
    save_quantized_model(
        model=model,
        save_dir=save_path,
        configs=configs,
        metadata={
            'base_model': base_model_name,
            'strategy': strategy_name
        }
    )

    # Clean up
    print("\n[4/4] Cleaning up...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ Memory freed")

    print(f"\n{'='*80}")
    print(f"Checkpoint ready at: {save_path}")
    print(f"{'='*80}\n")

    return save_path


def estimate_serialized_size(checkpoint_path: str) -> Dict[str, float]:
    """Estimate the size of a serialized checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Size information
    """
    weights_path = os.path.join(checkpoint_path, 'model.safetensors')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    size_bytes = os.path.getsize(weights_path)

    return {
        'weights_size_bytes': size_bytes,
        'weights_size_mb': size_bytes / (1024**2),
        'weights_size_gb': size_bytes / (1024**3),
        'checkpoint_path': checkpoint_path
    }


def load_quantized_checkpoint(
    checkpoint_path: str,
    device_map: str = "auto"
):
    """Load a quantized checkpoint from custom format.

    Args:
        checkpoint_path: Path to checkpoint directory
        device_map: Device mapping

    Returns:
        Loaded model
    """
    weights_path = os.path.join(checkpoint_path, 'model.safetensors')
    config_path = os.path.join(checkpoint_path, 'quantization_config.json')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load metadata to get base model name
    with open(config_path, 'r') as f:
        quant_config = json.load(f)

    base_model = quant_config.get('metadata', {}).get('base_model')
    if not base_model:
        raise ValueError("Base model not specified in checkpoint metadata")

    return create_quantized_model_from_weights(
        model_name=base_model,
        weights_path=weights_path,
        quant_config_path=config_path,
        device_map=device_map
    )
