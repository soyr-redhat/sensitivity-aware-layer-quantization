"""Model and tokenizer loading utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

from ..config import ModelConfig


def load_model(config: ModelConfig):
    """Load Mixtral model with proper configuration.

    Args:
        config: ModelConfig with model loading parameters

    Returns:
        Loaded model
    """
    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    print(f"Loading model: {config.name}")
    print(f"  dtype: {config.torch_dtype}")
    print(f"  device_map: {config.device_map}")

    # Build kwargs for model loading
    model_kwargs = {
        "cache_dir": config.cache_dir,
        "device_map": config.device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }

    # Add quantization config if specified
    if hasattr(config, 'load_in_4bit') and config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        print("  Using 4-bit quantization (NF4)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(
                getattr(config, 'bnb_4bit_compute_dtype', 'float16'),
                torch.float16
            ),
            bnb_4bit_quant_type=getattr(config, 'bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=getattr(config, 'bnb_4bit_use_double_quant', True),
        )

        model_kwargs['quantization_config'] = bnb_config
        # Remove torch_dtype when using quantization
        model_kwargs.pop('torch_dtype')

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        **model_kwargs
    )

    model.eval()
    return model


def load_tokenizer(config: ModelConfig):
    """Load tokenizer for the model.

    Args:
        config: ModelConfig with model name

    Returns:
        Loaded tokenizer
    """
    print(f"Loading tokenizer: {config.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(config: ModelConfig) -> Tuple:
    """Convenience function to load both model and tokenizer.

    Args:
        config: ModelConfig with model parameters

    Returns:
        Tuple of (model, tokenizer)
    """
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    return model, tokenizer
