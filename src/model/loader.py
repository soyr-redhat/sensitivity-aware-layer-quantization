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

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        cache_dir=config.cache_dir,
        device_map=config.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code
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
