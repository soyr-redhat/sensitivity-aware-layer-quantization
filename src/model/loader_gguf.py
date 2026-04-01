"""GGUF model loading utilities using llama.cpp."""

from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import Optional


class GGUFModelWrapper:
    """Wrapper for GGUF models to provide a consistent interface."""

    def __init__(self, llama_model: Llama, repo_id: str, filename: str):
        self.model = llama_model
        self.repo_id = repo_id
        self.filename = filename
        # Track routing decisions
        self.last_routing_decisions = None

    def __call__(self, input_ids, **kwargs):
        """Forward pass compatible with transformers interface."""
        # Convert input_ids to text tokens if needed
        # Note: This is a simplified version - llama.cpp doesn't expose
        # routing decisions directly, so we'll need to patch this

        # For now, return a placeholder
        # We'll need to modify llama.cpp or use a different approach
        raise NotImplementedError(
            "GGUF routing extraction requires custom llama.cpp build. "
            "See docs for building llama.cpp with routing hooks."
        )

    @property
    def device(self):
        """Compatibility property."""
        return "cpu"

    def eval(self):
        """Compatibility method."""
        pass


def load_gguf_model(config):
    """Load a GGUF quantized model using llama.cpp.

    Args:
        config: ModelConfig with GGUF parameters

    Returns:
        GGUFModelWrapper instance
    """
    print(f"Loading GGUF model: {config.repo_id}")
    print(f"  File: {config.filename}")

    # Download model from HuggingFace
    print("  Downloading from HuggingFace Hub...")
    model_path = hf_hub_download(
        repo_id=config.repo_id,
        filename=config.filename,
        cache_dir=getattr(config, 'cache_dir', None)
    )

    print(f"  Model downloaded to: {model_path}")
    print(f"  Loading with llama.cpp...")

    # Load with llama.cpp
    llama_model = Llama(
        model_path=model_path,
        n_ctx=getattr(config, 'n_ctx', 2048),
        n_gpu_layers=getattr(config, 'n_gpu_layers', 0),
        n_threads=getattr(config, 'n_threads', 8),
        verbose=getattr(config, 'verbose', False)
    )

    print("  Model loaded successfully!")

    return GGUFModelWrapper(llama_model, config.repo_id, config.filename)


def load_gguf_tokenizer(config):
    """Load tokenizer for GGUF model.

    Note: llama.cpp handles tokenization internally, but we need
    a transformers tokenizer for compatibility with our pipeline.
    """
    from transformers import AutoTokenizer

    # Use the base model's tokenizer
    base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print(f"Loading tokenizer from: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
