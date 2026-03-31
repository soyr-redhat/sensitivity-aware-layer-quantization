"""Dataset loading utilities."""

from datasets import load_dataset
from typing import List
import random

from ..config import DataConfig


def load_dataset_samples(config: DataConfig) -> List[str]:
    """Load samples from a dataset for routing analysis.

    Args:
        config: DataConfig with dataset parameters

    Returns:
        List of text samples
    """
    print(f"Loading dataset: {config.dataset_name}")
    print(f"  num_samples: {config.num_samples}")
    print(f"  max_length: {config.max_length}")

    # Set random seed for reproducibility
    random.seed(config.seed)

    # Load dataset based on name
    if config.dataset_name == "openwebtext":
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        text_key = "text"
    elif config.dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        text_key = "text"
    elif config.dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_key = "text"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # Collect samples
    samples = []
    for i, item in enumerate(dataset):
        if len(samples) >= config.num_samples:
            break

        text = item[text_key].strip()

        # Filter out very short or empty texts
        if len(text) < 50:
            continue

        # Truncate to approximate character length (rough estimate)
        # We'll do proper tokenization later
        approx_char_limit = config.max_length * 4  # ~4 chars per token
        if len(text) > approx_char_limit:
            text = text[:approx_char_limit]

        samples.append(text)

    print(f"Loaded {len(samples)} samples")
    return samples
