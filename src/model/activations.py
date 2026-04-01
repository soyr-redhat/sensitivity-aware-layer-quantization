"""Activation profiling for prompt-aware quantization."""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class ActivationHook:
    """Hook to capture activation statistics from model layers.

    This hook intercepts layer outputs to measure activation magnitudes,
    which helps identify which parts of the model are "active" for
    different prompt types.
    """

    def __init__(self, track_attention: bool = True, track_mlp: bool = True):
        self.track_attention = track_attention
        self.track_mlp = track_mlp

        # Store activation statistics (not raw activations - too large)
        self.attn_stats = defaultdict(list)  # layer_idx -> [stats_per_sample]
        self.mlp_stats = defaultdict(list)

        self.layer_count = 0
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on attention and MLP layers.

        Args:
            model: The transformer model to hook into
        """
        self.hooks = []
        layer_idx = 0

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                # Hook attention output
                if self.track_attention and hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        self._create_attn_hook_fn(layer_idx)
                    )
                    self.hooks.append(hook)

                # Hook MLP output
                if self.track_mlp and hasattr(layer, 'mlp'):
                    hook = layer.mlp.register_forward_hook(
                        self._create_mlp_hook_fn(layer_idx)
                    )
                    self.hooks.append(hook)

                layer_idx += 1

        self.layer_count = layer_idx
        print(f"Registered hooks on {self.layer_count} layers")

    def _create_attn_hook_fn(self, layer_idx: int):
        """Create a hook function for attention layers."""
        def hook_fn(module, input, output):
            with torch.no_grad():
                # output is typically (hidden_states, ...) or just hidden_states
                hidden_states = output[0] if isinstance(output, tuple) else output

                # Compute statistics over the activation
                stats = self._compute_activation_stats(hidden_states)
                self.attn_stats[layer_idx].append(stats)

        return hook_fn

    def _create_mlp_hook_fn(self, layer_idx: int):
        """Create a hook function for MLP layers."""
        def hook_fn(module, input, output):
            with torch.no_grad():
                # MLP output is typically just the tensor
                hidden_states = output

                stats = self._compute_activation_stats(hidden_states)
                self.mlp_stats[layer_idx].append(stats)

        return hook_fn

    def _compute_activation_stats(self, tensor: torch.Tensor) -> Dict:
        """Compute statistics over activations.

        Args:
            tensor: Activation tensor [batch, seq_len, hidden_dim]

        Returns:
            Dictionary with activation statistics
        """
        # Move to CPU and compute stats
        tensor = tensor.float().cpu()

        # Aggregate over batch and sequence dimensions
        # We want per-feature statistics
        flat = tensor.view(-1, tensor.size(-1))  # [batch*seq_len, hidden_dim]

        abs_vals = torch.abs(flat)

        return {
            'mean': abs_vals.mean(dim=0).numpy(),      # [hidden_dim]
            'max': abs_vals.max(dim=0)[0].numpy(),     # [hidden_dim]
            'std': abs_vals.std(dim=0).numpy(),        # [hidden_dim]
            'l2_norm': torch.norm(flat, dim=0).numpy() # [hidden_dim]
        }

    def clear(self):
        """Clear stored activation statistics."""
        self.attn_stats.clear()
        self.mlp_stats.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activation_summary(self) -> Dict:
        """Get summary of activation statistics across all samples.

        Returns:
            Dictionary with aggregated activation statistics per layer
        """
        summary = {
            'attention': {},
            'mlp': {}
        }

        # Aggregate attention statistics
        for layer_idx in range(self.layer_count):
            if layer_idx in self.attn_stats and self.attn_stats[layer_idx]:
                # Stack all samples for this layer
                means = np.stack([s['mean'] for s in self.attn_stats[layer_idx]])
                maxs = np.stack([s['max'] for s in self.attn_stats[layer_idx]])

                summary['attention'][f'layer_{layer_idx}'] = {
                    'mean_activation': means.mean(axis=0).tolist(),
                    'max_activation': maxs.max(axis=0).tolist(),
                    'avg_mean': float(means.mean()),
                    'avg_max': float(maxs.max()),
                }

            if layer_idx in self.mlp_stats and self.mlp_stats[layer_idx]:
                means = np.stack([s['mean'] for s in self.mlp_stats[layer_idx]])
                maxs = np.stack([s['max'] for s in self.mlp_stats[layer_idx]])

                summary['mlp'][f'layer_{layer_idx}'] = {
                    'mean_activation': means.mean(axis=0).tolist(),
                    'max_activation': maxs.max(axis=0).tolist(),
                    'avg_mean': float(means.mean()),
                    'avg_max': float(maxs.max()),
                }

        return summary


def collect_activations(
    model,
    tokenizer,
    texts: List[str],
    prompt_type: str,
    max_length: int = 512
) -> Dict:
    """Collect activation statistics for a batch of texts.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for the model
        texts: List of input texts
        prompt_type: Type of prompts (e.g., 'code', 'math')
        max_length: Maximum sequence length

    Returns:
        Dictionary containing activation statistics and metadata
    """
    hook = ActivationHook(track_attention=True, track_mlp=True)
    hook.register_hooks(model)

    try:
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )

            # Move to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Run forward pass
            with torch.no_grad():
                _ = model(**inputs)

        # Get summary statistics
        summary = hook.get_activation_summary()

        return {
            'prompt_type': prompt_type,
            'num_samples': len(texts),
            'num_layers': hook.layer_count,
            'activation_stats': summary
        }

    finally:
        hook.remove_hooks()
