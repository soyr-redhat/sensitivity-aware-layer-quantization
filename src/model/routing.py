"""Router hook and routing decision extraction for MoE models."""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class RouterHook:
    """Hook to capture routing decisions from MoE layers.

    This hook intercepts the router outputs to log which experts
    are selected for each token at each layer.
    """

    def __init__(self):
        self.routing_decisions = defaultdict(list)
        self.routing_weights = defaultdict(list)
        self.layer_count = 0
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on all MoE router modules.

        Args:
            model: The MoE model to hook into
        """
        self.hooks = []
        layer_idx = 0

        # For Mixtral, try multiple potential structures
        # 1. Try direct access to model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    hook = layer.block_sparse_moe.gate.register_forward_hook(
                        self._create_hook_fn(layer_idx)
                    )
                    self.hooks.append(hook)
                    layer_idx += 1

        # 2. Fallback: search through all named modules
        if layer_idx == 0:
            for name, module in model.named_modules():
                if 'block_sparse_moe' in name and hasattr(module, 'gate'):
                    hook = module.gate.register_forward_hook(
                        self._create_hook_fn(layer_idx)
                    )
                    self.hooks.append(hook)
                    layer_idx += 1

        self.layer_count = layer_idx
        print(f"Registered hooks on {self.layer_count} MoE layers")

        if self.layer_count == 0:
            print("WARNING: No MoE layers found! Model structure:")
            print(f"  Has 'model' attr: {hasattr(model, 'model')}")
            if hasattr(model, 'model'):
                print(f"  Has 'model.layers' attr: {hasattr(model.model, 'layers')}")
                if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                    print(f"  First layer type: {type(model.model.layers[0])}")
                    print(f"  First layer has block_sparse_moe: {hasattr(model.model.layers[0], 'block_sparse_moe')}")

    def _create_hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # output is the router logits: [batch_size * seq_len, num_experts]
            # We want to capture which experts are selected
            with torch.no_grad():
                router_logits = output
                # Get top-k experts (Mixtral uses top-2)
                top_k = 2  # Mixtral configuration
                routing_weights, selected_experts = torch.topk(
                    router_logits, top_k, dim=-1
                )

                # Store the routing decisions
                self.routing_decisions[layer_idx].append(
                    selected_experts.cpu().numpy()
                )
                self.routing_weights[layer_idx].append(
                    torch.softmax(routing_weights, dim=-1).cpu().numpy()
                )

        return hook_fn

    def clear(self):
        """Clear stored routing decisions."""
        self.routing_decisions.clear()
        self.routing_weights.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_routing_summary(self) -> Dict:
        """Get a summary of routing decisions.

        Returns:
            Dictionary with routing statistics
        """
        summary = {}

        for layer_idx in range(self.layer_count):
            if layer_idx not in self.routing_decisions:
                continue

            # Concatenate all routing decisions for this layer
            decisions = np.concatenate(self.routing_decisions[layer_idx], axis=0)
            weights = np.concatenate(self.routing_weights[layer_idx], axis=0)

            # Calculate statistics
            expert_counts = np.bincount(decisions.flatten(), minlength=8)
            expert_usage = expert_counts / expert_counts.sum()

            summary[f'layer_{layer_idx}'] = {
                'expert_counts': expert_counts.tolist(),
                'expert_usage': expert_usage.tolist(),
                'total_routing_decisions': len(decisions),
                'avg_routing_weights': weights.mean(axis=0).tolist()
            }

        return summary


def collect_routing_decisions(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512
) -> Dict:
    """Collect routing decisions for a batch of texts.

    Args:
        model: The MoE model
        tokenizer: Tokenizer for the model
        texts: List of input texts
        max_length: Maximum sequence length

    Returns:
        Dictionary containing routing decisions and metadata
    """
    hook = RouterHook()
    hook.register_hooks(model)

    all_routing_data = []

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

            # Collect routing decisions for this sample
            routing_data = {
                'text': text,
                'num_tokens': inputs['attention_mask'].sum().item(),
                'routing_decisions': {
                    f'layer_{i}': hook.routing_decisions[i][-1].copy()
                    for i in range(hook.layer_count)
                },
                'routing_weights': {
                    f'layer_{i}': hook.routing_weights[i][-1].copy()
                    for i in range(hook.layer_count)
                }
            }
            all_routing_data.append(routing_data)

        # Get summary statistics
        summary = hook.get_routing_summary()

        return {
            'routing_data': all_routing_data,
            'summary': summary,
            'num_samples': len(texts),
            'num_layers': hook.layer_count
        }

    finally:
        hook.remove_hooks()
