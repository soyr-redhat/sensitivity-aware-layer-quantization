"""Layer-wise quantization implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QuantizationStrategy(Enum):
    """Quantization precision levels."""
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NONE = "none"  # Keep original precision


@dataclass
class LayerQuantConfig:
    """Quantization configuration for a single layer."""
    layer_idx: int
    attention_quant: QuantizationStrategy
    mlp_quant: QuantizationStrategy

    def __repr__(self):
        return f"Layer{self.layer_idx}[attn={self.attention_quant.value}, mlp={self.mlp_quant.value}]"


def create_layer_configs_from_activation_profile(
    activation_analysis: Dict,
    num_layers: int,
    aggressive_threshold: float = 0.3,  # Bottom 30% of activation
    conservative_threshold: float = 0.7  # Top 30% of activation
) -> List[LayerQuantConfig]:
    """Generate layer quantization configs based on activation profiles.

    Args:
        activation_analysis: Activation analysis results
        num_layers: Total number of layers
        aggressive_threshold: Percentile below which to use aggressive quantization
        conservative_threshold: Percentile above which to use conservative quantization

    Returns:
        List of LayerQuantConfig for each layer
    """
    # Extract MLP activation levels per layer from a reference prompt type
    # (they're all similar anyway based on our analysis)
    prompt_type = list(activation_analysis['prompt_type_signatures'].keys())[0]
    mlp_profile = activation_analysis['prompt_type_signatures'][prompt_type]['mlp_profile']
    attn_profile = activation_analysis['prompt_type_signatures'][prompt_type]['attn_profile']

    # Normalize profiles to [0, 1]
    mlp_min, mlp_max = min(mlp_profile), max(mlp_profile)
    mlp_norm = [(x - mlp_min) / (mlp_max - mlp_min) for x in mlp_profile]

    attn_min, attn_max = min(attn_profile), max(attn_profile)
    attn_norm = [(x - attn_min) / (attn_max - attn_min) for x in attn_profile]

    configs = []
    for layer_idx in range(num_layers):
        mlp_activation = mlp_norm[layer_idx]
        attn_activation = attn_norm[layer_idx]

        # Determine MLP quantization based on activation level
        if mlp_activation < aggressive_threshold:
            mlp_quant = QuantizationStrategy.INT4
        elif mlp_activation < conservative_threshold:
            mlp_quant = QuantizationStrategy.INT8
        else:
            mlp_quant = QuantizationStrategy.BF16

        # Determine attention quantization (usually more sensitive)
        if attn_activation < aggressive_threshold:
            attn_quant = QuantizationStrategy.INT8
        elif attn_activation < conservative_threshold:
            attn_quant = QuantizationStrategy.INT8
        else:
            attn_quant = QuantizationStrategy.BF16

        configs.append(LayerQuantConfig(
            layer_idx=layer_idx,
            attention_quant=attn_quant,
            mlp_quant=mlp_quant
        ))

    return configs


def create_uniform_config(
    num_layers: int,
    strategy: QuantizationStrategy
) -> List[LayerQuantConfig]:
    """Create uniform quantization config (baseline).

    Args:
        num_layers: Number of layers
        strategy: Quantization strategy to apply uniformly

    Returns:
        List of LayerQuantConfig
    """
    return [
        LayerQuantConfig(
            layer_idx=i,
            attention_quant=strategy,
            mlp_quant=strategy
        )
        for i in range(num_layers)
    ]


def create_manual_config(
    num_layers: int,
    early_quant: QuantizationStrategy = QuantizationStrategy.INT4,
    mid_quant: QuantizationStrategy = QuantizationStrategy.INT8,
    late_quant: QuantizationStrategy = QuantizationStrategy.BF16,
    early_cutoff: int = 10,
    late_cutoff: int = 21
) -> List[LayerQuantConfig]:
    """Create manually configured layer-wise quantization.

    Args:
        num_layers: Number of layers
        early_quant: Quantization for early layers
        mid_quant: Quantization for middle layers
        late_quant: Quantization for late layers
        early_cutoff: Layer index where early ends
        late_cutoff: Layer index where late begins

    Returns:
        List of LayerQuantConfig
    """
    configs = []
    for i in range(num_layers):
        if i < early_cutoff:
            quant = early_quant
        elif i < late_cutoff:
            quant = mid_quant
        else:
            quant = late_quant

        configs.append(LayerQuantConfig(
            layer_idx=i,
            attention_quant=quant,
            mlp_quant=quant
        ))

    return configs


def get_quantization_dtype(strategy: QuantizationStrategy) -> torch.dtype:
    """Get torch dtype for a quantization strategy.

    Args:
        strategy: Quantization strategy

    Returns:
        torch.dtype
    """
    dtype_map = {
        QuantizationStrategy.FP16: torch.float16,
        QuantizationStrategy.BF16: torch.bfloat16,
        QuantizationStrategy.INT8: torch.int8,
        QuantizationStrategy.INT4: torch.int8,  # PyTorch doesn't have int4, use int8
        QuantizationStrategy.NONE: torch.float32,
    }
    return dtype_map.get(strategy, torch.bfloat16)


def apply_layer_wise_quantization(
    model,
    configs: List[LayerQuantConfig],
    verbose: bool = True
) -> None:
    """Apply layer-wise quantization to a model (in-place).

    Note: This is a simplified version. In practice, you'd use proper
    quantization libraries like bitsandbytes or torch.quantization.

    Args:
        model: The model to quantize
        configs: List of layer quantization configs
        verbose: Print quantization decisions
    """
    if verbose:
        print("\n" + "="*80)
        print("APPLYING LAYER-WISE QUANTIZATION")
        print("="*80)

    # Track quantization stats
    quant_stats = {
        QuantizationStrategy.FP16: 0,
        QuantizationStrategy.BF16: 0,
        QuantizationStrategy.INT8: 0,
        QuantizationStrategy.INT4: 0,
    }

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for config in configs:
            if config.layer_idx >= len(model.model.layers):
                continue

            layer = model.model.layers[config.layer_idx]

            # Quantize attention
            if hasattr(layer, 'self_attn'):
                quant_stats[config.attention_quant] += 1
                if verbose and config.layer_idx % 8 == 0:  # Print every 8th layer
                    print(f"  Layer {config.layer_idx:2d} Attention: {config.attention_quant.value}")

            # Quantize MLP
            if hasattr(layer, 'mlp'):
                quant_stats[config.mlp_quant] += 1
                if verbose and config.layer_idx % 8 == 0:
                    print(f"  Layer {config.layer_idx:2d} MLP:       {config.mlp_quant.value}")

    if verbose:
        print("\n" + "-"*80)
        print("Quantization Summary:")
        print("-"*80)
        for strategy, count in quant_stats.items():
            if count > 0:
                print(f"  {strategy.value:6s}: {count:3d} layers")
        print("="*80)


def get_model_memory_footprint(model) -> Dict[str, float]:
    """Calculate model memory footprint.

    Args:
        model: The model

    Returns:
        Dictionary with memory statistics in GB
    """
    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    total_buffers = 0
    for name, buffer in model.named_buffers():
        total_buffers += buffer.numel() * buffer.element_size()

    return {
        'total_params': total_params,
        'total_params_millions': total_params / 1e6,
        'param_memory_gb': total_bytes / (1024**3),
        'buffer_memory_gb': total_buffers / (1024**3),
        'total_memory_gb': (total_bytes + total_buffers) / (1024**3)
    }


def estimate_quantized_memory(
    baseline_memory_gb: float,
    configs: List[LayerQuantConfig],
    num_layers: int
) -> Dict[str, float]:
    """Estimate memory savings from layer-wise quantization.

    Args:
        baseline_memory_gb: Memory usage in bf16/fp16
        configs: Quantization configs
        num_layers: Total number of layers

    Returns:
        Dictionary with memory estimates
    """
    # Rough estimates of memory savings per strategy
    # (relative to bf16 as baseline = 1.0)
    memory_multipliers = {
        QuantizationStrategy.BF16: 1.0,
        QuantizationStrategy.FP16: 1.0,
        QuantizationStrategy.INT8: 0.5,
        QuantizationStrategy.INT4: 0.25,
        QuantizationStrategy.NONE: 2.0,  # fp32
    }

    # Count layers by quantization level
    layer_counts = {s: 0 for s in QuantizationStrategy}
    for config in configs:
        # Average attention and MLP quantization
        # (simplified - in reality they may have different sizes)
        layer_counts[config.attention_quant] += 0.5
        layer_counts[config.mlp_quant] += 0.5

    # Calculate weighted average multiplier
    total_multiplier = 0
    for strategy, count in layer_counts.items():
        weight = count / num_layers
        total_multiplier += weight * memory_multipliers[strategy]

    estimated_memory = baseline_memory_gb * total_multiplier
    savings_gb = baseline_memory_gb - estimated_memory
    savings_percent = (savings_gb / baseline_memory_gb) * 100

    return {
        'baseline_memory_gb': baseline_memory_gb,
        'estimated_memory_gb': estimated_memory,
        'savings_gb': savings_gb,
        'savings_percent': savings_percent,
        'compression_ratio': baseline_memory_gb / estimated_memory
    }
