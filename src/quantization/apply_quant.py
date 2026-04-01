"""Actual quantization implementation using bitsandbytes."""

import torch
import torch.nn as nn
from typing import List, Dict
from .layer_quant import LayerQuantConfig, QuantizationStrategy
import gc


def quantize_linear_int8(linear_layer: nn.Linear) -> nn.Module:
    """Quantize a linear layer to INT8 using bitsandbytes.

    Args:
        linear_layer: The linear layer to quantize

    Returns:
        Quantized linear layer
    """
    try:
        import bitsandbytes as bnb

        # Create quantized linear layer
        quant_layer = bnb.nn.Linear8bitLt(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=linear_layer.bias is not None,
            has_fp16_weights=False,
            threshold=6.0
        )

        # Copy weights
        quant_layer.weight = bnb.nn.Int8Params(
            linear_layer.weight.data,
            requires_grad=False,
            has_fp16_weights=False
        )

        if linear_layer.bias is not None:
            quant_layer.bias = linear_layer.bias

        return quant_layer

    except ImportError:
        print("Warning: bitsandbytes not available, returning original layer")
        return linear_layer


def quantize_layer_weights(
    layer,
    attention_quant: QuantizationStrategy,
    mlp_quant: QuantizationStrategy,
    layer_idx: int,
    verbose: bool = False
) -> None:
    """Quantize weights in a transformer layer.

    Args:
        layer: The transformer layer
        attention_quant: Quantization strategy for attention
        mlp_quant: Quantization strategy for MLP
        layer_idx: Layer index for logging
        verbose: Print quantization actions
    """
    # Quantize attention weights
    if attention_quant == QuantizationStrategy.INT8:
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn

            # Quantize Q, K, V projections
            if hasattr(attn, 'q_proj'):
                attn.q_proj = quantize_linear_int8(attn.q_proj)
            if hasattr(attn, 'k_proj'):
                attn.k_proj = quantize_linear_int8(attn.k_proj)
            if hasattr(attn, 'v_proj'):
                attn.v_proj = quantize_linear_int8(attn.v_proj)
            if hasattr(attn, 'o_proj'):
                attn.o_proj = quantize_linear_int8(attn.o_proj)

            if verbose and layer_idx % 4 == 0:
                print(f"    Layer {layer_idx}: Quantized attention to INT8")

    # Quantize MLP weights
    if mlp_quant == QuantizationStrategy.INT8:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp

            # Quantize MLP layers (gate, up, down projections for Mistral)
            if hasattr(mlp, 'gate_proj'):
                mlp.gate_proj = quantize_linear_int8(mlp.gate_proj)
            if hasattr(mlp, 'up_proj'):
                mlp.up_proj = quantize_linear_int8(mlp.up_proj)
            if hasattr(mlp, 'down_proj'):
                mlp.down_proj = quantize_linear_int8(mlp.down_proj)

            if verbose and layer_idx % 4 == 0:
                print(f"    Layer {layer_idx}: Quantized MLP to INT8")

    # INT4 quantization note: bitsandbytes INT4 is more complex
    # For now, we'll treat INT4 same as INT8 or skip it
    # In production, you'd use proper 4-bit quantization

    # BF16/FP16/NONE - keep as is (already loaded in those dtypes)


def apply_layer_wise_quantization_real(
    model,
    configs: List[LayerQuantConfig],
    verbose: bool = True
) -> None:
    """Apply actual layer-wise quantization to model (in-place).

    Args:
        model: The model to quantize
        configs: List of layer quantization configs
        verbose: Print progress
    """
    if verbose:
        print("\n" + "="*80)
        print("APPLYING REAL LAYER-WISE QUANTIZATION")
        print("="*80)
        print("\nNote: This modifies the model in-place")
        print("INT8: Using bitsandbytes Int8 quantization")
        print("INT4: Treated as INT8 for now (4-bit more complex)")
        print("BF16/FP16: Keeping original precision")
        print("")

    # Check if bitsandbytes is available
    try:
        import bitsandbytes as bnb
        print("✓ bitsandbytes available\n")
    except ImportError:
        print("✗ bitsandbytes not available - quantization will be skipped")
        print("Install with: pip install bitsandbytes\n")
        return

    # Track stats
    layers_quantized = 0

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        total_layers = len(model.model.layers)

        for config in configs:
            if config.layer_idx >= total_layers:
                continue

            layer = model.model.layers[config.layer_idx]

            # Map INT4 to INT8 for now (simplified)
            attn_quant = config.attention_quant
            mlp_quant = config.mlp_quant

            if attn_quant == QuantizationStrategy.INT4:
                attn_quant = QuantizationStrategy.INT8
            if mlp_quant == QuantizationStrategy.INT4:
                mlp_quant = QuantizationStrategy.INT8

            # Quantize if needed
            if attn_quant == QuantizationStrategy.INT8 or mlp_quant == QuantizationStrategy.INT8:
                quantize_layer_weights(
                    layer=layer,
                    attention_quant=attn_quant,
                    mlp_quant=mlp_quant,
                    layer_idx=config.layer_idx,
                    verbose=verbose
                )
                layers_quantized += 1

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"\n✓ Quantized {layers_quantized}/{len(configs)} layers")
        print("="*80)


def load_quantized_model(
    model_name: str,
    cache_dir: str,
    quantization_config: Dict,
    device_map: str = "auto"
):
    """Load a uniformly quantized model using HuggingFace config.

    Args:
        model_name: Model name/path
        cache_dir: Cache directory
        quantization_config: Quantization configuration dict
        device_map: Device mapping

    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # Create BitsAndBytesConfig
    if quantization_config.get('load_in_8bit'):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quantization_config.get('load_in_4bit'):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        trust_remote_code=True
    )

    model.eval()
    return model
