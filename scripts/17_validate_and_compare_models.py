#!/usr/bin/env python3
"""Validate and compare GGUF models: check sizes and analyze compression."""

import os
import json
from pathlib import Path

# Mistral-7B parameter count (approximate)
MISTRAL_7B_PARAMS = 7.24e9


def get_model_info(model_path, model_name):
    """Get model file information."""
    if not os.path.exists(model_path):
        return {
            "name": model_name,
            "status": "NOT_FOUND",
            "size_gb": 0,
            "bpw": 0
        }

    size_bytes = os.path.getsize(model_path)
    size_gb = size_bytes / (1024**3)
    size_mb = size_bytes / (1024**2)

    # Estimate bits per weight
    bpw = (size_bytes * 8) / MISTRAL_7B_PARAMS

    return {
        "name": model_name,
        "status": "OK",
        "path": model_path,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_gb": size_gb,
        "bpw": bpw
    }


def main():
    gguf_dir = "/workspace/gguf_models"
    output_dir = "/workspace/outputs/benchmarks"
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("GGUF MODEL VALIDATION AND COMPARISON")
    print("="*80)
    print()

    # Define models
    models = {
        # Uniform baselines
        "Q2_K": os.path.join(gguf_dir, "mistral-7b-q2k.gguf"),
        "Q4_K_M": os.path.join(gguf_dir, "mistral-7b-q4km.gguf"),
        "Q6_K": os.path.join(gguf_dir, "mistral-7b-q6k.gguf"),
        "Q8_0": os.path.join(gguf_dir, "mistral-7b-q8.gguf"),
        "F16": os.path.join(gguf_dir, "mistral-7b-f16.gguf"),

        # Mixed-precision
        "aggressive_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-aggressive_mixed.gguf"),
        "balanced_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-balanced_mixed.gguf"),
        "conservative_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-conservative_mixed.gguf"),
    }

    # Collect info
    results = {}
    print("Uniform Quantization Baselines:")
    print("-" * 40)
    for name in ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0", "F16"]:
        info = get_model_info(models[name], name)
        results[name] = info
        if info["status"] == "OK":
            print(f"  ✓ {name:<20} {info['size_gb']:>6.2f} GB ({info['bpw']:>5.2f} BPW)")
        else:
            print(f"  ✗ {name:<20} NOT FOUND")

    print()
    print("Mixed-Precision Variants:")
    print("-" * 40)
    for name in ["aggressive_mixed", "balanced_mixed", "conservative_mixed"]:
        info = get_model_info(models[name], name)
        results[name] = info
        if info["status"] == "OK":
            print(f"  ✓ {name:<20} {info['size_gb']:>6.2f} GB ({info['bpw']:>5.2f} BPW)")
        else:
            print(f"  ✗ {name:<20} NOT FOUND")

    print()
    print("="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    print()

    # Size comparisons
    print("Mixed vs Uniform at Similar Sizes:")
    print("-" * 40)

    comparisons = []

    # aggressive_mixed vs Q2_K
    if results["aggressive_mixed"]["status"] == "OK" and results["Q2_K"]["status"] == "OK":
        agg = results["aggressive_mixed"]
        q2k = results["Q2_K"]
        diff_pct = ((agg["size_gb"] - q2k["size_gb"]) / q2k["size_gb"]) * 100
        comp = {
            "mixed": "aggressive_mixed",
            "uniform": "Q2_K",
            "mixed_size": agg["size_gb"],
            "uniform_size": q2k["size_gb"],
            "diff_pct": diff_pct
        }
        comparisons.append(comp)
        print(f"  aggressive_mixed ({agg['size_gb']:.2f}GB) vs Q2_K ({q2k['size_gb']:.2f}GB):")
        print(f"    Size difference: {diff_pct:+.1f}%")
        print(f"    BPW: {agg['bpw']:.2f} vs {q2k['bpw']:.2f}")
        print()

    # balanced_mixed vs Q4_K_M
    if results["balanced_mixed"]["status"] == "OK" and results["Q4_K_M"]["status"] == "OK":
        bal = results["balanced_mixed"]
        q4 = results["Q4_K_M"]
        diff_pct = ((bal["size_gb"] - q4["size_gb"]) / q4["size_gb"]) * 100
        comp = {
            "mixed": "balanced_mixed",
            "uniform": "Q4_K_M",
            "mixed_size": bal["size_gb"],
            "uniform_size": q4["size_gb"],
            "diff_pct": diff_pct
        }
        comparisons.append(comp)
        print(f"  balanced_mixed ({bal['size_gb']:.2f}GB) vs Q4_K_M ({q4['size_gb']:.2f}GB):")
        print(f"    Size difference: {diff_pct:+.1f}%")
        print(f"    BPW: {bal['bpw']:.2f} vs {q4['bpw']:.2f}")
        print()

    # conservative_mixed vs Q6_K
    if results["conservative_mixed"]["status"] == "OK" and results["Q6_K"]["status"] == "OK":
        con = results["conservative_mixed"]
        q6 = results["Q6_K"]
        diff_pct = ((con["size_gb"] - q6["size_gb"]) / q6["size_gb"]) * 100
        comp = {
            "mixed": "conservative_mixed",
            "uniform": "Q6_K",
            "mixed_size": con["size_gb"],
            "uniform_size": q6["size_gb"],
            "diff_pct": diff_pct
        }
        comparisons.append(comp)
        print(f"  conservative_mixed ({con['size_gb']:.2f}GB) vs Q6_K ({q6['size_gb']:.2f}GB):")
        print(f"    Size difference: {diff_pct:+.1f}%")
        print(f"    BPW: {con['bpw']:.2f} vs {q6['bpw']:.2f}")
        print()

    # Compression ratios vs F16
    if results["F16"]["status"] == "OK":
        f16_size = results["F16"]["size_gb"]
        print("Compression vs F16 Baseline:")
        print("-" * 40)
        for name in ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0",
                     "aggressive_mixed", "balanced_mixed", "conservative_mixed"]:
            if results[name]["status"] == "OK":
                compression = (1 - results[name]["size_gb"] / f16_size) * 100
                print(f"  {name:<20} {compression:>5.1f}% smaller than F16")
        print()

    # Save results
    output_file = os.path.join(output_dir, "model_comparison.json")
    with open(output_file, 'w') as f:
        json.dump({
            "models": results,
            "comparisons": comparisons
        }, f, indent=2)

    print("="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"\nOutput: {output_file}")
    print()

    # Summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'Model':<22} {'Size (GB)':<12} {'BPW':<8} {'vs F16':<12} {'Type':<12}")
    print("-" * 80)

    for name in ["F16", "Q8_0", "Q6_K", "Q4_K_M", "Q2_K",
                 "conservative_mixed", "balanced_mixed", "aggressive_mixed"]:
        if name in results and results[name]["status"] == "OK":
            r = results[name]
            comp = ""
            if results["F16"]["status"] == "OK":
                compression = (1 - r["size_gb"] / results["F16"]["size_gb"]) * 100
                comp = f"-{compression:.1f}%"

            model_type = "Mixed" if "mixed" in name else "Uniform"
            print(f"{name:<22} {r['size_gb']:>10.2f}  {r['bpw']:>6.2f}  {comp:<12} {model_type:<12}")

    print()


if __name__ == '__main__':
    main()
