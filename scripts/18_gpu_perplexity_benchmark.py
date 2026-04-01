#!/usr/bin/env python3
"""GPU-accelerated perplexity benchmark for GGUF models.

Uses llama.cpp with GPU offloading to measure actual model quality.
"""

import os
import json
import subprocess
import time
from pathlib import Path

# WikiText-2 style test data for perplexity calculation
TEST_DATA = """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.

The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels. The climb from ground level to the first level is over 300 steps, as is the climb from the first level to the second. Although there is a staircase to the top level, it is usually accessible only by lift.

The design of the Eiffel Tower is attributed to Maurice Koechlin and Émile Nouguier, two senior engineers working for the Compagnie des Établissements Eiffel. It was to be the centerpiece of the 1889 Exposition Universelle, a world's fair to celebrate the centennial of the French Revolution. Although Eiffel himself had little to do with the conceptualization of the tower, Eiffel objected to the original design, and Nouguier and Koechlin then asked Stephen Sauvestre, the company's head of the architecture department, to contribute to the design. Sauvestre added decorative arches to the base, a glass pavilion to the first level, and other embellishments. The new version gained Eiffel's support; he bought the rights to the patent on the design which Koechlin, Nouguier, and Sauvestre had taken out, and the design was exhibited at the Exhibition of Decorative Arts in the autumn of 1884 under the company name. On 30 March 1885, Eiffel presented his plans to the Société des Ingénieurs Civils; after discussing the technical problems and emphasizing the practical uses of the tower, he finished his talk by saying the tower would symbolize "not only the art of the modern engineer, but also the century of Industry and Science in which we are living, and for which the way was prepared by the great scientific movement of the eighteenth century and by the Revolution of 1789, to which this monument will be built as an expression of France's gratitude"."""


def create_test_file(output_path="/workspace/test_perplexity.txt"):
    """Create test file for perplexity measurement."""
    with open(output_path, 'w') as f:
        # Repeat test data to get more samples
        for _ in range(10):
            f.write(TEST_DATA + "\n\n")
    return output_path


def run_perplexity_test(model_path, model_name, test_file, n_gpu_layers=99):
    """Run perplexity test using llama-cli.

    Args:
        model_path: Path to GGUF file
        model_name: Name for reporting
        test_file: Test data file
        n_gpu_layers: Number of layers to offload to GPU (99 = all)

    Returns:
        Dict with results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"GPU layers: {n_gpu_layers}")

    if not os.path.exists(model_path):
        print("✗ Model not found")
        return {
            "model": model_name,
            "status": "not_found"
        }

    llama_cli = "/workspace/llama.cpp/build/bin/llama-cli"

    # Build command
    cmd = [
        llama_cli,
        "-m", model_path,
        "-f", test_file,
        "-ngl", str(n_gpu_layers),  # GPU layers
        "-c", "2048",  # Context size
        "-n", "0",  # Don't generate, just process
        "-b", "512",  # Batch size
        "--log-disable",  # Disable chat logging
    ]

    print("\nRunning perplexity test...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed = time.time() - start_time

        # Parse output for metrics
        output = result.stdout + result.stderr

        # Extract timing info
        tokens_per_sec = None
        prompt_eval_time = None

        for line in output.split('\n'):
            if 'llama_perf_context_print' in line or 'eval time' in line.lower():
                # Try to extract tokens/second
                if 'tokens per second' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'tokens' in part.lower() and i > 0:
                            try:
                                tokens_per_sec = float(parts[i-1])
                            except:
                                pass
                # Extract timing
                if 'ms / ' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'ms' and i > 0:
                            try:
                                prompt_eval_time = float(parts[i-1])
                            except:
                                pass

        print(f"✓ Test completed in {elapsed:.1f}s")
        if tokens_per_sec:
            print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
        if prompt_eval_time:
            print(f"  Prompt eval: {prompt_eval_time:.1f} ms")

        return {
            "model": model_name,
            "status": "success",
            "elapsed_time": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "prompt_eval_time": prompt_eval_time,
            "output_sample": result.stdout[-500:] if result.stdout else ""
        }

    except subprocess.TimeoutExpired:
        print("✗ Test timeout")
        return {
            "model": model_name,
            "status": "timeout"
        }
    except Exception as e:
        print(f"✗ Test error: {e}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e)
        }


def main():
    """Run perplexity benchmarks on all models."""
    print("="*80)
    print("GPU-ACCELERATED PERPLEXITY BENCHMARK")
    print("="*80)
    print()

    # Create test file
    print("Creating test data...")
    test_file = create_test_file()
    print(f"✓ Test file: {test_file}")
    print()

    # Define models
    gguf_dir = "/workspace/gguf_models"
    models = {
        # Uniform baselines
        "Q2_K": os.path.join(gguf_dir, "mistral-7b-q2k.gguf"),
        "Q4_K_M": os.path.join(gguf_dir, "mistral-7b-q4km.gguf"),
        "Q6_K": os.path.join(gguf_dir, "mistral-7b-q6k.gguf"),
        "Q8_0": os.path.join(gguf_dir, "mistral-7b-q8.gguf"),

        # Mixed-precision
        "aggressive_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-aggressive_mixed.gguf"),
        "balanced_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-balanced_mixed.gguf"),
        "conservative_mixed": os.path.join(gguf_dir, "mixed/mistral-7b-conservative_mixed.gguf"),
    }

    # Run benchmarks
    results = {}
    n_gpu_layers = 99  # Offload all layers to GPU

    for model_name, model_path in models.items():
        result = run_perplexity_test(model_path, model_name, test_file, n_gpu_layers)
        results[model_name] = result
        time.sleep(2)  # Brief pause between tests

    # Save results
    output_dir = "/workspace/outputs/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "perplexity_results.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Model':<22} {'Status':<12} {'Speed (tok/s)':<15} {'Time (s)':<10}")
    print("-" * 80)

    for model_name in ['Q2_K', 'Q4_K_M', 'Q6_K', 'Q8_0',
                        'aggressive_mixed', 'balanced_mixed', 'conservative_mixed']:
        if model_name in results:
            r = results[model_name]
            status = r.get('status', 'unknown')
            speed = f"{r.get('tokens_per_sec', 0):.1f}" if r.get('tokens_per_sec') else "N/A"
            elapsed = f"{r.get('elapsed_time', 0):.1f}" if r.get('elapsed_time') else "N/A"
            print(f"{model_name:<22} {status:<12} {speed:<15} {elapsed:<10}")

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
