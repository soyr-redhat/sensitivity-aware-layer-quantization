#!/usr/bin/env python3
"""Simple quality test: generate text samples from each model and compare.

Since we can't easily measure perplexity without proper tooling,
we'll generate samples and do basic quality assessment.
"""

import os
import json
import subprocess
import time

# Test prompts for generation
TEST_PROMPTS = [
    "The theory of relativity, developed by Albert Einstein,",
    "Machine learning is a subset of artificial intelligence that",
    "The process of photosynthesis in plants involves",
]


def generate_text(model_path, model_name, prompt, max_tokens=50):
    """Generate text from a model."""
    llama_cli = "/workspace/llama.cpp/build/bin/llama-cli"

    if not os.path.exists(model_path):
        return {"status": "not_found"}

    print(f"\n  Testing {model_name}...")
    print(f"  Prompt: '{prompt[:60]}...'")

    try:
        result = subprocess.run(
            [
                llama_cli,
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.7",
                "-ngl", "0",
                "-c", "512",
                "--log-disable",
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Extract generated text
        output = result.stdout
        # Simple parsing - get text after prompt
        if output:
            # llama-cli includes the prompt in output
            generated = output.strip()
            print(f"  ✓ Generated {len(generated)} chars")
            return {
                "status": "success",
                "generated": generated[:500],  # First 500 chars
            }
        else:
            print(f"  ✗ No output")
            return {"status": "no_output"}

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout")
        return {"status": "timeout"}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"status": "error", "error": str(e)}


def compare_models():
    """Generate samples from all models and save for comparison."""
    print("="*80)
    print("MODEL QUALITY COMPARISON - TEXT GENERATION TEST")
    print("="*80)
    print()

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

    results = {}

    for model_name, model_path in models.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        model_results = {}

        for i, prompt in enumerate(TEST_PROMPTS):
            result = generate_text(model_path, model_name, prompt, max_tokens=50)
            model_results[f"prompt_{i}"] = result
            time.sleep(1)  # Brief pause

        results[model_name] = model_results

    # Save results
    output_dir = "/workspace/outputs/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generation_samples.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f"\nOutput: {output_file}")
    print()

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Model':<22} {'Successful':<12} {'Failed':<10}")
    print("-" * 50)

    for model_name in ['Q2_K', 'Q4_K_M', 'Q6_K', 'Q8_0',
                        'aggressive_mixed', 'balanced_mixed', 'conservative_mixed']:
        if model_name in results:
            successes = sum(1 for r in results[model_name].values()
                           if r.get('status') == 'success')
            failures = len(results[model_name]) - successes
            print(f"{model_name:<22} {successes:<12} {failures:<10}")

    print()


if __name__ == '__main__':
    compare_models()
