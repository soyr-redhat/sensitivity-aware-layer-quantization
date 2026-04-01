#!/usr/bin/env python3
"""
Comprehensive benchmark suite using lm-evaluation-harness.
Matches the rigor of RedHatAI's quantization evaluations.

Runs standard benchmarks:
- GSM8k: Math reasoning
- MMLU-Pro: General knowledge
- IfEval: Instruction following
- GPQA: Science Q&A
- Math: Mathematical problems

Usage:
    python scripts/05_benchmark_lmeval.py \\
        --baseline models/mistral-7b-f16.gguf \\
        --quantized models/mistral-7b-conservative-mixed.gguf \\
        --output results/benchmarks.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# Benchmark tasks matching RedHatAI's evaluation suite
BENCHMARK_TASKS = {
    "gsm8k": {
        "name": "GSM8k",
        "task": "gsm8k",
        "num_fewshot": 0,
        "description": "Grade school math reasoning"
    },
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "task": "mmlu_pro",
        "num_fewshot": 0,
        "description": "Multi-task language understanding"
    },
    "ifeval": {
        "name": "IfEval",
        "task": "ifeval",
        "num_fewshot": 0,
        "description": "Instruction following"
    },
    "gpqa": {
        "name": "GPQA Diamond",
        "task": "gpqa_diamond",
        "num_fewshot": 0,
        "description": "Graduate-level science Q&A"
    },
    "math": {
        "name": "Math",
        "task": "math_500",
        "num_fewshot": 0,
        "description": "Mathematical problem solving"
    }
}


def run_lmeval(
    model_path: str,
    tasks: List[str],
    num_fewshot: int = 0,
    batch_size: str = "auto"
) -> Dict:
    """
    Run lm-evaluation-harness on a GGUF model.

    Args:
        model_path: Path to GGUF model file
        tasks: List of task names to evaluate
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of results per task
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Construct lm-eval command
    # Note: For GGUF models, we need to use the llama.cpp backend
    cmd = [
        "lm_eval",
        "--model", "gguf",
        "--model_args", f"filename={model_path}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
        "--output_path", "results/",
        "--log_samples"
    ]

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse output (lm-eval outputs JSON results)
        # This is a simplified parser - actual implementation may need adjustment
        print(result.stdout)

        return parse_lmeval_output(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error running lm-eval: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def parse_lmeval_output(output: str) -> Dict:
    """
    Parse lm-eval output to extract scores.

    Returns:
        Dictionary mapping task names to accuracy scores
    """
    # lm-eval outputs results in JSON format
    # This is a placeholder - actual parsing depends on output format
    results = {}

    # TODO: Parse the actual JSON output from lm-eval
    # For now, return empty dict
    return results


def calculate_recovery(baseline_score: float, quantized_score: float) -> float:
    """
    Calculate recovery percentage: (quantized / baseline) * 100
    """
    if baseline_score == 0:
        return 0.0
    return (quantized_score / baseline_score) * 100


def format_results_table(baseline_results: Dict, quantized_results: Dict, model_name: str) -> str:
    """
    Format results in a markdown table matching RedHatAI's style.

    Returns:
        Markdown formatted table string
    """
    lines = []
    lines.append(f"# Benchmark Results: {model_name}\n")
    lines.append("| Benchmark | Baseline | Quantized | Recovery (%) |")
    lines.append("|-----------|----------|-----------|--------------|")

    for task_id, task_info in BENCHMARK_TASKS.items():
        task_name = task_info["name"]
        baseline = baseline_results.get(task_id, 0.0)
        quantized = quantized_results.get(task_id, 0.0)
        recovery = calculate_recovery(baseline, quantized)

        lines.append(
            f"| {task_name} ({task_info['num_fewshot']}-shot) | "
            f"{baseline:.2f} | {quantized:.2f} | {recovery:.2f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks on quantized models"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline model (F16 or Q8_0)"
    )
    parser.add_argument(
        "--quantized",
        type=str,
        required=True,
        help="Path to quantized model to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_results.json",
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list of tasks, or 'all' for all tasks"
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the model (for results display)"
    )

    args = parser.parse_args()

    # Determine which tasks to run
    if args.tasks == "all":
        tasks_to_run = list(BENCHMARK_TASKS.keys())
    else:
        tasks_to_run = [t.strip() for t in args.tasks.split(",")]

    # Validate tasks
    for task in tasks_to_run:
        if task not in BENCHMARK_TASKS:
            print(f"Error: Unknown task '{task}'")
            print(f"Available tasks: {', '.join(BENCHMARK_TASKS.keys())}")
            sys.exit(1)

    # Extract task names for lm-eval
    lmeval_tasks = [BENCHMARK_TASKS[t]["task"] for t in tasks_to_run]

    print("="*80)
    print("COMPREHENSIVE QUANTIZATION BENCHMARK")
    print("="*80)
    print(f"Baseline model: {args.baseline}")
    print(f"Quantized model: {args.quantized}")
    print(f"Tasks: {', '.join([BENCHMARK_TASKS[t]['name'] for t in tasks_to_run])}")
    print("="*80)

    # Run baseline benchmarks
    print("\n📊 Running baseline model benchmarks...")
    baseline_results = run_lmeval(
        args.baseline,
        lmeval_tasks,
        batch_size=args.batch_size
    )

    # Run quantized benchmarks
    print("\n📊 Running quantized model benchmarks...")
    quantized_results = run_lmeval(
        args.quantized,
        lmeval_tasks,
        batch_size=args.batch_size
    )

    # Calculate recovery percentages
    results = {
        "baseline": args.baseline,
        "quantized": args.quantized,
        "tasks": {},
        "summary": {
            "avg_recovery": 0.0,
            "min_recovery": 100.0,
            "max_recovery": 0.0
        }
    }

    recoveries = []
    for task_id in tasks_to_run:
        baseline_score = baseline_results.get(task_id, 0.0)
        quantized_score = quantized_results.get(task_id, 0.0)
        recovery = calculate_recovery(baseline_score, quantized_score)
        recoveries.append(recovery)

        results["tasks"][task_id] = {
            "name": BENCHMARK_TASKS[task_id]["name"],
            "baseline": baseline_score,
            "quantized": quantized_score,
            "recovery": recovery
        }

    # Calculate summary statistics
    if recoveries:
        results["summary"]["avg_recovery"] = sum(recoveries) / len(recoveries)
        results["summary"]["min_recovery"] = min(recoveries)
        results["summary"]["max_recovery"] = max(recoveries)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Print formatted table
    model_name = args.model_name or Path(args.quantized).stem
    table = format_results_table(baseline_results, quantized_results, model_name)
    print("\n" + table)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average recovery: {results['summary']['avg_recovery']:.2f}%")
    print(f"Min recovery: {results['summary']['min_recovery']:.2f}%")
    print(f"Max recovery: {results['summary']['max_recovery']:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
