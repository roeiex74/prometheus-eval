#!/usr/bin/env python3
"""
Run Prompt Engineering Experiments

This script runs experiments comparing different prompt techniques:
- Baseline: Simple prompts
- Few-Shot: Prompts with 1-3 examples
- Chain-of-Thought: Step-by-step reasoning prompts
- CoT++: CoT with majority voting

Usage:
    python run_experiments.py --dataset sentiment --max-samples 10
    python run_experiments.py --dataset math --all
    python run_experiments.py --dataset logic
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.inference.openai_provider import OpenAIProvider
from src.variator.baseline import BaselineVariator
from src.variator.few_shot import FewShotVariator
from src.variator.cot import ChainOfThoughtVariator
from src.variator.cot_plus import CoTPlusVariator
from src.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt engineering experiments"
    )
    parser.add_argument(
        "--dataset",
        choices=["sentiment", "math", "logic", "all"],
        default="sentiment",
        help="Which dataset to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (for quick testing)"
    )
    parser.add_argument(
        "--variators",
        nargs="+",
        choices=["baseline", "fewshot", "cot", "cotplus", "all"],
        default=["all"],
        help="Which variators to test"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model to use"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please create a .env file with your API key")
        sys.exit(1)

    # Initialize LLM provider
    print(f"Initializing OpenAI provider with model: {args.model}")
    provider = OpenAIProvider(
        api_key=api_key,
        default_model=args.model,
        rpm_limit=60
    )

    # Initialize variators
    variators_map = {
        "baseline": BaselineVariator(),
        "fewshot": FewShotVariator(max_examples=3),
        "cot": ChainOfThoughtVariator(),
        "cotplus": CoTPlusVariator(num_samples=3),  # Reduced for cost
    }

    # Select variators to test
    if "all" in args.variators:
        selected_variators = list(variators_map.values())
    else:
        selected_variators = [variators_map[v] for v in args.variators]

    print(f"\nTesting {len(selected_variators)} variators:")
    for v in selected_variators:
        print(f"  - {v.__class__.__name__}")

    # Determine datasets to test
    dataset_paths = {
        "sentiment": "data/datasets/sentiment_analysis.json",
        "math": "data/datasets/math_reasoning.json",
        "logic": "data/datasets/logical_reasoning.json",
    }

    if args.dataset == "all":
        datasets_to_test = list(dataset_paths.items())
    else:
        datasets_to_test = [(args.dataset, dataset_paths[args.dataset])]

    # Initialize experiment runner
    runner = ExperimentRunner(
        llm_provider=provider,
        num_workers=4,  # Parallel processing
        save_dir="./results/experiments"
    )

    # Run experiments
    all_experiment_results = []

    for dataset_name, dataset_path in datasets_to_test:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        try:
            result = runner.run_experiment(
                dataset_path=dataset_path,
                variators=selected_variators,
                max_samples=args.max_samples,
                experiment_name=f"{dataset_name}"
            )

            all_experiment_results.append({
                "dataset": dataset_name,
                "result": result
            })

            # Print comparison
            print(f"\n{'-'*80}")
            print(f"Comparison Summary for {dataset_name}:")
            print(f"{'-'*80}")
            comparison = result["comparison"]

            print("\nAccuracy by Variator:")
            for variator_name, accuracy in sorted(
                comparison["accuracies"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {variator_name:25} {accuracy:6.2%}")

            if comparison["improvements"]:
                print("\nImprovement over Baseline:")
                for variator_name, improvement in sorted(
                    comparison["improvements"].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    print(f"  {variator_name:25} {improvement:+6.1f}%")

            print(f"\nBest Variator: {comparison['best_variator']}")

        except Exception as e:
            print(f"Error running experiment on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("All Experiments Complete!")
    print(f"{'='*80}")
    print("\nResults saved to: ./results/experiments/")
    print("\nNext steps:")
    print("  1. Review results in ./results/experiments/")
    print("  2. Run Jupyter notebook for visualization: notebooks/results_analysis.ipynb")
    print("  3. Generate publication-quality graphs for the report")


if __name__ == "__main__":
    main()
