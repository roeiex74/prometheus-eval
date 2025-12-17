"""
Experiment Runner

Orchestrates experiments across different prompt variators using multiprocessing.

Building Block Documentation:

Input Data:
    - dataset_path: str - Path to dataset JSON file
    - variators: List[BaseVariator] - List of variators to test
    - llm_provider: AbstractLLMProvider - LLM provider for inference

Output Data:
    - results: Dict - Complete experiment results with accuracy, timing, costs
    - comparison_data: Dict - Data formatted for visualization

Setup Data:
    - num_workers: int - Number of parallel workers (default: cpu_count)
    - max_samples: Optional[int] - Limit number of samples (for testing)
    - save_dir: str - Directory to save results
"""

import json
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional
from pathlib import Path
import traceback

from src.variator.base import BaseVariator
from src.inference.base import AbstractLLMProvider
from src.experiments.evaluator import AccuracyEvaluator


class ExperimentRunner:
    """
    Runs comparative experiments across multiple prompt variators.

    Uses multiprocessing to parallelize LLM API calls for faster execution.
    Tracks accuracy, timing, token usage, and costs for each variator.
    """

    def __init__(
        self,
        llm_provider: AbstractLLMProvider,
        num_workers: Optional[int] = None,
        save_dir: str = "./results/experiments"
    ):
        """
        Initialize experiment runner.

        Args:
            llm_provider: LLM provider for inference
            num_workers: Number of parallel workers (defaults to CPU count)
            save_dir: Directory to save experiment results
        """
        self.llm_provider = llm_provider
        self.num_workers = num_workers or min(cpu_count(), 4)  # Cap at 4 to avoid rate limits
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluator
        self.evaluator = AccuracyEvaluator(
            case_sensitive=False,
            normalize_whitespace=True,
            fuzzy_match=True,
            fuzzy_threshold=0.85
        )

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load dataset from JSON file.

        Args:
            dataset_path: Path to dataset file

        Returns:
            List of dataset items

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError(f"Dataset must be a JSON array, got {type(dataset)}")

        return dataset

    def run_experiment(
        self,
        dataset_path: str,
        variators: List[BaseVariator],
        max_samples: Optional[int] = None,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete experiment across all variators.

        Args:
            dataset_path: Path to dataset file
            variators: List of variators to test
            max_samples: Optional limit on samples (for quick testing)
            experiment_name: Optional name for this experiment

        Returns:
            Dictionary with complete experiment results
        """
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {experiment_name or 'Unnamed'}")
        print(f"{'='*60}\n")

        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = self.load_dataset(dataset_path)

        # Limit samples if specified
        if max_samples:
            dataset = dataset[:max_samples]
            print(f"Using {len(dataset)} samples (limited from full dataset)")
        else:
            print(f"Loaded {len(dataset)} samples")

        # Extract inputs and expected outputs
        inputs = [item["input"] for item in dataset]
        expected = [item["expected"] for item in dataset]

        # Run experiments for each variator
        all_results = {}
        for variator in variators:
            variator_name = variator.__class__.__name__
            print(f"\n{'-'*60}")
            print(f"Testing {variator_name}...")
            print(f"{'-'*60}")

            result = self._run_single_variator(
                variator=variator,
                inputs=inputs,
                expected=expected,
                dataset_items=dataset
            )

            all_results[variator_name] = result

            # Print summary
            print(f"\nResults for {variator_name}:")
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Correct: {result['correct_count']}/{result['total_count']}")
            print(f"  Total Time: {result['total_time']:.2f}s")
            print(f"  Avg Time per Sample: {result['avg_time_per_sample']:.2f}s")

        # Create comparison summary
        comparison = self._create_comparison(all_results)

        # Save results
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            experiment_id = f"{experiment_name}_{experiment_id}"

        self._save_results(experiment_id, all_results, comparison, dataset_path)

        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"Results saved to: {self.save_dir / experiment_id}")
        print(f"{'='*60}\n")

        return {
            "experiment_id": experiment_id,
            "dataset": dataset_path,
            "num_samples": len(dataset),
            "variators_tested": list(all_results.keys()),
            "results": all_results,
            "comparison": comparison,
        }

    def _run_single_variator(
        self,
        variator: BaseVariator,
        inputs: List[str],
        expected: List[str],
        dataset_items: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run experiment for a single variator.

        Args:
            variator: The variator to test
            inputs: List of input prompts
            expected: List of expected outputs
            dataset_items: Original dataset items

        Returns:
            Dictionary with variator results
        """
        start_time = time.time()

        # Generate prompts using variator
        prompts = []
        for input_text in inputs:
            try:
                # Handle different variator types
                if hasattr(variator, 'generate_prompt'):
                    if variator.__class__.__name__ == 'FewShotVariator':
                        # FewShot needs examples - use first 3 from dataset
                        examples = [
                            {"input": item["input"], "output": item["expected"]}
                            for item in dataset_items[:3]
                        ]
                        prompt_data = variator.generate_prompt(input_text, examples=examples)
                    else:
                        prompt_data = variator.generate_prompt(input_text)

                    prompts.append(prompt_data["prompt"])
            except Exception as e:
                print(f"Warning: Failed to generate prompt: {e}")
                prompts.append(input_text)  # Fallback to original

        # Get predictions using multiprocessing
        print(f"Running inference with {self.num_workers} workers...")
        predictions = self._run_parallel_inference(prompts)

        # Evaluate accuracy
        eval_results = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=expected,
            dataset_items=dataset_items
        )

        end_time = time.time()
        total_time = end_time - start_time

        return {
            **eval_results,
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(inputs) if inputs else 0,
            "variator_config": variator.get_metadata(),
            "predictions": predictions[:5],  # Save first 5 for inspection
        }

    def _run_parallel_inference(self, prompts: List[str]) -> List[str]:
        """
        Run LLM inference in parallel using multiprocessing.

        Args:
            prompts: List of prompts to process

        Returns:
            List of model responses
        """
        # For small datasets, run sequentially to avoid overhead
        if len(prompts) < self.num_workers:
            return [self._process_single_prompt(p) for p in prompts]

        # Use multiprocessing for larger datasets
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single_prompt, prompts)

        return results

    def _process_single_prompt(self, prompt: str) -> str:
        """
        Process a single prompt through the LLM.

        Args:
            prompt: The prompt to process

        Returns:
            Model response

        Note: This method is called by worker processes
        """
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=256
            )
            return response.strip()
        except Exception as e:
            print(f"Error processing prompt: {e}")
            traceback.print_exc()
            return ""

    def _create_comparison(self, all_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create comparison summary across all variators.

        Args:
            all_results: Results from all variators

        Returns:
            Comparison summary dictionary
        """
        comparison = {
            "accuracies": {},
            "times": {},
            "best_variator": None,
            "improvements": {},
        }

        baseline_accuracy = None
        baseline_name = None

        for variator_name, results in all_results.items():
            comparison["accuracies"][variator_name] = results["accuracy"]
            comparison["times"][variator_name] = results["total_time"]

            # Track baseline
            if "Baseline" in variator_name:
                baseline_accuracy = results["accuracy"]
                baseline_name = variator_name

        # Calculate improvements over baseline
        if baseline_accuracy is not None:
            for variator_name, accuracy in comparison["accuracies"].items():
                if variator_name != baseline_name:
                    improvement = ((accuracy - baseline_accuracy) / baseline_accuracy * 100
                                   if baseline_accuracy > 0 else 0)
                    comparison["improvements"][variator_name] = improvement

        # Find best variator
        if comparison["accuracies"]:
            comparison["best_variator"] = max(
                comparison["accuracies"].items(),
                key=lambda x: x[1]
            )[0]

        return comparison

    def _save_results(
        self,
        experiment_id: str,
        all_results: Dict,
        comparison: Dict,
        dataset_path: str
    ):
        """
        Save experiment results to files.

        Args:
            experiment_id: Unique experiment identifier
            all_results: Results from all variators
            comparison: Comparison summary
            dataset_path: Path to dataset used
        """
        experiment_dir = self.save_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = experiment_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_id": experiment_id,
                "dataset": str(dataset_path),
                "timestamp": datetime.now().isoformat(),
                "results": all_results,
                "comparison": comparison,
            }, f, indent=2)

        # Save comparison summary (easy to visualize)
        summary_file = experiment_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nResults saved to: {experiment_dir}")
