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



# Global worker support
_worker_provider = None

def init_worker(provider_class, provider_name, api_key, default_model, temperature, max_tokens, timeout):
    """Initialize the global provider in the worker process."""
    global _worker_provider
    print(f"DEBUG: init_worker called in pid {os.getpid()}")
    try:
        _worker_provider = provider_class(
            api_key=api_key,
            default_model=default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        print(f"DEBUG: _worker_provider initialized: {_worker_provider}")
    except Exception as e:
        print(f"Failed to initialize worker provider: {e}")
        import traceback
        traceback.print_exc()

def run_inference_task(prompt: str) -> str:
    """Top-level task function for worker processes."""
    global _worker_provider
    if _worker_provider is None:
        return "Error: Provider not initialized"
    
    start_time = time.time()
    try:
        response = _worker_provider.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=256
        )
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Calculate tokens
        input_tokens = _worker_provider.count_tokens(prompt)
        output_tokens = _worker_provider.count_tokens(response)
        
        return {
            "output": response.strip(),
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return {
            "output": "",
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e)
        }

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

        # Extract text for evaluation
        prediction_texts = [p["output"] for p in predictions]
        
        # Aggregate metrics
        total_input_tokens = sum(p["input_tokens"] for p in predictions)
        total_output_tokens = sum(p["output_tokens"] for p in predictions)
        avg_latency = sum(p["latency_ms"] for p in predictions) / len(predictions) if predictions else 0

        # Evaluate accuracy
        eval_results = self.evaluator.evaluate(
            predictions=prediction_texts,
            ground_truth=expected,
            dataset_items=dataset_items
        )

        end_time = time.time()
        total_time = end_time - start_time

        return {
            **eval_results,
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(inputs) if inputs else 0,
            "avg_latency_ms": avg_latency,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "variator_config": variator.get_metadata(),
            "predictions": predictions,  # Save all detailed predictions
        }


    def _run_parallel_inference(self, prompts: List[str]) -> List[str]:
        """
        Run LLM inference in parallel using multiprocessing.
        """
        # For small datasets, run sequentially (but we need to use the instance provider here)
        if len(prompts) < self.num_workers:
             return [self._process_single_prompt_sequential(p) for p in prompts]

        # Use multiprocessing for larger datasets
        # Extract config to pass to workers
        provider = self.llm_provider
        
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(
                type(provider),
                provider.provider_name,
                provider.api_key,
                provider.default_model,
                provider.temperature,
                provider.max_tokens,
                provider.timeout
            )
        ) as pool:
            results = pool.map(run_inference_task, prompts)

        return results

    def _process_single_prompt_sequential(self, prompt: str) -> Dict[str, Any]:
        """Sequential processing using the instance provider."""
        start_time = time.time()
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=256
            )
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            input_tokens = self.llm_provider.count_tokens(prompt)
            output_tokens = self.llm_provider.count_tokens(response)
            
            return {
                "output": response.strip(),
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return {
                "output": "",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e)
            }

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
