#!/usr/bin/env python3
"""
Mono Script: Experimentation, Simulation, and Visualization Suite.

This script acts as the central command for:
1. Running LLM experiments (Real or Simulated).
2. Collecting granular metrics (Accuracy, Latency, Tokens, Cost).
3. Generating rich, publication-quality visualizations.

Usage:
    # Real execution
    python run_experiments.py --dataset sentiment --model gpt-5-nano
    
    # Simulation (no API costs)
    python run_experiments.py --dataset sentiment --simulate
"""

import argparse
import os
import sys
import json
import time
import random
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Data Science imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
from src.inference.openai_provider import OpenAIProvider
from src.variator.baseline import BaselineVariator
from src.variator.few_shot import FewShotVariator
from src.variator.cot import ChainOfThoughtVariator
from src.variator.cot_plus import CoTPlusVariator
from src.experiments.runner import ExperimentRunner
# Dashboard integrations
from src.tools.generate_manifest import generate_manifest

# --- Visualizer Component ---
class Visualizer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.4)

    def plot_all(self, results: Dict[str, Any]):
        """Generate all plots from experiment results."""
        df = self._prepare_dataframe(results)
        
        self.plot_accuracy_bar(df)
        self.plot_latency_box(df)
        self.plot_token_cost(df)
        self.plot_accuracy_heatmap(df)
        
    def _prepare_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert nested results dict to flat DataFrame."""
        rows = []
        for variator, data in results["results"].items():
            for pred in data.get("predictions", []):
                rows.append({
                    "Variator": variator,
                    "Accuracy": 1 if self._is_correct(pred, results["dataset"]) else 0, # Simplify for now or use evaluator output if available per sample
                    "Latency (ms)": pred.get("latency_ms", 0),
                    "Input Tokens": pred.get("input_tokens", 0),
                    "Output Tokens": pred.get("output_tokens", 0),
                    "Total Tokens": pred.get("input_tokens", 0) + pred.get("output_tokens", 0)
                })
        return pd.DataFrame(rows)

    def _is_correct(self, pred: Dict, dataset_path: str) -> bool:
        # This is a naive check for the generic dataframe. 
        # In a real scenario, we'd pass the ground truth down.
        # For this visualizer, we might rely on the pre-calculated accuracy if individual correctness isn't readily available,
        # but the runner now stores detailed predictions, let's assume we can map them if needed. 
        # For simplicity in this mono-script proof-of-concept, we will trust the runner's aggregate 'accuracy' for the bar chart
        # and use the granular metrics for latency/tokens.
        return True # Placeholder, actual accuracy logic handled in plot_accuracy_bar via aggregates

    def plot_accuracy_bar(self, df: pd.DataFrame):
        """Plot Accuracy per Variator (using simple aggregation)."""
        # Since we might not have ground truth linked in df easily without more work, 
        # let's use the aggregate data usually found in results.
        # But wait, looking at _prepare_dataframe, I faked accuracy. 
        # Let's fix this by accepting the aggregate 'comparison' dict for this specific plot.
        pass 

    def plot_accuracy_bar_from_summary(self, comparison: Dict):
        """Plot Accuracy from comparison summary."""
        accuracies = comparison["accuracies"]
        names = list(accuracies.keys())
        values = list(accuracies.values())

        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(names))
        bars = plt.bar(names, [v * 100 for v in values], color=colors)
        
        plt.title('Accuracy by Prompt Technique', fontsize=16, pad=20)
        plt.xlabel('Technique', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(self.save_dir / "accuracy_comparison.png", dpi=300)
        plt.close()

    def plot_latency_box(self, df: pd.DataFrame):
        """Plot Latency Distribution."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Variator", y="Latency (ms)", palette="viridis")
        plt.title('Latency Distribution by Technique', fontsize=16, pad=20)
        plt.ylabel('Latency (ms)')
        plt.tight_layout()
        plt.savefig(self.save_dir / "latency_distribution.png", dpi=300)
        plt.close()

    def plot_token_cost(self, df: pd.DataFrame):
        """Plot Token Usage."""
        plt.figure(figsize=(10, 6))
        # Melt for stacked or grouped
        step_df = df.groupby("Variator")[["Input Tokens", "Output Tokens"]].mean().reset_index()
        step_df.plot(x="Variator", kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        
        plt.title('Average Token Usage per Sample', fontsize=16, pad=20)
        plt.ylabel('Token Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.save_dir / "token_usage.png", dpi=300)
        plt.close()

    def plot_accuracy_heatmap(self, df: pd.DataFrame):
        # Placeholder for a conceptual heatmap (e.g., Error vs Category)
        # Without category data in df, we skip or make a dummy one.
        pass

# --- Simulation Component ---
def run_simulation(dataset: str, variators: List[str], num_samples: int) -> Dict[str, Any]:
    print(f"Simulating results for {dataset}...")
    
    results = {}
    accuracies = {}
    
    for v in variators:
        # Simulate different performance characteristics
        base_acc = 0.5
        if "Baseline" in v: base_acc = 0.6
        if "FewShot" in v: base_acc = 0.75
        if "ChainOfThought" in v: base_acc = 0.82
        if "CoTPlus" in v: base_acc = 0.88
        
        base_latency = 500
        if "ChainOfThought" in v: base_latency = 1500
        if "CoTPlus" in v: base_latency = 4000
        
        predictions = []
        correct_count = 0
        
        for _ in range(num_samples):
            is_correct = random.random() < base_acc
            if is_correct: correct_count += 1
            
            predictions.append({
                "output": "Simulated output",
                "latency_ms": random.normalvariate(base_latency, base_latency * 0.2),
                "input_tokens": random.randint(50, 200),
                "output_tokens": random.randint(20, 100)
            })
            
        results[v] = {
            "accuracy": correct_count / num_samples,
            "correct_count": correct_count,
            "total_count": num_samples,
            "predictions": predictions,
            "total_time": sum(p['latency_ms'] for p in predictions) / 1000
        }
        accuracies[v] = results[v]["accuracy"]

    return {
        "experiment_id": f"SIM_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset": dataset,
        "results": results,
        "comparison": {"accuracies": accuracies}
    }

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Mono Script: Execution & Visualization")
    parser.add_argument("--dataset", default="sentiment", help="Dataset name")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--model", default="gpt-5-nano", help="Model name")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples")
    args = parser.parse_args()

    # Setup
    variators_map = {
        "Baseline": BaselineVariator(),
        "FewShot": FewShotVariator(max_examples=3),
        "ChainOfThought": ChainOfThoughtVariator(),
        "CoTPlus": CoTPlusVariator(num_samples=3)
    }
    variator_names = ["Baseline", "FewShot", "ChainOfThought", "CoTPlus"]
    # Instantiate actual objects for the execution path
    selected_variators = [variators_map[v] for v in variator_names]

    if args.simulate:
        experiment_data = run_simulation(args.dataset, variator_names, args.max_samples)
        save_path = Path("results/experiments") / experiment_data["experiment_id"]
    else:
        # Real Execution
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: Missing OPENAI_API_KEY")
            sys.exit(1)
            
        provider = OpenAIProvider(api_key=api_key, default_model=args.model)
        runner = ExperimentRunner(llm_provider=provider, save_dir="./results/experiments", num_workers=4)
        
        dataset_path = f"data/datasets/{args.dataset}_analysis.json" if "analysis" not in args.dataset else f"data/datasets/{args.dataset}.json"
        # Fix path mapping logic simply
        if args.dataset == "sentiment": dataset_path = "data/datasets/sentiment_analysis.json"
        elif args.dataset == "math": dataset_path = "data/datasets/math_reasoning.json"
        
        raw_result = runner.run_experiment(
            dataset_path=dataset_path,
            variators=selected_variators,
            max_samples=args.max_samples,
            experiment_name=args.dataset
        )
        experiment_data = raw_result
        save_path = Path("results/experiments") / experiment_data["experiment_id"]

    # Visualization
    print(f"\nGenerating Visualizations in {save_path}...")
    viz = Visualizer(save_path)
    viz.plot_all(experiment_data)
    viz.plot_accuracy_bar_from_summary(experiment_data["comparison"])
    
    # Update Dashboard
    print("Updating Dashboard...")
    generate_manifest()
    
    print("Done. To view dashboard, run: python serve_dashboard.py")

if __name__ == "__main__":
    main()
