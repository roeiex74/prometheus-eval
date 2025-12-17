
#!/usr/bin/env python3
"""
Generate Publication-Quality Visualizations from Experiment Results

This script processes experiment results and generates 300 DPI visualizations
for all datasets (sentiment, math, logic).

Usage:
    python notebooks/generate_plots.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from pathlib import Path
import sys

# Configure plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('ggplot')  # Fallback

sns.set_palette("husl")


def get_all_experiment_dirs():
    """Find all experiment result directories for all datasets."""
    results_dir = Path("results/experiments")

    if not results_dir.exists():
        return []

    experiment_dirs = []
    for dataset_name in ['sentiment', 'math', 'logic']:
        # Find all directories matching dataset pattern
        pattern = str(results_dir / f"{dataset_name}*")
        dirs = glob.glob(pattern)

        if dirs:
            # Get most recent for this dataset
            latest = max(dirs, key=os.path.getmtime)
            experiment_dirs.append((dataset_name, Path(latest)))

    return experiment_dirs


def plot_results(results_dir, dataset_name):
    """Generate publication-quality plots at 300 DPI."""
    summary_path = results_dir / "summary.json"

    if not summary_path.exists():
        print(f"  ⚠️  No summary.json found in {results_dir}")
        return False

    print(f"\n  Processing {dataset_name} dataset results...")

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Prepare DataFrames
    accuracies = data.get("accuracies", {})
    times = data.get("times", {})

    if not accuracies:
        print(f"  ⚠️  No accuracy data found")
        return False

    variators = list(accuracies.keys())
    acc_values = [accuracies[v] for v in variators]
    time_values = [times.get(v, 0) for v in variators]

    df_acc = pd.DataFrame({"Variator": variators, "Accuracy": acc_values})
    df_time = pd.DataFrame({"Variator": variators, "Total Time (s)": time_values})

    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_acc, x="Variator", y="Accuracy", palette="viridis")
    plt.title(f"Accuracy by Prompt Technique\n{dataset_name.title()} Dataset",
              fontsize=16, fontweight='bold')
    plt.xlabel("Prompting Technique", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)

    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2%', fontsize=10, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_acc = results_dir / "accuracy_comparison.png"
    plt.savefig(output_acc, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_acc}")
    plt.close()

    # Plot 2: Latency/Time Comparison
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_time, x="Variator", y="Total Time (s)", palette="coolwarm")
    plt.title(f"Total Execution Time by Prompt Technique\n{dataset_name.title()} Dataset",
              fontsize=16, fontweight='bold')
    plt.xlabel("Prompting Technique", fontsize=12, fontweight='bold')
    plt.ylabel("Total Execution Time (seconds)", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f s', fontsize=10, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_lat = results_dir / "latency_comparison.png"
    plt.savefig(output_lat, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_lat}")
    plt.close()

    return True


def generate_combined_visualization(experiment_dirs):
    """Generate a combined visualization showing all datasets."""
    if not experiment_dirs:
        return

    print("\n  Generating combined visualization...")

    all_data = []
    for dataset_name, results_dir in experiment_dirs:
        summary_path = results_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                data = json.load(f)
                accuracies = data.get("accuracies", {})

                for variator, acc in accuracies.items():
                    all_data.append({
                        'Dataset': dataset_name.title(),
                        'Variator': variator,
                        'Accuracy': acc
                    })

    if not all_data:
        print("  ⚠️  No data available for combined visualization")
        return

    df_combined = pd.DataFrame(all_data)

    # Combined bar plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_combined, x='Variator', y='Accuracy', hue='Dataset', palette="Set2")
    plt.title("Accuracy Comparison Across All Datasets",
              fontsize=18, fontweight='bold')
    plt.xlabel("Prompting Technique", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(title='Dataset', fontsize=11, title_fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save to results/visualizations directory
    viz_dir = Path("results/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_combined = viz_dir / "overall_accuracy_comparison.png"
    plt.savefig(output_combined, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_combined}")
    plt.close()


def main():
    print("=" * 80)
    print("Generating Publication-Quality Visualizations (300 DPI)")
    print("=" * 80)

    experiment_dirs = get_all_experiment_dirs()

    if not experiment_dirs:
        print("\n❌ No experiment results found in results/experiments/")
        print("\nTo run experiments:")
        print("  python run_experiments.py --dataset all")
        return

    print(f"\nFound {len(experiment_dirs)} experiment result(s):")
    for dataset_name, path in experiment_dirs:
        print(f"  • {dataset_name}: {path.name}")

    # Generate individual plots for each dataset
    success_count = 0
    for dataset_name, results_dir in experiment_dirs:
        if plot_results(results_dir, dataset_name):
            success_count += 1

    # Generate combined visualization
    if success_count > 0:
        generate_combined_visualization(experiment_dirs)

    print("\n" + "=" * 80)
    print(f"✅ Successfully generated visualizations for {success_count} dataset(s)")
    print("=" * 80)

    if success_count > 0:
        print("\nVisualization locations:")
        for dataset_name, results_dir in experiment_dirs:
            print(f"  • {dataset_name}: {results_dir}/")
        print(f"  • Combined: results/visualizations/")


if __name__ == "__main__":
    main()
