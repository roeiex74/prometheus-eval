
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
    plt.style.use('ggplot') # Fallback
    
sns.set_palette("husl")

def get_latest_results_dir():
    results_dir = Path("results/experiments")
    # Join with cwd to ensure absolute path matching if needed, but Path works relative
    dirs = [d for d in results_dir.iterdir() if d.is_dir() and "sentiment" in d.name]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def plot_results(results_dir):
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found in {results_dir}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Prepare DataFrames
    accuracies = data.get("accuracies", {})
    times = data.get("times", {})
    
    variators = list(accuracies.keys())
    acc_values = [accuracies[v] for v in variators]
    time_values = [times.get(v, 0) for v in variators]

    df_acc = pd.DataFrame({"Variator": variators, "Accuracy": acc_values})
    df_time = pd.DataFrame({"Variator": variators, "Total Time (s)": time_values})

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_acc, x="Variator", y="Accuracy")
    plt.title(f"Accuracy by Prompt Technique\n({results_dir.name})")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    # Add values on bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f')
        
    plt.tight_layout()
    output_acc = results_dir / "accuracy_comparison.png"
    plt.savefig(output_acc)
    print(f"Saved accuracy plot to {output_acc}")

    # Plot Latency/Time
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_time, x="Variator", y="Total Time (s)")
    plt.title(f"Total Execution Time by Prompt Technique\n({results_dir.name})")
    plt.xticks(rotation=45)
    # Add values on bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f')

    plt.tight_layout()
    output_lat = results_dir / "latency_comparison.png"
    plt.savefig(output_lat)
    print(f"Saved latency plot to {output_lat}")
    
    # Also save to artifacts brain location for ease of access if needed? 
    # The user asked to "check the visualizations". 
    # I should strictly answer with the paths, the user can open them.

if __name__ == "__main__":
    latest_dir = get_latest_results_dir()
    if latest_dir:
        print(f"Processing latest results: {latest_dir}")
        plot_results(latest_dir)
    else:
        print("No results found.")
