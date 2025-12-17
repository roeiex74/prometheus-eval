import json
import os
from pathlib import Path
from datetime import datetime

def generate_manifest(results_dir="results/experiments", output_file="dashboard/manifest.json"):
    """
    Scans the results directory and generates a manifest JSON file for the dashboard.
    """
    base_path = Path(results_dir)
    experiments = []

    if not base_path.exists():
        print(f"Warning: {results_dir} does not exist.")
        return

    # Scan for experiment directories
    for entry in base_path.iterdir():
        if entry.is_dir():
            results_file = entry / "results.json"
            summary_file = entry / "summary.json"
            
            if results_file.exists() and summary_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Extract key metadata
                    exp_data = {
                        "id": results_data.get("experiment_id", entry.name),
                        "timestamp": results_data.get("timestamp", datetime.now().isoformat()),
                        "dataset": results_data.get("dataset", "unknown"),
                        "best_variator": summary_data.get("best_variator", "N/A"),
                        "variators": list(summary_data.get("accuracies", {}).keys()),
                        "accuracies": summary_data.get("accuracies", {}),
                        "path": str(entry)
                    }
                    experiments.append(exp_data)
                except Exception as e:
                    print(f"Error reading {entry.name}: {e}")

    # Sort by timestamp descending
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)

    # Save manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "experiments": experiments
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest generated at {output_file} with {len(experiments)} experiments.")

if __name__ == "__main__":
    generate_manifest()
