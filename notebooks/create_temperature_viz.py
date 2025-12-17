#!/usr/bin/env python3
"""
Create temperature sensitivity visualization from notebook data.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Temperature sensitivity data from the notebook (cell-5)
temperatures = [0.0, 0.2, 0.5, 0.7, 1.0]
semantic_stability = [0.98, 0.95, 0.88, 0.75, 0.60]
bleu_score = [0.85, 0.82, 0.75, 0.65, 0.45]
pass_at_1 = [0.90, 0.88, 0.80, 0.70, 0.50]

# Create visualization
plt.figure(figsize=(12, 7))
plt.plot(temperatures, semantic_stability, marker='o', linewidth=2.5,
         markersize=10, label='Semantic Stability', color='#2E86AB')
plt.plot(temperatures, bleu_score, marker='s', linewidth=2.5,
         markersize=10, label='BLEU Score', color='#A23B72')
plt.plot(temperatures, pass_at_1, marker='^', linewidth=2.5,
         markersize=10, label='Pass@1', color='#F18F01')

plt.title('Sensitivity Analysis: Metric Degradation with Temperature',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Temperature', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.4, 1.05)

# Add annotation for optimal temperature
plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, linewidth=2)
plt.text(0.7, 0.95, 'Optimal Temp\n(0.7)',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()

# Save to results/visualizations
viz_dir = Path('results/visualizations')
viz_dir.mkdir(parents=True, exist_ok=True)
output_path = viz_dir / 'temperature_sensitivity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")
plt.close()
