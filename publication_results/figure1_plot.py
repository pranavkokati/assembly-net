#!/usr/bin/env python3
"""
Generate Figure 1: Assembly Non-Equivalence
Requires: matplotlib, numpy

Run: python figure1_plot.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
with open('publication_results/figure1_data.json', 'r') as f:
    data = json.load(f)

t1 = data['trajectory_1']
t2 = data['trajectory_2']

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel A: β₁(t) evolution
ax1 = axes[0]
ax1.plot(t1['times_normalized'], t1['beta1'],
         color=t1['color'], linewidth=2, label=t1['label'])
ax1.plot(t2['times_normalized'], t2['beta1'],
         color=t2['color'], linewidth=2, label=t2['label'])

# Mark loop formation times
for lt in t1['loop_formation_times_normalized'][:10]:
    ax1.axvline(lt, color=t1['color'], alpha=0.2, linestyle='--')
for lt in t2['loop_formation_times_normalized'][:10]:
    ax1.axvline(lt, color=t2['color'], alpha=0.2, linestyle='--')

ax1.set_xlabel('Normalized Time (t/T)', fontsize=11)
ax1.set_ylabel('β₁ (Cycle Count)', fontsize=11)
ax1.set_title('(A) Topology Evolution', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.set_xlim(0, 1)
ax1.grid(True, alpha=0.3)

# Panel B: Final structure comparison
ax2 = axes[1]
categories = ['Edges', 'Cycles (β₁)']
vals1 = [t1['final_edges'], t1['final_beta1']]
vals2 = [t2['final_edges'], t2['final_beta1']]

x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width/2, vals1, width, color=t1['color'], label=t1['label'])
ax2.bar(x + width/2, vals2, width, color=t2['color'], label=t2['label'])
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('(B) Final Structure\n(Nearly Identical)', fontsize=12, fontweight='bold')
ax2.legend()

# Panel C: Emergent properties comparison
ax3 = axes[2]
categories = ['ELDI', 'Mech. Score']
vals1 = [t1['eldi'], t1['mechanical_score']]
vals2 = [t2['eldi'], t2['mechanical_score']]

x = np.arange(len(categories))
bars1 = ax3.bar(x - width/2, vals1, width, color=t1['color'], label=t1['label'])
bars2 = ax3.bar(x + width/2, vals2, width, color=t2['color'], label=t2['label'])
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.set_ylabel('Value', fontsize=11)
ax3.set_title('(C) Emergent Properties\n(Different!)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.set_ylim(0, 1.1)

# Add class labels
ax3.annotate(t1['mechanical_class'], xy=(1 - width/2, vals1[1] + 0.05),
             ha='center', fontsize=9, color=t1['color'])
ax3.annotate(t2['mechanical_class'], xy=(1 + width/2, vals2[1] + 0.05),
             ha='center', fontsize=9, color=t2['color'])

plt.tight_layout()
plt.savefig('publication_results/figure1_assembly_nonequivalence.png', dpi=300, bbox_inches='tight')
plt.savefig('publication_results/figure1_assembly_nonequivalence.pdf', bbox_inches='tight')
print("Saved: figure1_assembly_nonequivalence.png/pdf")
plt.show()
