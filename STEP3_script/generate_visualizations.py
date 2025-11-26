"""
Generate visualizations for Step 3 Pareto frontier from CSV data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load data
df = pd.read_csv('STEP3_script/step3_pareto_22x4.csv')

# Create output directory
import os
os.makedirs('STEP3_Graph', exist_ok=True)

print(f"Loaded {len(df)} Pareto solutions")
print(f"Distance range: {df['distance'].min():.2f} - {df['distance'].max():.2f}")
print(f"Workload range: {df['workload_max'].min():.4f} - {df['workload_max'].max():.4f}")

# ========================================================================
# Figure 1: Main Pareto Frontier (Distance vs Workload)
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Plot Pareto curve
ax.plot(df['distance'], df['workload_max'], 
        'o-', linewidth=2.5, markersize=10, color='#2E86AB', 
        label='Pareto Frontier', zorder=3)

# Highlight extreme points
min_dist_idx = df['distance'].idxmin()
min_work_idx = df['workload_max'].idxmin()

ax.scatter(df.loc[min_dist_idx, 'distance'], df.loc[min_dist_idx, 'workload_max'],
          s=300, marker='D', color='#06A77D', edgecolors='black', linewidth=2,
          label='Min Distance', zorder=5)

ax.scatter(df.loc[min_work_idx, 'distance'], df.loc[min_work_idx, 'workload_max'],
          s=300, marker='s', color='#E63946', edgecolors='black', linewidth=2,
          label='Min Workload', zorder=5)

ax.set_xlabel('Total Distance', fontsize=13, fontweight='bold')
ax.set_ylabel('Maximum Workload', fontsize=13, fontweight='bold')
ax.set_title('Step 3: Office Relocation Pareto Frontier\n(22 Bricks / 4 SRs)', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('STEP3_Graph/pareto_22x4_frontier.png', dpi=300, bbox_inches='tight')
print("✓ Saved: STEP3_Graph/pareto_22x4_frontier.png")
plt.close()

# ========================================================================
# Figure 2: Trade-off Analysis (Dual Axis)
# ========================================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

# Distance on left axis
color1 = '#2E86AB'
ax1.set_xlabel('Solution Index (by α)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Distance', fontsize=13, fontweight='bold', color=color1)
ax1.plot(df.index, df['distance'], 'o-', color=color1, linewidth=2.5, 
        markersize=8, label='Distance')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Workload on right axis
ax2 = ax1.twinx()
color2 = '#E63946'
ax2.set_ylabel('Maximum Workload', fontsize=13, fontweight='bold', color=color2)
ax2.plot(df.index, df['workload_max'], 's-', color=color2, linewidth=2.5, 
        markersize=8, label='Max Workload')
ax2.tick_params(axis='y', labelcolor=color2)

# Title
ax1.set_title('Step 3: Distance vs Workload Trade-off\n(Varying α from 0 to 1)', 
             fontsize=14, fontweight='bold', pad=15)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=11)

plt.tight_layout()
plt.savefig('STEP3_Graph/pareto_22x4_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: STEP3_Graph/pareto_22x4_tradeoff.png")
plt.close()

# ========================================================================
# Figure 3: Relocation Analysis
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Displaced offices vs Distance (colored by workload)
ax = axes[0]
scatter = ax.scatter(df['distance'], df['displaced'], 
                    c=df['workload_max'], cmap='RdYlGn_r', 
                    s=150, edgecolors='black', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Displaced Offices', fontsize=12, fontweight='bold')
ax.set_title('Office Relocation Impact', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yticks([3, 4])
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Max Workload', fontsize=11)

# Right: Distribution of displaced offices
ax = axes[1]
displaced_counts = df['displaced'].value_counts().sort_index()
colors = ['#2E86AB', '#06A77D']
bars = ax.bar(displaced_counts.index, displaced_counts.values, 
             color=colors, edgecolor='black', linewidth=1.5, width=0.6)
ax.set_xlabel('Number of Displaced Offices', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Solutions', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Relocation Scenarios', fontsize=13, fontweight='bold')
ax.set_xticks([3, 4])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('STEP3_Graph/pareto_22x4_relocation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: STEP3_Graph/pareto_22x4_relocation.png")
plt.close()

# ========================================================================
# Figure 4: Alpha vs Objectives
# ========================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Normalize both objectives to 0-1 range for comparison
dist_norm = (df['distance'] - df['distance'].min()) / (df['distance'].max() - df['distance'].min())
work_norm = (df['workload_max'] - df['workload_max'].min()) / (df['workload_max'].max() - df['workload_max'].min())

ax.plot(df['alpha'], dist_norm, 'o-', linewidth=2.5, markersize=8, 
       color='#2E86AB', label='Distance (normalized)')
ax.plot(df['alpha'], work_norm, 's-', linewidth=2.5, markersize=8, 
       color='#E63946', label='Max Workload (normalized)')

ax.set_xlabel('Weight α (0=workload only, 1=distance only)', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalized Objective Value', fontsize=13, fontweight='bold')
ax.set_title('Step 3: Effect of Weight Parameter α on Objectives', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Add annotations for key alpha values
key_alphas = [0.0, 0.36, 1.0]
for alpha_val in key_alphas:
    if alpha_val in df['alpha'].values:
        idx = df[df['alpha'] == alpha_val].index[0]
        ax.axvline(x=alpha_val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(alpha_val, 1.02, f'α={alpha_val:.2f}', 
               ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('STEP3_Graph/pareto_22x4_alpha.png', dpi=300, bbox_inches='tight')
print("✓ Saved: STEP3_Graph/pareto_22x4_alpha.png")
plt.close()

print("\n✓ All visualizations generated successfully!")
