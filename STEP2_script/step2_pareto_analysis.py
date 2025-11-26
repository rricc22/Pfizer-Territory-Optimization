"""
Pfizer Territory Optimization - Step 2 Complete Pareto Analysis
Generates Pareto frontiers for 3 workload scenarios (100 bricks / 10 SRs)

Similar to Step 1 analysis but scaled to 100x10 instance

Authors: Decision Modelling Project 2025-26
Date: November 2025
"""

from step2_extensions import Step2Extensions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_pareto_comparison(pareto_results: dict, output_file: str = 'STEP2_Graph/pareto_comparison_scenarios.png'):
    """
    Plot all 3 Pareto frontiers on the same graph
    
    Args:
        pareto_results: Dict with keys as scenario names, values as DataFrames
        output_file: Output filename for plot
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for idx, (scenario_name, df) in enumerate(pareto_results.items()):
        ax.plot(df['disruption'], df['distance'], 
                marker=markers[idx], 
                color=colors[idx], 
                label=scenario_name,
                linewidth=2.5,
                markersize=8,
                alpha=0.8)
    
    ax.set_xlabel('Disruption (Weighted Index Sum)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Distance', fontsize=13, fontweight='bold')
    ax.set_title('Step 2 Pareto Frontier Comparison: Distance vs Disruption\n100 Bricks / 10 SRs - Three Workload Scenarios', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto comparison plot to {output_file}")
    plt.close()


def plot_individual_pareto(df: pd.DataFrame, scenario_name: str, output_file: str):
    """Plot individual Pareto frontier with annotations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Pareto curve
    ax.plot(df['disruption'], df['distance'], 
            marker='o', color='#2E86AB', linewidth=2.5, markersize=10,
            label='Pareto-optimal solutions')
    
    # Highlight extreme points
    min_dist_idx = df['distance'].idxmin()
    min_disr_idx = df['disruption'].idxmin()
    
    ax.scatter(df.loc[min_dist_idx, 'disruption'], 
              df.loc[min_dist_idx, 'distance'],
              color='red', s=200, marker='*', zorder=5,
              label=f'Min Distance: {df.loc[min_dist_idx, "distance"]:.2f}')
    
    ax.scatter(df.loc[min_disr_idx, 'disruption'], 
              df.loc[min_disr_idx, 'distance'],
              color='green', s=200, marker='*', zorder=5,
              label=f'Min Disruption: {df.loc[min_disr_idx, "disruption"]:.4f}')
    
    ax.set_xlabel('Disruption (Weighted Index Sum)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Distance', fontsize=13, fontweight='bold')
    ax.set_title(f'Step 2 Pareto Frontier: {scenario_name}\n100 Bricks / 10 SRs', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {scenario_name} plot to {output_file}")
    plt.close()


def plot_workload_distribution(df: pd.DataFrame, scenario_name: str, output_file: str, 
                               wl_min: float, wl_max: float):
    """
    Plot workload distribution across all Pareto solutions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Box plot of workload ranges
    workload_data = []
    for idx in df.index:
        workloads = list(df.loc[idx, 'workloads'].values())
        workload_data.append({
            'solution': idx,
            'min': min(workloads),
            'max': max(workloads),
            'mean': np.mean(workloads),
            'std': np.std(workloads)
        })
    
    wl_df = pd.DataFrame(workload_data)
    
    ax1.fill_between(wl_df['solution'], wl_df['min'], wl_df['max'], 
                     alpha=0.3, color='#2E86AB', label='Min-Max Range')
    ax1.plot(wl_df['solution'], wl_df['mean'], 'o-', 
            color='#2E86AB', linewidth=2, markersize=6, label='Mean')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
    ax1.axhline(y=wl_min, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=wl_max, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.fill_between(wl_df['solution'], wl_min, wl_max, 
                     alpha=0.1, color='red', label=f'Bounds [{wl_min}, {wl_max}]')
    
    ax1.set_xlabel('Solution Index (Pareto Frontier)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Workload', fontsize=12, fontweight='bold')
    ax1.set_title('Workload Range Across Pareto Solutions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Workload standard deviation
    ax2.plot(wl_df['solution'], wl_df['std'], 'o-', 
            color='#F18F01', linewidth=2, markersize=6)
    ax2.set_xlabel('Solution Index (Pareto Frontier)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Workload Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_title('Workload Balance Quality', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{scenario_name} - Workload Distribution Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved workload distribution to {output_file}")
    plt.close()


def plot_assignment_heatmap(solution: dict, title: str, output_file: str, 
                            bricks: list, srs: list):
    """Create heatmap showing brick-SR assignments (downsampled for 100 bricks)"""
    # Create matrix
    assignment_matrix = np.zeros((len(bricks), len(srs)))
    
    for j_idx, j in enumerate(srs):
        for i in solution['assignment'][j]:
            i_idx = bricks.index(i)
            assignment_matrix[i_idx, j_idx] = 1
    
    fig, ax = plt.subplots(figsize=(10, 16))
    
    sns.heatmap(assignment_matrix, 
                xticklabels=[f'SR{j}' for j in srs],
                yticklabels=[f'B{i}' if i % 10 == 1 else '' for i in bricks],  # Show every 10th brick
                cmap=['white', '#2E86AB'],
                cbar=False,
                linewidths=0,
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sales Representative', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brick (every 10th labeled)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved assignment heatmap to {output_file}")
    plt.close()


def create_summary_table(pareto_results: dict) -> pd.DataFrame:
    """Create summary table comparing scenarios"""
    summary = []
    
    for scenario_name, df in pareto_results.items():
        min_dist_idx = df['distance'].idxmin()
        min_disr_idx = df['disruption'].idxmin()
        
        summary.append({
            'Scenario': scenario_name,
            'Pareto Solutions': len(df),
            'Min Distance': df.loc[min_dist_idx, 'distance'],
            'Max Distance': df['distance'].max(),
            'Min Disruption': df.loc[min_disr_idx, 'disruption'],
            'Max Disruption': df['disruption'].max(),
            'Distance Range': df['distance'].max() - df['distance'].min(),
            'Avg Workload Max': df['workload_max'].mean(),
            'Avg Workload Std': df['workload_std'].mean()
        })
    
    return pd.DataFrame(summary)


def main():
    """Main analysis pipeline for Step 2"""
    print("\n" + "="*80)
    print(" "*20 + "PFIZER TERRITORY OPTIMIZATION - STEP 2")
    print(" "*10 + "Complete Pareto Frontier Analysis (100 Bricks / 10 SRs)")
    print("="*80 + "\n")
    
    # Initialize optimizer
    step2 = Step2Extensions()
    
    # Create output directory
    import os
    os.makedirs('STEP2_Graph', exist_ok=True)
    
    # Define scenarios (same as Step 1 for consistency)
    scenarios = {
        'Scenario 1: [0.8, 1.2]': (0.8, 1.2),
        'Scenario 2: [0.85, 1.15]': (0.85, 1.15),
        'Scenario 3: [0.9, 1.1]': (0.9, 1.1)
    }
    
    # Generate Pareto frontiers
    pareto_results = {}
    pareto_solutions = {}
    
    for scenario_name, (wl_min, wl_max) in scenarios.items():
        print(f"\n{'='*80}")
        print(f"GENERATING PARETO FRONTIER: {scenario_name}")
        print(f"{'='*80}")
        
        # Generate Pareto frontier using epsilon-constraint method
        df = step2.epsilon_constraint_pareto(wl_min, wl_max, num_points=20, verbose=False)
        pareto_results[scenario_name] = df
        
        # Store extreme solutions
        min_dist_idx = df['distance'].idxmin()
        min_disr_idx = df['disruption'].idxmin()
        
        pareto_solutions[f"{scenario_name} - Min Distance"] = {
            'assignment': df.loc[min_dist_idx, 'assignment'],
            'workloads': df.loc[min_dist_idx, 'workloads']
        }
        
        pareto_solutions[f"{scenario_name} - Min Disruption"] = {
            'assignment': df.loc[min_disr_idx, 'assignment'],
            'workloads': df.loc[min_disr_idx, 'workloads']
        }
        
        print(f"\n✓ Scenario complete: {len(df)} Pareto-optimal solutions found")
        print(f"  Distance range: [{df['distance'].min():.2f}, {df['distance'].max():.2f}]")
        print(f"  Disruption range: [{df['disruption'].min():.4f}, {df['disruption'].max():.4f}]")
        print(f"  Avg solve time: {df['solve_time'].mean():.2f}s")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save Pareto data to pickle for later use
    with open('STEP2_Graph/pareto_results_100x10.pkl', 'wb') as f:
        pickle.dump({
            'pareto_frontiers': pareto_results,
            'solutions': pareto_solutions,
            'scenarios': scenarios,
            'timestamp': datetime.now().isoformat()
        }, f)
    print("✓ Saved Pareto results to STEP2_Graph/pareto_results_100x10.pkl")
    
    # Save to CSV
    for scenario_name, df in pareto_results.items():
        filename = f"STEP2_Graph/pareto_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.csv"
        df[['distance', 'disruption', 'epsilon', 'workload_max', 'workload_min', 'workload_std', 'solve_time']].to_csv(filename, index=False)
        print(f"✓ Saved {scenario_name} data to {filename}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Comparison plot
    plot_pareto_comparison(pareto_results)
    
    # 2. Individual Pareto plots
    for scenario_name, df in pareto_results.items():
        filename = f"STEP2_Graph/pareto_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.png"
        plot_individual_pareto(df, scenario_name, filename)
    
    # 3. Workload distribution plots
    for scenario_name, (wl_min, wl_max) in scenarios.items():
        df = pareto_results[scenario_name]
        filename = f"STEP2_Graph/workload_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.png"
        plot_workload_distribution(df, scenario_name, filename, wl_min, wl_max)
    
    # 4. Assignment heatmaps for Scenario 1
    scenario1_name = 'Scenario 1: [0.8, 1.2]'
    plot_assignment_heatmap(
        pareto_solutions[f"{scenario1_name} - Min Distance"],
        'Step 2 Assignment: Scenario 1 - Min Distance',
        'STEP2_Graph/assignment_heatmap_scenario1_mindist.png',
        step2.bricks, step2.srs
    )
    plot_assignment_heatmap(
        pareto_solutions[f"{scenario1_name} - Min Disruption"],
        'Step 2 Assignment: Scenario 1 - Min Disruption',
        'STEP2_Graph/assignment_heatmap_scenario1_mindisr.png',
        step2.bricks, step2.srs
    )
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary_df = create_summary_table(pareto_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('STEP2_Graph/pareto_summary.csv', index=False)
    print("\n✓ Saved summary table to STEP2_Graph/pareto_summary.csv")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS - STEP 2 (100x10)")
    print("="*80)
    
    print("\n1. SCENARIO COMPARISON:")
    for scenario_name, df in pareto_results.items():
        min_dist = df['distance'].min()
        max_dist = df['distance'].max()
        improvement = ((max_dist - min_dist) / max_dist) * 100
        print(f"\n   {scenario_name}")
        print(f"   • Best distance: {min_dist:.2f} ({improvement:.1f}% improvement vs worst)")
        print(f"   • Pareto solutions: {len(df)}")
        print(f"   • Trade-off range: {max_dist - min_dist:.2f}")
        print(f"   • Avg solve time: {df['solve_time'].mean():.1f}s per solution")
    
    print("\n2. SCALABILITY (vs Step 1 22x4):")
    print(f"   • Instance size: 100 bricks × 10 SRs (vs 22 × 4)")
    print(f"   • Decision variables: ~1000 (vs ~88)")
    print(f"   • Avg solve time: {pareto_results['Scenario 1: [0.8, 1.2]']['solve_time'].mean():.1f}s (vs ~0.3s for 22x4)")
    print(f"   • Scalability factor: ~{pareto_results['Scenario 1: [0.8, 1.2]']['solve_time'].mean() / 0.3:.1f}x")
    
    print("\n3. WORKLOAD FLEXIBILITY:")
    s1_range = pareto_results['Scenario 1: [0.8, 1.2]']['distance'].max() - pareto_results['Scenario 1: [0.8, 1.2]']['distance'].min()
    s3_range = pareto_results['Scenario 3: [0.9, 1.1]']['distance'].max() - pareto_results['Scenario 3: [0.9, 1.1]']['distance'].min()
    print(f"   • Wide bounds [0.8, 1.2]: Distance range = {s1_range:.2f}")
    print(f"   • Tight bounds [0.9, 1.1]: Distance range = {s3_range:.2f}")
    print(f"   • Flexibility impact: {((s1_range - s3_range) / s3_range * 100):.0f}% more optimization potential with wider bounds")
    
    print("\n" + "="*80)
    print("STEP 2 PARETO ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files in STEP2_Graph/:")
    print("  • pareto_results_100x10.pkl - Complete results data")
    print("  • pareto_Scenario_*.csv - Pareto frontier data (3 files)")
    print("  • pareto_Scenario_*.png - Pareto frontier plots (3 files)")
    print("  • workload_Scenario_*.png - Workload distributions (3 files)")
    print("  • pareto_comparison_scenarios.png - Multi-scenario comparison")
    print("  • assignment_heatmap_*.png - Assignment visualizations (2 files)")
    print("  • pareto_summary.csv - Summary statistics table")
    print(f"\n  Total: 13 new files generated")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
