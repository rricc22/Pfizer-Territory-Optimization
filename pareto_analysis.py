"""
Pfizer Territory Optimization - Complete Pareto Analysis
Generates Pareto frontiers for 3 workload scenarios and creates visualizations

Authors: Decision Modelling Project 2025-26
Date: November 2025
"""

from pfizer_optimization import PfizerOptimization
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

def plot_pareto_comparison(pareto_results: dict, output_file: str = 'pareto_comparison.png'):
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
    ax.set_ylabel('Total Distance (km)', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier Comparison: Distance vs Disruption\nThree Workload Balance Scenarios', 
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
              label=f'Min Distance: {df.loc[min_dist_idx, "distance"]:.2f} km')
    
    ax.scatter(df.loc[min_disr_idx, 'disruption'], 
              df.loc[min_disr_idx, 'distance'],
              color='green', s=200, marker='*', zorder=5,
              label=f'Min Disruption: {df.loc[min_disr_idx, "disruption"]:.4f}')
    
    ax.set_xlabel('Disruption (Weighted Index Sum)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Distance (km)', fontsize=13, fontweight='bold')
    ax.set_title(f'Pareto Frontier: {scenario_name}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {scenario_name} plot to {output_file}")
    plt.close()


def plot_assignment_heatmap(solution: dict, title: str, output_file: str, 
                            bricks: list, srs: list):
    """Create heatmap showing brick-SR assignments"""
    # Create matrix
    assignment_matrix = np.zeros((len(bricks), len(srs)))
    
    for j_idx, j in enumerate(srs):
        for i in solution['assignment'][j]:
            i_idx = bricks.index(i)
            assignment_matrix[i_idx, j_idx] = 1
    
    fig, ax = plt.subplots(figsize=(8, 12))
    
    sns.heatmap(assignment_matrix, 
                xticklabels=[f'SR{j}' for j in srs],
                yticklabels=[f'B{i}' for i in bricks],
                cmap=['white', '#2E86AB'],
                cbar=False,
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sales Representative', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brick', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved assignment heatmap to {output_file}")
    plt.close()


def plot_workload_comparison(solutions: dict, output_file: str):
    """
    Compare workload distributions across different solutions
    
    Args:
        solutions: Dict with solution names as keys, solution dicts as values
    """
    fig, axes = plt.subplots(1, len(solutions), figsize=(5*len(solutions), 5))
    
    if len(solutions) == 1:
        axes = [axes]
    
    for idx, (name, sol) in enumerate(solutions.items()):
        srs = sorted(sol['workloads'].keys())
        workloads = [sol['workloads'][j] for j in srs]
        
        colors = ['#2E86AB' if 0.8 <= w <= 1.2 else '#F18F01' for w in workloads]
        
        axes[idx].bar([f'SR{j}' for j in srs], workloads, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
        axes[idx].axhline(y=0.8, color='red', linestyle=':', linewidth=1.5, label='Min (0.8)')
        axes[idx].axhline(y=1.2, color='red', linestyle=':', linewidth=1.5, label='Max (1.2)')
        
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Workload', fontsize=11)
        axes[idx].set_ylim(0, 1.4)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Workload Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved workload comparison to {output_file}")
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
            'Min Distance (km)': df.loc[min_dist_idx, 'distance'],
            'Max Distance (km)': df['distance'].max(),
            'Min Disruption': df.loc[min_disr_idx, 'disruption'],
            'Max Disruption': df['disruption'].max(),
            'Disruption at Min Dist': df.loc[min_dist_idx, 'disruption'],
            'Distance at Min Disr': df.loc[min_disr_idx, 'distance']
        })
    
    return pd.DataFrame(summary)


def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "PFIZER TERRITORY OPTIMIZATION")
    print(" "*15 + "Complete Pareto Frontier Analysis")
    print("="*80 + "\n")
    
    # Initialize optimizer
    opt = PfizerOptimization()
    opt.load_data()
    
    # Analyze current assignment
    print("\n" + "="*80)
    print("CURRENT ASSIGNMENT ANALYSIS")
    print("="*80)
    current_sol = opt.analyze_current_assignment()
    opt.print_solution_comparison(current_sol, "Current Assignment")
    
    # Define scenarios
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
        
        df = opt.epsilon_constraint_method(wl_min, wl_max, num_points=25, verbose=False)
        pareto_results[scenario_name] = df
        
        # Store extreme solutions
        min_dist_idx = df['distance'].idxmin()
        min_disr_idx = df['disruption'].idxmin()
        
        pareto_solutions[f"{scenario_name} - Min Distance"] = {
            'assignment': df.loc[min_dist_idx, 'assignment'],
            'workloads': df.loc[min_dist_idx, 'workloads'],
            'distances': {j: sum(opt.distances[i, j] for i in df.loc[min_dist_idx, 'assignment'][j]) 
                         for j in opt.srs},
            'total_distance': df.loc[min_dist_idx, 'distance']
        }
        
        pareto_solutions[f"{scenario_name} - Min Disruption"] = {
            'assignment': df.loc[min_disr_idx, 'assignment'],
            'workloads': df.loc[min_disr_idx, 'workloads'],
            'distances': {j: sum(opt.distances[i, j] for i in df.loc[min_disr_idx, 'assignment'][j]) 
                         for j in opt.srs},
            'total_distance': df.loc[min_disr_idx, 'distance']
        }
        
        print(f"\n✓ Scenario complete: {len(df)} Pareto-optimal solutions found")
        print(f"  Distance range: [{df['distance'].min():.2f}, {df['distance'].max():.2f}] km")
        print(f"  Disruption range: [{df['disruption'].min():.4f}, {df['disruption'].max():.4f}]")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save Pareto data to pickle for later use
    with open('pareto_results.pkl', 'wb') as f:
        pickle.dump({
            'pareto_frontiers': pareto_results,
            'solutions': pareto_solutions,
            'scenarios': scenarios,
            'timestamp': datetime.now().isoformat()
        }, f)
    print("✓ Saved Pareto results to pareto_results.pkl")
    
    # Save to CSV
    for scenario_name, df in pareto_results.items():
        filename = f"pareto_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.csv"
        df[['distance', 'disruption', 'epsilon']].to_csv(filename, index=False)
        print(f"✓ Saved {scenario_name} data to {filename}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Comparison plot
    plot_pareto_comparison(pareto_results, 'pareto_comparison.png')
    
    # 2. Individual Pareto plots
    for scenario_name, df in pareto_results.items():
        filename = f"pareto_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.png"
        plot_individual_pareto(df, scenario_name, filename)
    
    # 3. Workload comparisons
    for scenario_name in scenarios.keys():
        sol_dict = {
            'Min Distance': pareto_solutions[f"{scenario_name} - Min Distance"],
            'Min Disruption': pareto_solutions[f"{scenario_name} - Min Disruption"],
            'Current': current_sol
        }
        filename = f"workload_{scenario_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace(':', '')}.png"
        plot_workload_comparison(sol_dict, filename)
    
    # 4. Assignment heatmaps for Scenario 1
    scenario1_name = 'Scenario 1: [0.8, 1.2]'
    plot_assignment_heatmap(
        pareto_solutions[f"{scenario1_name} - Min Distance"],
        'Assignment: Scenario 1 - Min Distance',
        'assignment_heatmap_scenario1_mindist.png',
        opt.bricks, opt.srs
    )
    plot_assignment_heatmap(
        pareto_solutions[f"{scenario1_name} - Min Disruption"],
        'Assignment: Scenario 1 - Min Disruption',
        'assignment_heatmap_scenario1_mindisr.png',
        opt.bricks, opt.srs
    )
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary_df = create_summary_table(pareto_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('pareto_summary.csv', index=False)
    print("\n✓ Saved summary table to pareto_summary.csv")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n1. SCENARIO COMPARISON:")
    for scenario_name, df in pareto_results.items():
        min_dist = df['distance'].min()
        improvement = ((187.41 - min_dist) / 187.41) * 100
        print(f"\n   {scenario_name}")
        print(f"   • Best distance: {min_dist:.2f} km ({improvement:.1f}% improvement)")
        print(f"   • Pareto solutions: {len(df)}")
        print(f"   • Trade-off range: {df['distance'].max() - df['distance'].min():.2f} km")
    
    print("\n2. WORKLOAD FLEXIBILITY:")
    s1_sols = len(pareto_results['Scenario 1: [0.8, 1.2]'])
    s3_sols = len(pareto_results['Scenario 3: [0.9, 1.1]'])
    print(f"   • Wide bounds [0.8, 1.2]: {s1_sols} solutions")
    print(f"   • Tight bounds [0.9, 1.1]: {s3_sols} solutions")
    print(f"   • Flexibility impact: {((s1_sols - s3_sols) / s3_sols * 100):.0f}% more solutions with wider bounds")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • pareto_results.pkl - Complete results data")
    print("  • pareto_*.csv - Pareto frontier data for each scenario")
    print("  • pareto_*.png - Pareto frontier plots")
    print("  • workload_*.png - Workload distribution comparisons")
    print("  • assignment_heatmap_*.png - Assignment visualizations")
    print("  • pareto_summary.csv - Summary statistics table")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
