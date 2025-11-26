"""
Pfizer Territory Optimization - Step 3 Office Relocation
=========================================================

Step 3: Office relocation optimization where center bricks (office locations) are decision variables

Key objectives:
1. Total distance (minimize)
2. Workload fairness - MinMax (minimize maximum workload)
3. Number of relocated offices (minimize disruption)

Mathematical formulation based on colleague's work but adapted for Gurobi

Authors: Decision Modelling Project 2025-26
Date: November 2025
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Step3OfficeRelocation:
    """
    Step 3: SR Office Relocation Optimization
    
    Mathematical Model:
    -------------------
    Decision Variables:
      - x[i,j] ∈ {0,1}: brick i assigned to office at brick j
      - y[j] ∈ {0,1}: brick j contains an office
      - wm: maximum workload across all offices
    
    Constraints:
      1. Each brick → exactly one office: Σ_j x[i,j] = 1, ∀i
      2. Exactly n offices: Σ_j y[j] = n
      3. Assignment validity: x[i,j] ≤ y[j], ∀i,j
      4. Workload tracking: wm ≥ Σ_i w[i]·x[i,j], ∀j with y[j]=1
    
    Objectives:
      1. Minimize total distance: Σ_i,j d[i,j]·x[i,j]
      2. Minimize max workload: wm
      3. Minimize displaced offices: Σ_{j∈J0} (1-y[j]) where J0=initial offices
    """
    
    def __init__(self, data_path: str = 'data/data-100x10.xlsx'):
        """Initialize with data"""
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load 100 bricks / 10 SRs data"""
        print(f"\n{'='*70}")
        print("STEP 3: Loading data for office relocation...")
        print(f"{'='*70}")
        
        # Load and clean data
        df = pd.read_excel(self.data_path, skiprows=3)
        df.columns = ['id', 'x_coord', 'y_coord', 'index', 'current_zone', 'office']
        df = df[df['id'].notna()].copy()
        
        # Convert to proper types
        df['id'] = df['id'].astype(int)
        df['x_coord'] = df['x_coord'].astype(float)
        df['y_coord'] = df['y_coord'].astype(float)
        df['index'] = df['index'].astype(float)
        df['current_zone'] = df['current_zone'].astype(float).astype(int)
        df['office'] = df['office'].astype(int)
        
        self.df = df
        self.bricks = list(df['id'])
        self.n_bricks = len(self.bricks)
        self.n_srs = df['current_zone'].max()
        
        # Workload
        self.workload = dict(zip(df['id'], df['index']))
        
        # Coordinates
        self.coords = df[['x_coord', 'y_coord']].values
        
        # Initial offices
        self.initial_offices = list(df[df['office'] == 1]['id'])
        
        # Calculate distance matrix between ALL bricks
        self._calculate_distance_matrix()
        
        print(f"✓ Data loaded")
        print(f"  - {self.n_bricks} bricks")
        print(f"  - {self.n_srs} SRs")
        print(f"  - Total workload: {sum(self.workload.values()):.4f}")
        print(f"  - Initial offices: {self.initial_offices}")
        
    def _calculate_distance_matrix(self):
        """Calculate distance matrix between all bricks"""
        self.distance_matrix = np.zeros((self.n_bricks, self.n_bricks))
        
        for i in range(self.n_bricks):
            for j in range(self.n_bricks):
                dist = np.sqrt(
                    (self.coords[i][0] - self.coords[j][0])**2 + 
                    (self.coords[i][1] - self.coords[j][1])**2
                )
                self.distance_matrix[i, j] = dist
        
        # Create dictionary for easy access (1-indexed)
        self.distances = {}
        for i in self.bricks:
            for j in self.bricks:
                self.distances[(i, j)] = self.distance_matrix[i-1, j-1]
    
    # ========================================================================
    # Model 1: Minimize Total Distance
    # ========================================================================
    
    def model_minimize_distance(self, verbose: bool = True):
        """
        Minimize total distance with office relocation
        """
        print(f"\n{'='*70}")
        print("MODEL 1: Minimize Total Distance (Office Relocation)")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_MinDistance")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
        
        # Constraint 1: Each brick assigned to exactly one office
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        
        # Constraint 2: Exactly n offices
        m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
        
        # Constraint 3: Can only assign to brick with office
        m.addConstrs(
            (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
            name="AssignOnlyToOffice"
        )
        
        # Objective: Minimize total distance
        obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                         for i in self.bricks for j in self.bricks)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = self._extract_solution(m, x, y)
            print(f"\n✓ Optimal solution found")
            print(f"  Total distance: {solution['total_distance']:.4f}")
            print(f"  Office locations: {solution['office_locations']}")
            print(f"  Displaced offices: {solution['displaced_offices']}/{self.n_srs}")
            print(f"  Max workload: {solution['workload_max']:.4f}")
            
            return m, solution
        else:
            print(f"\n✗ No optimal solution (status: {m.status})")
            return m, None
    
    # ========================================================================
    # Model 2: Minimize Maximum Workload (MinMax)
    # ========================================================================
    
    def model_minimize_maxworkload(self, verbose: bool = True):
        """
        Minimize maximum workload (workload fairness) with office relocation
        """
        print(f"\n{'='*70}")
        print("MODEL 2: Minimize Maximum Workload (Office Relocation)")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_MinMaxWorkload")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
        wm = m.addVar(vtype=GRB.CONTINUOUS, name="wm")  # maximum workload
        
        # Constraints
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
        m.addConstrs(
            (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
            name="AssignOnlyToOffice"
        )
        
        # Workload tracking: wm ≥ workload of each office
        # Only need to track for bricks that have offices (y[j] = 1)
        # Use Big-M method: wm ≥ Σ_i w[i]·x[i,j] - M·(1-y[j])
        M = sum(self.workload.values()) + 1  # Big M
        m.addConstrs(
            (wm >= gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) - M * (1 - y[j])
             for j in self.bricks),
            name="MaxWorkload"
        )
        
        # Objective: Minimize maximum workload
        m.setObjective(wm, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = self._extract_solution(m, x, y)
            print(f"\n✓ Optimal solution found")
            print(f"  Max workload: {solution['workload_max']:.4f}")
            print(f"  Min workload: {solution['workload_min']:.4f}")
            print(f"  Workload std: {solution['workload_std']:.4f}")
            print(f"  Office locations: {solution['office_locations']}")
            print(f"  Displaced offices: {solution['displaced_offices']}/{self.n_srs}")
            print(f"  Total distance: {solution['total_distance']:.4f}")
            
            return m, solution
        else:
            print(f"\n✗ No optimal solution (status: {m.status})")
            return m, None
    
    # ========================================================================
    # Model 3: Bi-objective (Distance + Workload MinMax)
    # ========================================================================
    
    def model_biobjective(self, alpha: float = 0.5, verbose: bool = True):
        """
        Bi-objective model: weighted sum of distance and max workload
        
        Args:
            alpha: weight for distance (0 = only workload, 1 = only distance)
        """
        print(f"\n{'='*70}")
        print(f"BI-OBJECTIVE MODEL (alpha={alpha:.2f})")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_Biobjective")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
        wm = m.addVar(vtype=GRB.CONTINUOUS, name="wm")
        
        # Constraints
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
        m.addConstrs(
            (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
            name="AssignOnlyToOffice"
        )
        
        M = sum(self.workload.values()) + 1
        m.addConstrs(
            (wm >= gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) - M * (1 - y[j])
             for j in self.bricks),
            name="MaxWorkload"
        )
        
        # Bi-objective: weighted sum
        distance_obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                                  for i in self.bricks for j in self.bricks)
        
        # Normalize objectives for fair weighting
        # Rough normalization based on expected ranges
        distance_norm = distance_obj / 20  # Expected range ~10-20
        workload_norm = wm  # Expected range ~0.8-1.5
        
        obj = alpha * distance_norm + (1 - alpha) * workload_norm
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = self._extract_solution(m, x, y)
            print(f"\n✓ Optimal solution found")
            print(f"  Total distance: {solution['total_distance']:.4f}")
            print(f"  Max workload: {solution['workload_max']:.4f}")
            print(f"  Office locations: {solution['office_locations']}")
            print(f"  Displaced offices: {solution['displaced_offices']}/{self.n_srs}")
            
            return m, solution
        else:
            print(f"\n✗ No optimal solution (status: {m.status})")
            return m, None
    
    # ========================================================================
    # Model 4: Three-objective with epsilon-constraint
    # ========================================================================
    
    def model_three_objective_epsilon(self, num_relocated: int, 
                                     max_workload_eps: float = None,
                                     verbose: bool = False):
        """
        Three-objective model using epsilon-constraint:
        - Primary: Minimize distance
        - Constraint 1: Exactly num_relocated offices relocated
        - Constraint 2: Max workload ≤ max_workload_eps (optional)
        
        Args:
            num_relocated: Number of offices that must be relocated (0 to n_srs)
            max_workload_eps: Upper bound on maximum workload (optional)
        """
        m = gp.Model(f"Step3_ThreeObj_Reloc{num_relocated}")
        m.setParam('OutputFlag', 0 if not verbose else 1)
        
        # Decision variables
        x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
        y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
        wm = m.addVar(vtype=GRB.CONTINUOUS, name="wm")
        
        # Standard constraints
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
        m.addConstrs(
            (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
            name="AssignOnlyToOffice"
        )
        
        M = sum(self.workload.values()) + 1
        m.addConstrs(
            (wm >= gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) - M * (1 - y[j])
             for j in self.bricks),
            name="MaxWorkload"
        )
        
        # Epsilon constraint 1: Number of relocated offices
        # Relocated = offices NOT in initial positions
        # Count: Σ_{j not in J0} y[j] = num_relocated
        # Or equivalently: Σ_{j in J0} y[j] = n_srs - num_relocated
        initial_kept = self.n_srs - num_relocated
        m.addConstr(
            gp.quicksum(y[j] for j in self.initial_offices) == initial_kept,
            name="RelocatedOffices"
        )
        
        # Epsilon constraint 2: Maximum workload (optional)
        if max_workload_eps is not None:
            m.addConstr(wm <= max_workload_eps, name="MaxWorkloadEpsilon")
        
        # Objective: Minimize distance (with small tie-breaker for workload)
        distance_obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                                  for i in self.bricks for j in self.bricks)
        obj = distance_obj + 0.001 * wm  # Tie-breaker
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = self._extract_solution(m, x, y)
            solution['num_relocated_constraint'] = num_relocated
            return m, solution
        else:
            return m, None
    
    def generate_pareto_three_objectives(self, verbose: bool = False):
        """
        Generate Pareto frontier for three objectives:
        1. Total distance
        2. Maximum workload
        3. Number of relocated offices
        
        Strategy: For each possible number of relocations (0 to n_srs),
        generate a Pareto curve of distance vs workload
        """
        print(f"\n{'='*70}")
        print("GENERATING 3-OBJECTIVE PARETO FRONTIER")
        print(f"{'='*70}")
        
        results = []
        
        # For each possible number of relocated offices
        for num_relocated in range(self.n_srs + 1):
            print(f"\n  Scenario: {num_relocated} relocated offices...")
            
            # Find range of workloads for this relocation count
            # Min workload
            m1, sol1 = self.model_three_objective_epsilon(
                num_relocated=num_relocated,
                max_workload_eps=None,
                verbose=False
            )
            
            if sol1 is None:
                print(f"    No feasible solution for {num_relocated} relocations")
                continue
            
            # For this relocation count, generate multiple solutions
            # by varying max workload constraint
            min_workload = sol1['workload_max']
            
            # Try a few workload levels
            workload_levels = [min_workload + i * 0.05 for i in range(5)]
            
            for wl_eps in workload_levels:
                m, sol = self.model_three_objective_epsilon(
                    num_relocated=num_relocated,
                    max_workload_eps=wl_eps,
                    verbose=False
                )
                
                if sol:
                    sol['num_relocated'] = num_relocated
                    results.append({
                        'num_relocated': num_relocated,
                        'distance': sol['total_distance'],
                        'workload_max': sol['workload_max'],
                        'workload_min': sol['workload_min'],
                        'workload_std': sol['workload_std'],
                        'office_locations': sol['office_locations'],
                        'displaced_offices': sol['displaced_offices']
                    })
        
        df_results = pd.DataFrame(results)
        
        # Remove duplicates
        df_results = df_results.drop_duplicates(
            subset=['num_relocated', 'distance', 'workload_max']
        )
        
        print(f"\n✓ Generated {len(df_results)} Pareto-optimal solutions")
        print(f"  Relocation scenarios: {sorted(df_results['num_relocated'].unique())}")
        
        return df_results
    
    # ========================================================================
    # Helper methods
    # ========================================================================
    
    def _extract_solution(self, model, x, y):
        """Extract solution from optimized model"""
        if model.status != GRB.OPTIMAL:
            return None
        
        # Find office locations
        office_locations = []
        for j in self.bricks:
            if y[j].X > 0.5:
                office_locations.append(j)
        
        # Count displaced offices
        displaced = sum(1 for j in self.initial_offices if y[j].X < 0.5)
        
        # Extract assignments and calculate workloads
        assignments = {j: [] for j in office_locations}
        workloads = {j: 0 for j in office_locations}
        
        for i in self.bricks:
            for j in office_locations:
                if x[i, j].X > 0.5:
                    assignments[j].append(i)
                    workloads[j] += self.workload[i]
        
        # Calculate metrics
        total_distance = sum(
            self.distances[i, j] * x[i, j].X
            for i in self.bricks for j in self.bricks if y[j].X > 0.5
        )
        
        workload_values = list(workloads.values())
        
        return {
            'total_distance': total_distance,
            'office_locations': sorted(office_locations),
            'displaced_offices': displaced,
            'assignments': assignments,
            'workloads': workloads,
            'workload_max': max(workload_values),
            'workload_min': min(workload_values),
            'workload_mean': np.mean(workload_values),
            'workload_std': np.std(workload_values)
        }
    
    def visualize_solution(self, solution, title="Office Relocation Solution"):
        """Visualize solution with office locations and assignments"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Initial configuration
        ax = axes[0]
        ax.scatter(self.coords[:, 0], self.coords[:, 1], c='lightblue', s=30, alpha=0.6)
        
        # Initial offices
        for office_id in self.initial_offices:
            idx = office_id - 1
            ax.scatter(self.coords[idx, 0], self.coords[idx, 1], 
                      c='red', marker='*', s=300, edgecolors='black', linewidths=1.5,
                      label='Initial Office' if office_id == self.initial_offices[0] else '')
        
        ax.set_title('Initial Configuration', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: New configuration
        ax = axes[1]
        
        # Color map for offices
        colors = plt.cm.tab10(np.linspace(0, 1, len(solution['office_locations'])))
        
        # Draw assignments
        for idx, office_id in enumerate(solution['office_locations']):
            office_idx = office_id - 1
            office_coord = self.coords[office_idx]
            color = colors[idx]
            
            # Draw bricks assigned to this office
            for brick_id in solution['assignments'][office_id]:
                brick_idx = brick_id - 1
                brick_coord = self.coords[brick_idx]
                
                # Draw line from brick to office
                ax.plot([brick_coord[0], office_coord[0]], 
                       [brick_coord[1], office_coord[1]], 
                       c=color, alpha=0.2, linewidth=0.5)
                
                # Draw brick
                ax.scatter(brick_coord[0], brick_coord[1], 
                          c=[color], s=30, alpha=0.6)
            
            # Draw office
            marker = '*' if office_id in self.initial_offices else 's'
            ax.scatter(office_coord[0], office_coord[1], 
                      c=[color], marker=marker, s=300, edgecolors='black', linewidths=1.5)
        
        ax.set_title(f'{title}\n' + 
                    f'Distance: {solution["total_distance"]:.2f}, ' +
                    f'Max Workload: {solution["workload_max"]:.2f}, ' +
                    f'Displaced: {solution["displaced_offices"]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_pareto_3d(self, df_results):
        """Visualize 3D Pareto frontier"""
        fig = plt.figure(figsize=(16, 6))
        
        # 3D scatter
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(df_results['num_relocated'], 
                            df_results['distance'], 
                            df_results['workload_max'],
                            c=df_results['num_relocated'], 
                            cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Relocated Offices')
        ax1.set_ylabel('Total Distance')
        ax1.set_zlabel('Max Workload')
        ax1.set_title('3-Objective Pareto Frontier', fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Relocated')
        
        # 2D: Distance vs Relocations
        ax2 = fig.add_subplot(132)
        for reloc in sorted(df_results['num_relocated'].unique()):
            subset = df_results[df_results['num_relocated'] == reloc]
            ax2.scatter(subset['num_relocated'], subset['distance'], 
                       label=f'{reloc} relocated', s=100, alpha=0.7)
        ax2.set_xlabel('Number of Relocated Offices')
        ax2.set_ylabel('Total Distance')
        ax2.set_title('Distance vs Relocations', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 2D: Workload vs Relocations
        ax3 = fig.add_subplot(133)
        for reloc in sorted(df_results['num_relocated'].unique()):
            subset = df_results[df_results['num_relocated'] == reloc]
            ax3.scatter(subset['num_relocated'], subset['workload_max'], 
                       label=f'{reloc} relocated', s=100, alpha=0.7)
        ax3.set_xlabel('Number of Relocated Offices')
        ax3.set_ylabel('Maximum Workload')
        ax3.set_title('Workload vs Relocations', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        return fig


def main():
    """Main execution for Step 3"""
    print("\n" + "="*70)
    print("PFIZER OPTIMIZATION - STEP 3 OFFICE RELOCATION")
    print("="*70)
    
    step3 = Step3OfficeRelocation()
    
    # Model 1: Minimize distance
    print("\n" + "="*70)
    print("TESTING MODEL 1: Minimize Distance")
    print("="*70)
    m1, sol1 = step3.model_minimize_distance(verbose=True)
    
    if sol1:
        fig1 = step3.visualize_solution(sol1, "Model 1: Min Distance")
        fig1.savefig('step3_model1_min_distance.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved: step3_model1_min_distance.png")
    
    # Model 2: Minimize max workload
    print("\n" + "="*70)
    print("TESTING MODEL 2: Minimize Maximum Workload")
    print("="*70)
    m2, sol2 = step3.model_minimize_maxworkload(verbose=True)
    
    if sol2:
        fig2 = step3.visualize_solution(sol2, "Model 2: Min Max Workload")
        fig2.savefig('step3_model2_min_workload.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved: step3_model2_min_workload.png")
    
    # Model 3: Bi-objective with different weights
    print("\n" + "="*70)
    print("TESTING MODEL 3: Bi-Objective Optimization")
    print("="*70)
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    biobjective_results = []
    
    for alpha in alphas:
        m, sol = step3.model_biobjective(alpha=alpha, verbose=False)
        if sol:
            biobjective_results.append({
                'alpha': alpha,
                'distance': sol['total_distance'],
                'workload_max': sol['workload_max'],
                'displaced': sol['displaced_offices']
            })
            print(f"  α={alpha:.2f}: Distance={sol['total_distance']:.2f}, "
                  f"Workload={sol['workload_max']:.2f}, Displaced={sol['displaced_offices']}")
    
    # Model 4: Three-objective Pareto frontier
    print("\n" + "="*70)
    print("TESTING MODEL 4: Three-Objective Pareto Frontier")
    print("="*70)
    
    df_pareto = step3.generate_pareto_three_objectives(verbose=False)
    
    # Save results
    df_pareto.to_csv('step3_pareto_three_objectives.csv', index=False)
    print("\n✓ Results saved: step3_pareto_three_objectives.csv")
    
    # Visualize
    fig_pareto = step3.visualize_pareto_3d(df_pareto)
    fig_pareto.savefig('step3_pareto_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: step3_pareto_3d.png")
    
    print("\n" + "="*70)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nSummary of Results:")
    print(f"  - Model 1 (Min Distance): {sol1['total_distance']:.2f}")
    print(f"  - Model 2 (Min Workload): {sol2['workload_max']:.2f}")
    print(f"  - Pareto solutions generated: {len(df_pareto)}")
    print(f"  - Files created: 3 PNG visualizations + 1 CSV")


if __name__ == "__main__":
    main()
