"""
Pfizer Territory Optimization - Step 2 Extensions
==================================================

Step 2 extensions to the basic model:
1. Test 100 bricks / 10 SRs instances with Pareto frontier
2. Partial brick assignment (one brick to multiple SRs)
3. Demand increase scenario (+25%) with new SR office placement

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


class Step2Extensions:
    """
    Extensions for Step 2 of the Pfizer optimization project
    """
    
    def __init__(self, data_path: str = 'data/data-100x10.xlsx'):
        """Initialize with 100 bricks / 10 SRs data"""
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load data from Excel file"""
        print(f"\n{'='*70}")
        print("Loading 100 bricks / 10 SRs data...")
        print(f"{'='*70}")
        
        # Load and clean data
        df = pd.read_excel(self.data_path, skiprows=3)
        df.columns = ['id', 'x_coord', 'y_coord', 'index', 'current_zone', 'office']
        df = df[df['id'] != 'x-coord'].copy()
        df = df.dropna(subset=['id']).copy()
        
        # Convert to proper types
        df = df[df['id'].notna()].copy()
        df['id'] = df['id'].astype(int)
        df['x_coord'] = df['x_coord'].astype(float)
        df['y_coord'] = df['y_coord'].astype(float)
        df['index'] = df['index'].astype(float)
        df['current_zone'] = df['current_zone'].astype(float).astype(int)
        df['office'] = df['office'].astype(int)
        
        self.df = df
        self.bricks = list(df['id'])
        self.srs = list(range(1, 11))  # 10 SRs
        self.n_bricks = len(self.bricks)
        self.n_srs = len(self.srs)
        
        # Workload dictionary
        self.workload = dict(zip(df['id'], df['index']))
        
        # Coordinates
        self.coords = df[['x_coord', 'y_coord']].values
        
        # Center bricks (offices)
        self.center_bricks = dict(zip(
            df[df['office'] == 1]['current_zone'].values,
            df[df['office'] == 1]['id'].values
        ))
        
        # Current assignment
        self.current_assignment = {}
        for sr in self.srs:
            self.current_assignment[sr] = df[df['current_zone'] == sr]['id'].tolist()
        
        # Calculate distance matrix
        self._calculate_distances()
        
        print(f"✓ Data loaded successfully")
        print(f"  - {self.n_bricks} bricks")
        print(f"  - {self.n_srs} sales representatives")
        print(f"  - Total workload: {sum(self.workload.values()):.4f}")
        print(f"  - Initial offices: {sorted(self.center_bricks.values())}")
        
    def _calculate_distances(self):
        """Calculate distance matrix between bricks and SR offices"""
        self.distances = {}
        
        for brick_id in self.bricks:
            brick_idx = brick_id - 1
            brick_coord = self.coords[brick_idx]
            
            for sr in self.srs:
                office_brick_id = self.center_bricks[sr]
                office_idx = office_brick_id - 1
                office_coord = self.coords[office_idx]
                
                # Euclidean distance
                dist = np.sqrt(
                    (brick_coord[0] - office_coord[0])**2 + 
                    (brick_coord[1] - office_coord[1])**2
                )
                self.distances[(brick_id, sr)] = dist
    
    def create_current_assignment_matrix(self):
        """Create binary matrix A for current assignment"""
        A = {}
        for i in self.bricks:
            for j in self.srs:
                A[(i, j)] = 1 if i in self.current_assignment[j] else 0
        return A
    
    # ========================================================================
    # STEP 2.1: Test models on 100x10 instance
    # ========================================================================
    
    def model_1_minimize_distance(self, wl_min: float = 0.8, wl_max: float = 1.2,
                                   verbose: bool = True) -> Tuple[gp.Model, dict]:
        """Model 1: Minimize total travel distance"""
        m = gp.Model("Model1_MinDistance_100x10")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
        
        # Constraint 1: Each brick to exactly one SR
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        
        # Constraint 2: Workload balance
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
             for j in self.srs), 
            name="WorkloadMin"
        )
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) <= wl_max 
             for j in self.srs), 
            name="WorkloadMax"
        )
        
        # Objective: Minimize total distance
        obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                         for i in self.bricks for j in self.srs)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        # Extract solution
        solution = self._extract_solution(m, x) if m.status == GRB.OPTIMAL else None
        
        return m, solution
    
    def model_2_minimize_disruption(self, wl_min: float = 0.8, wl_max: float = 1.2,
                                    verbose: bool = True) -> Tuple[gp.Model, dict]:
        """Model 2: Minimize disruption - Optimized for limited license"""
        m = gp.Model("Model2_MinDisruption_100x10")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
        
        # Current assignment
        A = self.create_current_assignment_matrix()
        
        # Constraints
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
             for j in self.srs), 
            name="WorkloadMin"
        )
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) <= wl_max 
             for j in self.srs), 
            name="WorkloadMax"
        )
        
        # Objective: Minimize disruption (only for changed assignments)
        # Instead of auxiliary variables, directly compute |x - A|
        # Since x, A are binary: |x - A| = x + A - 2*x*A = x(1-A) + A(1-x)
        # For binary: this simplifies to counting changes
        obj_terms = []
        for i in self.bricks:
            for j in self.srs:
                if A[(i, j)] == 1:
                    # Was assigned to j, disruption if NOT assigned
                    obj_terms.append(self.workload[i] * (1 - x[i, j]))
                else:
                    # Was NOT assigned to j, disruption if now assigned
                    obj_terms.append(self.workload[i] * x[i, j])
        
        m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        # Extract solution
        solution = self._extract_solution(m, x) if m.status == GRB.OPTIMAL else None
        
        return m, solution
    
    def epsilon_constraint_pareto(self, wl_min: float = 0.8, wl_max: float = 1.2,
                                  num_points: int = 20, verbose: bool = False) -> pd.DataFrame:
        """
        Generate Pareto frontier using epsilon-constraint method
        """
        print(f"\n{'='*70}")
        print(f"EPSILON-CONSTRAINT: Generating Pareto Frontier (100x10)")
        print(f"Workload bounds: [{wl_min}, {wl_max}]")
        print(f"{'='*70}")
        
        A = self.create_current_assignment_matrix()
        
        # Find min and max disruption
        print("\n1. Finding disruption range...")
        m2, sol2 = self.model_2_minimize_disruption(wl_min, wl_max, verbose=False)
        min_disruption = m2.objVal if m2.status == GRB.OPTIMAL else 0
        
        m1, sol1 = self.model_1_minimize_distance(wl_min, wl_max, verbose=False)
        # Calculate disruption for min distance solution
        max_disruption = self._calculate_disruption_from_model(m1, A)
        
        print(f"   Min disruption: {min_disruption:.4f}")
        print(f"   Max disruption: {max_disruption:.4f}")
        
        # Generate epsilon values
        epsilon_values = np.linspace(min_disruption, max_disruption, num_points)
        
        results = []
        print(f"\n2. Generating {num_points} Pareto points...")
        
        for idx, eps in enumerate(epsilon_values):
            if verbose or (idx % 5 == 0):
                print(f"   Point {idx+1}/{num_points}: ε = {eps:.4f}")
            
            # Solve with epsilon constraint
            m = gp.Model(f"Epsilon_{idx}")
            m.setParam('OutputFlag', 0)
            
            x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
            
            # Standard constraints
            m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
            m.addConstrs(
                (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
                 for j in self.srs), 
                name="WorkloadMin"
            )
            m.addConstrs(
                (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) <= wl_max 
                 for j in self.srs), 
                name="WorkloadMax"
            )
            
            # Disruption expression (without auxiliary variables)
            disruption_terms = []
            for i in self.bricks:
                for j in self.srs:
                    if A[(i, j)] == 1:
                        disruption_terms.append(self.workload[i] * (1 - x[i, j]))
                    else:
                        disruption_terms.append(self.workload[i] * x[i, j])
            disruption_expr = gp.quicksum(disruption_terms)
            
            # Epsilon constraint on disruption
            m.addConstr(disruption_expr <= eps, name="EpsilonConstraint")
            
            # Objective: minimize distance (with small tie-breaker)
            distance_obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                                      for i in self.bricks for j in self.srs)
            obj = distance_obj + 0.0001 * disruption_expr
            m.setObjective(obj, GRB.MINIMIZE)
            
            # Solve
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                distance = sum(self.distances[i, j] * x[i, j].X 
                             for i in self.bricks for j in self.srs)
                
                # Calculate disruption
                disruption = 0
                for i in self.bricks:
                    for j in self.srs:
                        if A[(i, j)] == 1:
                            disruption += self.workload[i] * (1 - x[i, j].X)
                        else:
                            disruption += self.workload[i] * x[i, j].X
                
                # Calculate workload metrics
                workloads = []
                for j in self.srs:
                    wl = sum(self.workload[i] * x[i, j].X for i in self.bricks)
                    workloads.append(wl)
                
                results.append({
                    'epsilon': eps,
                    'distance': distance,
                    'disruption': disruption,
                    'workload_max': max(workloads),
                    'workload_min': min(workloads),
                    'workload_std': np.std(workloads),
                    'num_changes': self._count_reassignments(m, x, A)
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.drop_duplicates(subset=['distance', 'disruption'])
        
        print(f"\n✓ Generated {len(df_results)} unique Pareto-optimal solutions")
        
        return df_results
    
    # ========================================================================
    # STEP 2.2: Partial assignment model
    # ========================================================================
    
    def model_partial_assignment(self, wl_min: float = 0.8, wl_max: float = 1.2,
                                max_splits: int = None, verbose: bool = True):
        """
        Model with partial brick assignment
        A brick can be assigned to multiple SRs with fractional assignments
        
        Args:
            max_splits: Maximum number of SRs a brick can be assigned to (None = no limit)
        """
        print(f"\n{'='*70}")
        print("MODEL WITH PARTIAL BRICK ASSIGNMENT")
        print(f"{'='*70}")
        
        m = gp.Model("PartialAssignment")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables: x[i,j] ∈ [0,1] (continuous, representing fraction)
        x = m.addVars(self.bricks, self.srs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        
        # Binary variable: z[i,j] = 1 if brick i assigned to SR j (even partially)
        z = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="z")
        
        # Constraint 1: Each brick fully assigned (sum of fractions = 1)
        m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="FullAssignment")
        
        # Constraint 2: Link x and z (if x > 0 then z = 1)
        for i in self.bricks:
            for j in self.srs:
                m.addConstr(x[i, j] <= z[i, j], name=f"Link_{i}_{j}")
        
        # Constraint 3: Limit splits per brick (optional)
        if max_splits is not None:
            m.addConstrs(
                (z.sum(i, '*') <= max_splits for i in self.bricks),
                name="MaxSplits"
            )
        
        # Constraint 4: Workload balance
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
             for j in self.srs), 
            name="WorkloadMin"
        )
        m.addConstrs(
            (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) <= wl_max 
             for j in self.srs), 
            name="WorkloadMax"
        )
        
        # Objective: Minimize total distance (weighted by assignment fraction)
        obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                         for i in self.bricks for j in self.srs)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = self._extract_partial_solution(m, x, z)
            print(f"\n✓ Optimal solution found")
            print(f"  Total distance: {m.objVal:.4f}")
            print(f"  Number of split bricks: {solution['num_split_bricks']}")
            print(f"  Total assignments: {solution['total_assignments']}")
            
            return m, solution
        else:
            print(f"\n✗ No optimal solution found (status: {m.status})")
            return m, None
    
    # ========================================================================
    # STEP 2.3: Demand increase scenario (+25%)
    # ========================================================================
    
    def model_new_sr_placement(self, demand_increase: float = 0.25, 
                               wl_min: float = 0.8, wl_max: float = 1.2,
                               verbose: bool = True):
        """
        Model for placing a new SR office when demand increases
        
        Uses a two-phase heuristic approach:
        1. Find best location for new office (p-median style)
        2. Optimize assignments given that location
        """
        print(f"\n{'='*70}")
        print(f"NEW SR PLACEMENT (Demand +{demand_increase*100:.0f}%)")
        print(f"{'='*70}")
        
        # New workload after increase
        new_workload = {i: self.workload[i] * (1 + demand_increase) 
                       for i in self.bricks}
        total_workload = sum(new_workload.values())
        n_new_srs = self.n_srs + 1
        
        print(f"  Original SRs: {self.n_srs}")
        print(f"  New SRs: {n_new_srs}")
        print(f"  Original workload: {sum(self.workload.values()):.4f}")
        print(f"  New workload: {total_workload:.4f}")
        
        # Phase 1: Find best office location using greedy heuristic
        # Try each brick as potential new office and solve assignment problem
        print(f"\n  Phase 1: Finding best location for new office...")
        
        best_location = None
        best_distance = float('inf')
        best_solution = None
        
        # Sample a subset of candidate locations for efficiency
        candidate_bricks = self.bricks[::5]  # Sample every 5th brick
        
        for candidate_idx, candidate_brick in enumerate(candidate_bricks):
            if verbose and candidate_idx % 5 == 0:
                print(f"    Testing location {candidate_idx+1}/{len(candidate_bricks)}...")
            
            # Create temporary model with this candidate as new office
            m_temp = gp.Model(f"TempModel_{candidate_brick}")
            m_temp.setParam('OutputFlag', 0)
            
            new_srs = list(range(1, n_new_srs + 1))
            x = m_temp.addVars(self.bricks, new_srs, vtype=GRB.BINARY, name="x")
            
            # Constraints
            m_temp.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
            m_temp.addConstrs(
                (gp.quicksum(new_workload[i] * x[i, j] for i in self.bricks) >= wl_min 
                 for j in new_srs), 
                name="WorkloadMin"
            )
            m_temp.addConstrs(
                (gp.quicksum(new_workload[i] * x[i, j] for i in self.bricks) <= wl_max 
                 for j in new_srs), 
                name="WorkloadMax"
            )
            
            # Calculate distances with this candidate as new office
            obj_terms = []
            
            # Existing SRs
            for j in range(1, self.n_srs + 1):
                office_brick = self.center_bricks[j]
                office_idx = office_brick - 1
                office_coord = self.coords[office_idx]
                
                for i in self.bricks:
                    brick_idx = i - 1
                    brick_coord = self.coords[brick_idx]
                    dist = np.sqrt(
                        (brick_coord[0] - office_coord[0])**2 + 
                        (brick_coord[1] - office_coord[1])**2
                    )
                    obj_terms.append(dist * x[i, j])
            
            # New SR with fixed office at candidate_brick
            new_office_idx = candidate_brick - 1
            new_office_coord = self.coords[new_office_idx]
            
            for i in self.bricks:
                brick_idx = i - 1
                brick_coord = self.coords[brick_idx]
                dist = np.sqrt(
                    (brick_coord[0] - new_office_coord[0])**2 + 
                    (brick_coord[1] - new_office_coord[1])**2
                )
                obj_terms.append(dist * x[i, n_new_srs])
            
            m_temp.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
            m_temp.optimize()
            
            if m_temp.status == GRB.OPTIMAL and m_temp.objVal < best_distance:
                best_distance = m_temp.objVal
                best_location = candidate_brick
                
                # Extract solution
                assignments = {j: [] for j in new_srs}
                workloads = {j: 0 for j in new_srs}
                
                for i in self.bricks:
                    for j in new_srs:
                        if x[i, j].X > 0.5:
                            assignments[j].append(i)
                            workloads[j] += new_workload[i]
                
                best_solution = {
                    'total_distance': m_temp.objVal,
                    'new_office_brick': candidate_brick,
                    'new_office_coord': self.coords[candidate_brick - 1],
                    'assignments': assignments,
                    'workloads': workloads,
                    'n_srs': n_new_srs
                }
        
        if best_solution:
            print(f"\n✓ Optimal solution found")
            print(f"  Total distance: {best_solution['total_distance']:.4f}")
            print(f"  New office at brick: {best_solution['new_office_brick']}")
            print(f"  New office coordinates: ({best_solution['new_office_coord'][0]:.4f}, "
                  f"{best_solution['new_office_coord'][1]:.4f})")
            print(f"\n  Workload distribution:")
            for j in range(1, n_new_srs + 1):
                wl = best_solution['workloads'][j]
                n_bricks = len(best_solution['assignments'][j])
                print(f"    SR {j}: {wl:.4f} ({n_bricks} bricks)")
            
            return None, best_solution  # Return None for model since we used temp models
        else:
            print(f"\n✗ No feasible solution found")
            return None, None
    
    # ========================================================================
    # Helper methods
    # ========================================================================
    
    def _extract_solution(self, model, x):
        """Extract solution from model"""
        if model.status != GRB.OPTIMAL:
            return None
        
        assignments = {j: [] for j in self.srs}
        workloads = {j: 0 for j in self.srs}
        
        for i in self.bricks:
            for j in self.srs:
                if x[i, j].X > 0.5:
                    assignments[j].append(i)
                    workloads[j] += self.workload[i]
        
        total_distance = sum(self.distances[i, j] * x[i, j].X 
                            for i in self.bricks for j in self.srs)
        
        return {
            'assignments': assignments,
            'workloads': workloads,
            'total_distance': total_distance,
            'objective': model.objVal
        }
    
    def _extract_partial_solution(self, model, x, z):
        """Extract solution from partial assignment model"""
        if model.status != GRB.OPTIMAL:
            return None
        
        assignments = {j: [] for j in self.srs}
        workloads = {j: 0 for j in self.srs}
        split_bricks = []
        
        for i in self.bricks:
            brick_srs = []
            for j in self.srs:
                if x[i, j].X > 0.01:  # Non-zero assignment
                    assignments[j].append((i, x[i, j].X))
                    workloads[j] += self.workload[i] * x[i, j].X
                    brick_srs.append((j, x[i, j].X))
            
            if len(brick_srs) > 1:
                split_bricks.append((i, brick_srs))
        
        return {
            'assignments': assignments,
            'workloads': workloads,
            'split_bricks': split_bricks,
            'num_split_bricks': len(split_bricks),
            'total_assignments': sum(len(assignments[j]) for j in self.srs),
            'total_distance': model.objVal
        }
    
    def _calculate_disruption_from_model(self, model, A):
        """Calculate disruption value from a solved model"""
        if model.status != GRB.OPTIMAL:
            return None
        
        x_vars = [v for v in model.getVars() if v.varName.startswith('x[')]
        disruption = 0
        
        for var in x_vars:
            # Parse variable name to get i, j
            name = var.varName
            i, j = map(int, name[2:-1].split(','))
            disruption += self.workload[i] * abs(var.X - A[(i, j)])
        
        return disruption
    
    def _count_reassignments(self, model, x, A):
        """Count number of reassigned bricks"""
        if model.status != GRB.OPTIMAL:
            return 0
        
        count = 0
        for i in self.bricks:
            current_sr = None
            for j in self.srs:
                if A[(i, j)] == 1:
                    current_sr = j
                    break
            
            new_sr = None
            for j in self.srs:
                if x[i, j].X > 0.5:
                    new_sr = j
                    break
            
            if current_sr != new_sr:
                count += 1
        
        return count
    
    def visualize_pareto_frontier(self, df_results, title_suffix=""):
        """Visualize Pareto frontier"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Distance vs Disruption
        ax = axes[0]
        ax.scatter(df_results['disruption'], df_results['distance'], 
                  c=df_results['workload_std'], cmap='viridis', s=100, alpha=0.7)
        ax.plot(df_results['disruption'], df_results['distance'], 
               'r--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Disruption (weighted)', fontsize=12)
        ax.set_ylabel('Total Distance', fontsize=12)
        ax.set_title(f'Pareto Frontier: Distance vs Disruption {title_suffix}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Workload Std Dev', fontsize=10)
        
        # Plot 2: Trade-off metrics
        ax = axes[1]
        ax2 = ax.twinx()
        
        ax.plot(range(len(df_results)), df_results['distance'], 
               'b-o', label='Distance', alpha=0.7)
        ax2.plot(range(len(df_results)), df_results['disruption'], 
                'r-s', label='Disruption', alpha=0.7)
        
        ax.set_xlabel('Solution Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12, color='b')
        ax2.set_ylabel('Disruption', fontsize=12, color='r')
        ax.set_title(f'Trade-off Analysis {title_suffix}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig


def main():
    """Main execution for Step 2"""
    print("\n" + "="*70)
    print("PFIZER OPTIMIZATION - STEP 2 EXTENSIONS")
    print("="*70)
    
    step2 = Step2Extensions()
    
    # Step 2.1: Test on 100x10 instance
    print("\n" + "="*70)
    print("STEP 2.1: Testing models on 100 bricks / 10 SRs")
    print("="*70)
    
    # Test Model 1
    m1, sol1 = step2.model_1_minimize_distance(verbose=True)
    
    # Test Model 2
    m2, sol2 = step2.model_2_minimize_disruption(verbose=True)
    
    # Generate Pareto frontier
    df_pareto = step2.epsilon_constraint_pareto(num_points=15)
    print("\nPareto frontier results:")
    print(df_pareto.to_string())
    
    # Save results
    df_pareto.to_csv('step2_pareto_100x10.csv', index=False)
    print("\n✓ Pareto results saved to step2_pareto_100x10.csv")
    
    # Visualize
    fig = step2.visualize_pareto_frontier(df_pareto, "(100x10)")
    fig.savefig('step2_pareto_100x10.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to step2_pareto_100x10.png")
    
    # Step 2.2: Partial assignment
    print("\n" + "="*70)
    print("STEP 2.2: Partial Brick Assignment")
    print("="*70)
    
    m_partial, sol_partial = step2.model_partial_assignment(max_splits=2, verbose=True)
    
    if sol_partial:
        print("\nSplit bricks details:")
        for brick_id, assignments in sol_partial['split_bricks'][:10]:
            print(f"  Brick {brick_id}:")
            for sr, fraction in assignments:
                print(f"    SR {sr}: {fraction:.2%}")
    
    # Step 2.3: New SR placement
    print("\n" + "="*70)
    print("STEP 2.3: Demand Increase Scenario")
    print("="*70)
    
    m_new, sol_new = step2.model_new_sr_placement(demand_increase=0.25, verbose=True)
    
    print("\n" + "="*70)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
