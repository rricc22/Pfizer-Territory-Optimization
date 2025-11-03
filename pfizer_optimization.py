"""
Pfizer Territory Optimization - Complete Solution
Multi-objective optimization for Sales Representative territory assignment

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


class PfizerOptimization:
    """
    Complete optimization framework for Pfizer SR territory assignment
    Supports mono-objective and multi-objective optimization
    """
    
    def __init__(self, data_path: str = 'data/'):
        """Initialize with data loading from Excel files"""
        self.data_path = data_path
        self.workload = None
        self.distances = None
        self.current_assignment = None
        self.bricks = None
        self.srs = None
        self.center_bricks = None
        
    def load_data(self):
        """Load workload and distance data from Excel files"""
        # Load index values (workload)
        df_index = pd.read_excel(f'{self.data_path}indexValues.xlsx', header=None)
        self.workload = dict(zip(df_index[0], df_index[1]))
        
        # Load distances
        df_dist = pd.read_excel(f'{self.data_path}distances.xlsx')
        df_dist_clean = df_dist.iloc[1:, 2:6]
        df_dist_clean.columns = [1, 2, 3, 4]
        df_dist_clean.index = range(1, 23)
        
        # Create distance dictionary
        self.distances = {}
        for brick in range(1, 23):
            for sr in range(1, 5):
                self.distances[(brick, sr)] = float(df_dist_clean.loc[brick, sr])
        
        # Define sets
        self.bricks = list(range(1, 23))
        self.srs = list(range(1, 5))
        
        # Center bricks (SR offices) - from Table 1 in PDF
        self.center_bricks = {1: 4, 2: 14, 3: 16, 4: 22}
        
        # Current assignment from Table 1
        self.current_assignment = {
            1: [4, 5, 6, 7, 8, 15],
            2: [10, 11, 12, 13, 14],
            3: [9, 16, 17, 18],
            4: [1, 2, 3, 19, 20, 21, 22]
        }
        
        print(f"✓ Data loaded successfully")
        print(f"  - {len(self.workload)} bricks")
        print(f"  - {len(self.srs)} sales representatives")
        print(f"  - Total workload: {sum(self.workload.values()):.4f}")
        
    def create_current_assignment_matrix(self):
        """Create binary matrix A for current assignment"""
        A = {}
        for i in self.bricks:
            for j in self.srs:
                A[(i, j)] = 1 if i in self.current_assignment[j] else 0
        return A
    
    def model_1_minimize_distance(self, wl_min: float = 0.8, wl_max: float = 1.2, 
                                   verbose: bool = True) -> Tuple[gp.Model, dict]:
        """
        Model 1: Minimize total travel distance
        Subject to workload balance constraints
        
        Returns: (model, solution_dict)
        """
        m = gp.Model("Model1_MinDistance")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
        
        # Constraint 1: Each brick assigned to exactly one SR
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
        """
        Model 2: Minimize disruption (weighted by index values)
        Subject to workload balance constraints
        
        Disruption = sum of index values of reassigned bricks
        """
        m = gp.Model("Model2_MinDisruption")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
        
        # Auxiliary variables for disruption (absolute value)
        # y[i,j] = |x[i,j] - A[i,j]|
        y = m.addVars(self.bricks, self.srs, vtype=GRB.CONTINUOUS, name="y")
        
        # Current assignment matrix
        A = self.create_current_assignment_matrix()
        
        # Constraint 1: Each brick assigned to exactly one SR
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
        
        # Constraint 3: Define absolute value for disruption
        for i in self.bricks:
            for j in self.srs:
                m.addConstr(y[i, j] >= x[i, j] - A[(i, j)], name=f"abs1_{i}_{j}")
                m.addConstr(y[i, j] >= A[(i, j)] - x[i, j], name=f"abs2_{i}_{j}")
        
        # Objective: Minimize disruption (weighted by index values)
        obj = gp.quicksum(self.workload[i] * y[i, j] 
                         for i in self.bricks for j in self.srs)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        m.optimize()
        
        # Extract solution
        solution = self._extract_solution(m, x) if m.status == GRB.OPTIMAL else None
        
        return m, solution
    
    def epsilon_constraint_method(self, wl_min: float = 0.8, wl_max: float = 1.2,
                                  num_points: int = 20, verbose: bool = False) -> pd.DataFrame:
        """
        Generate Pareto frontier using epsilon-constraint method
        
        Primary objective: Minimize distance
        Constraint: Disruption ≤ epsilon (varying)
        
        Returns DataFrame with Pareto-optimal solutions
        """
        print(f"\n{'='*70}")
        print(f"EPSILON-CONSTRAINT METHOD: Generating Pareto Frontier")
        print(f"Workload bounds: [{wl_min}, {wl_max}]")
        print(f"{'='*70}")
        
        A = self.create_current_assignment_matrix()
        
        # First, find range of disruption values
        # Min disruption (solve Model 2)
        print("\n1. Finding minimum disruption...")
        m_min_disr, _ = self.model_2_minimize_disruption(wl_min, wl_max, verbose=False)
        min_disruption = m_min_disr.ObjVal
        
        # Max disruption (solve Model 1, then calculate disruption)
        print("2. Finding maximum disruption...")
        m_max_disr, sol_max = self.model_1_minimize_distance(wl_min, wl_max, verbose=False)
        max_disruption = self._calculate_disruption(sol_max, A)
        
        print(f"\n   Disruption range: [{min_disruption:.4f}, {max_disruption:.4f}]")
        
        # Generate epsilon values
        epsilon_values = np.linspace(min_disruption, max_disruption, num_points)
        
        pareto_solutions = []
        
        print(f"\n3. Solving {num_points} epsilon-constraint problems...")
        
        for idx, eps in enumerate(epsilon_values):
            # Build epsilon-constraint model
            m = gp.Model(f"EpsilonConstraint_{idx}")
            m.setParam('OutputFlag', 0)
            
            # Decision variables
            x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
            y = m.addVars(self.bricks, self.srs, vtype=GRB.CONTINUOUS, name="y")
            
            # Assignment constraints
            m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
            
            # Workload constraints
            m.addConstrs(
                (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
                 for j in self.srs), name="WorkloadMin")
            m.addConstrs(
                (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) <= wl_max 
                 for j in self.srs), name="WorkloadMax")
            
            # Absolute value constraints
            for i in self.bricks:
                for j in self.srs:
                    m.addConstr(y[i, j] >= x[i, j] - A[(i, j)])
                    m.addConstr(y[i, j] >= A[(i, j)] - x[i, j])
            
            # Epsilon constraint on disruption
            disruption = gp.quicksum(self.workload[i] * y[i, j] 
                                    for i in self.bricks for j in self.srs)
            m.addConstr(disruption <= eps, name="EpsilonConstraint")
            
            # Objective: Minimize distance + small epsilon to ensure efficiency
            distance = gp.quicksum(self.distances[i, j] * x[i, j] 
                                  for i in self.bricks for j in self.srs)
            m.setObjective(distance + 0.0001 * disruption, GRB.MINIMIZE)
            
            # Solve
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                sol = self._extract_solution(m, x)
                dist_val = sum(self.distances[i, j] for j in self.srs 
                              for i in sol['assignment'][j])
                disr_val = self._calculate_disruption(sol, A)
                
                pareto_solutions.append({
                    'distance': dist_val,
                    'disruption': disr_val,
                    'epsilon': eps,
                    'assignment': sol['assignment'].copy(),
                    'workloads': sol['workloads'].copy()
                })
                
                if (idx + 1) % 5 == 0:
                    print(f"   Progress: {idx+1}/{num_points} solutions found")
        
        # Remove dominated solutions (sometimes duplicates occur)
        pareto_df = pd.DataFrame(pareto_solutions)
        pareto_df = self._filter_dominated(pareto_df)
        
        print(f"\n✓ Generated {len(pareto_df)} non-dominated solutions")
        print(f"{'='*70}\n")
        
        return pareto_df
    
    def _extract_solution(self, model: gp.Model, x_vars: dict) -> dict:
        """Extract solution from solved model"""
        assignment = {j: [] for j in self.srs}
        
        for i in self.bricks:
            for j in self.srs:
                if x_vars[i, j].X > 0.5:
                    assignment[j].append(i)
        
        # Calculate metrics
        workloads = {}
        distances_per_sr = {}
        
        for j in self.srs:
            workloads[j] = sum(self.workload[i] for i in assignment[j])
            distances_per_sr[j] = sum(self.distances[i, j] for i in assignment[j])
        
        return {
            'assignment': assignment,
            'workloads': workloads,
            'distances': distances_per_sr,
            'total_distance': sum(distances_per_sr.values())
        }
    
    def _calculate_disruption(self, solution: dict, current_A: dict) -> float:
        """Calculate disruption metric for a solution"""
        disruption = 0.0
        for j in self.srs:
            for i in solution['assignment'][j]:
                if current_A[(i, j)] == 0:  # Brick i newly assigned to SR j
                    disruption += self.workload[i]
        return disruption
    
    def _filter_dominated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove dominated solutions from dataframe"""
        # Sort and remove duplicates
        df = df.sort_values(['distance', 'disruption']).reset_index(drop=True)
        
        # Keep only non-dominated
        pareto = []
        for idx, row in df.iterrows():
            dominated = False
            for _, other in df.iterrows():
                if (other['distance'] <= row['distance'] and 
                    other['disruption'] <= row['disruption'] and
                    (other['distance'] < row['distance'] or other['disruption'] < row['disruption'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(row)
        
        return pd.DataFrame(pareto).reset_index(drop=True)
    
    def analyze_current_assignment(self) -> dict:
        """Analyze metrics of current assignment"""
        A = self.create_current_assignment_matrix()
        
        current_sol = {
            'assignment': self.current_assignment,
            'workloads': {},
            'distances': {}
        }
        
        for j in self.srs:
            current_sol['workloads'][j] = sum(self.workload[i] for i in self.current_assignment[j])
            current_sol['distances'][j] = sum(self.distances[i, j] for i in self.current_assignment[j])
        
        current_sol['total_distance'] = sum(current_sol['distances'].values())
        
        return current_sol
    
    def print_solution_comparison(self, solution: dict, title: str = "Solution"):
        """Pretty print solution with comparison to current"""
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        current = self.analyze_current_assignment()
        
        print(f"\n{'SR':<4} {'Bricks':<30} {'Workload':<12} {'Distance (km)':<15}")
        print(f"{'-'*70}")
        
        for j in self.srs:
            bricks_str = str(sorted(solution['assignment'][j]))[:28]
            wl = solution['workloads'][j]
            wl_status = "✓" if 0.8 <= wl <= 1.2 else "✗"
            dist = solution['distances'][j]
            print(f"{j:<4} {bricks_str:<30} {wl:>6.4f} {wl_status:<4} {dist:>8.2f}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<35} {sum(solution['workloads'].values()):>6.4f}      {solution['total_distance']:>8.2f}")
        
        # Comparison with current
        if 'total_distance' in current:
            savings = current['total_distance'] - solution['total_distance']
            pct = (savings / current['total_distance']) * 100 if current['total_distance'] > 0 else 0
            print(f"\n{'Current Total:':<35} {current['total_distance']:>15.2f} km")
            print(f"{'Improvement:':<35} {savings:>15.2f} km ({pct:.1f}%)")
        
        print(f"{'='*70}\n")


def generate_random_instance(num_bricks: int, num_srs: int, seed: int = 42) -> Dict:
    """
    Generate random instance for scalability testing
    
    Args:
        num_bricks: Number of bricks
        num_srs: Number of SRs
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with workload, distances, current_assignment
    """
    np.random.seed(seed)
    
    # Generate workloads that sum to num_srs (target 1.0 per SR)
    workload = {}
    raw_workloads = np.random.dirichlet(np.ones(num_bricks)) * num_srs
    for i in range(1, num_bricks + 1):
        workload[i] = raw_workloads[i-1]
    
    # Generate random brick coordinates
    brick_coords = {}
    for i in range(1, num_bricks + 1):
        brick_coords[i] = (np.random.uniform(0, 100), np.random.uniform(0, 100))
    
    # Generate SR office locations (random bricks)
    center_bricks = {}
    office_bricks = np.random.choice(range(1, num_bricks + 1), num_srs, replace=False)
    for j, brick in enumerate(office_bricks, 1):
        center_bricks[j] = brick
    
    # Calculate Euclidean distances
    distances = {}
    for i in range(1, num_bricks + 1):
        for j in range(1, num_srs + 1):
            office_brick = center_bricks[j]
            dist = np.sqrt((brick_coords[i][0] - brick_coords[office_brick][0])**2 +
                          (brick_coords[i][1] - brick_coords[office_brick][1])**2)
            distances[(i, j)] = dist
    
    # Generate random current assignment (balanced)
    current_assignment = {j: [] for j in range(1, num_srs + 1)}
    bricks_list = list(range(1, num_bricks + 1))
    np.random.shuffle(bricks_list)
    
    for i, brick in enumerate(bricks_list):
        sr = (i % num_srs) + 1
        current_assignment[sr].append(brick)
    
    return {
        'workload': workload,
        'distances': distances,
        'current_assignment': current_assignment,
        'center_bricks': center_bricks,
        'bricks': list(range(1, num_bricks + 1)),
        'srs': list(range(1, num_srs + 1))
    }


if __name__ == "__main__":
    # Quick test
    opt = PfizerOptimization()
    opt.load_data()
    
    print("\n" + "="*70)
    print("TESTING MODEL 1: MINIMIZE DISTANCE")
    print("="*70)
    m1, sol1 = opt.model_1_minimize_distance(verbose=False)
    opt.print_solution_comparison(sol1, "Model 1: Minimum Distance Solution")
    
    print("\n" + "="*70)
    print("TESTING MODEL 2: MINIMIZE DISRUPTION")
    print("="*70)
    m2, sol2 = opt.model_2_minimize_disruption(verbose=False)
    opt.print_solution_comparison(sol2, "Model 2: Minimum Disruption Solution")
