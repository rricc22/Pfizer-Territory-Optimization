"""
Step 3: Office Relocation for 22 Bricks / 4 SRs (Original Dataset)
Uses exact optimization models with Gurobi
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class Step3_22x4:
    """Step 3 Office Relocation for 22x4 dataset"""
    
    def __init__(self, data_path: str = 'data/'):
        """Initialize with original 22x4 data"""
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load 22 bricks / 4 SRs data from original format"""
        print(f"\n{'='*70}")
        print("STEP 3: Loading data for office relocation (22 bricks / 4 SRs)...")
        print(f"{'='*70}")
        
        # Load workload
        df_index = pd.read_excel(f'{self.data_path}indexValues.xlsx', header=None)
        self.workload = dict(zip(df_index[0], df_index[1]))
        
        # Load distances
        df_dist = pd.read_excel(f'{self.data_path}distances.xlsx')
        df_dist_clean = df_dist.iloc[1:, 2:6]
        df_dist_clean.columns = [1, 2, 3, 4]
        df_dist_clean.index = range(1, 23)
        
        # Initial offices (assumed from original problem - centers of 4 SR territories)
        # From Step 1, these were bricks: 6, 13, 17, 21
        self.initial_offices = [6, 13, 17, 21]
        
        self.bricks = list(range(1, 23))  # 22 bricks
        self.n_bricks = 22
        self.n_srs = 4
        
        # Calculate distance matrix between ALL bricks
        # We'll use Euclidean distance based on existing distance-to-office data
        self._calculate_distance_matrix(df_dist_clean)
        
        print(f"✓ Data loaded")
        print(f"  - {self.n_bricks} bricks")
        print(f"  - {self.n_srs} SRs")
        print(f"  - Total workload: {sum(self.workload.values()):.4f}")
        print(f"  - Initial offices: {self.initial_offices}")
        
    def _calculate_distance_matrix(self, df_dist):
        """Approximate distance matrix between all brick pairs"""
        # We have distances from each brick to 4 office locations
        # We'll use triangulation to estimate brick-to-brick distances
        
        self.distances = {}
        
        # For same brick
        for i in self.bricks:
            self.distances[(i, i)] = 0.0
        
        # For different bricks, estimate using existing office distances
        # d(brick_i, brick_j) ≈ d(brick_i, office_k) + d(office_k, brick_j) / 2
        for i in self.bricks:
            for j in self.bricks:
                if i != j:
                    # Get distances to all offices
                    d_i = [df_dist.loc[i, office] for office in [1, 2, 3, 4]]
                    d_j = [df_dist.loc[j, office] for office in [1, 2, 3, 4]]
                    
                    # Use minimum triangulated distance
                    estimates = []
                    for k in range(4):
                        estimates.append(abs(d_i[k] - d_j[k]))  # Lower bound
                        estimates.append((d_i[k] + d_j[k]) / 2)  # Average
                    
                    self.distances[(i, j)] = min(estimates)
    
    def model_minimize_distance(self, verbose=True):
        """
        Model 1: Minimize Total Travel Distance with Office Relocation
        
        Decision Variables:
          x[i,j]: brick i assigned to office at brick j
          y[j]: brick j has an office
        
        Objective: min Σ_i,j d[i,j] * x[i,j]
        """
        print(f"\n{'='*70}")
        print("MODEL 1: Minimize Total Distance (Office Relocation)")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_MinDistance_22x4")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = {}  # x[i,j] = 1 if brick i assigned to office at brick j
        for i in self.bricks:
            for j in self.bricks:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        y = {}  # y[j] = 1 if brick j has an office
        for j in self.bricks:
            y[j] = m.addVar(vtype=GRB.BINARY, name=f'y_{j}')
        
        m.update()
        
        # Objective: minimize total distance
        obj = gp.quicksum(self.distances[(i, j)] * x[i, j] 
                          for i in self.bricks for j in self.bricks)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        # 1. Each brick assigned to exactly one office
        for i in self.bricks:
            m.addConstr(gp.quicksum(x[i, j] for j in self.bricks) == 1, 
                       name=f'assign_{i}')
        
        # 2. Exactly n_srs offices
        m.addConstr(gp.quicksum(y[j] for j in self.bricks) == self.n_srs, 
                   name='num_offices')
        
        # 3. Can only assign to brick j if it has an office
        for i in self.bricks:
            for j in self.bricks:
                m.addConstr(x[i, j] <= y[j], name=f'valid_{i}_{j}')
        
        # 4. Workload balance constraint (optional - within 20% of average)
        avg_workload = sum(self.workload.values()) / self.n_srs
        for j in self.bricks:
            office_workload = gp.quicksum(self.workload[i] * x[i, j] 
                                         for i in self.bricks)
            m.addConstr(office_workload <= 1.2 * avg_workload * y[j], 
                       name=f'workload_max_{j}')
            m.addConstr(office_workload >= 0.8 * avg_workload * y[j], 
                       name=f'workload_min_{j}')
        
        # Optimize
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            # Extract solution
            solution = {
                'offices': [j for j in self.bricks if y[j].X > 0.5],
                'assignment': {},
                'total_distance': m.objVal,
                'workloads': {},
                'displaced_offices': sum(1 for j in self.initial_offices if y[j].X < 0.5)
            }
            
            for i in self.bricks:
                for j in self.bricks:
                    if x[i, j].X > 0.5:
                        solution['assignment'][i] = j
            
            # Calculate workloads
            for j in solution['offices']:
                workload = sum(self.workload[i] for i, office in solution['assignment'].items() 
                              if office == j)
                solution['workloads'][j] = workload
            
            solution['workload_max'] = max(solution['workloads'].values())
            solution['workload_min'] = min(solution['workloads'].values())
            
            print(f"\n✓ Optimal solution found!")
            print(f"  Total Distance: {solution['total_distance']:.4f}")
            print(f"  Offices: {solution['offices']}")
            print(f"  Max Workload: {solution['workload_max']:.4f}")
            print(f"  Displaced Offices: {solution['displaced_offices']}/{self.n_srs}")
            
            return m, solution
        else:
            print(f"  ✗ No optimal solution (status: {m.status})")
            return m, None
    
    def model_minimize_maxworkload(self, verbose=True):
        """Model 2: Minimize Maximum Workload"""
        print(f"\n{'='*70}")
        print("MODEL 2: Minimize Maximum Workload (Office Relocation)")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_MinMaxWorkload_22x4")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        # Decision variables
        x = {}
        for i in self.bricks:
            for j in self.bricks:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        y = {}
        for j in self.bricks:
            y[j] = m.addVar(vtype=GRB.BINARY, name=f'y_{j}')
        
        wm = m.addVar(vtype=GRB.CONTINUOUS, name='workload_max')
        
        m.update()
        
        # Objective: minimize maximum workload
        m.setObjective(wm, GRB.MINIMIZE)
        
        # Constraints
        for i in self.bricks:
            m.addConstr(gp.quicksum(x[i, j] for j in self.bricks) == 1, 
                       name=f'assign_{i}')
        
        m.addConstr(gp.quicksum(y[j] for j in self.bricks) == self.n_srs, 
                   name='num_offices')
        
        for i in self.bricks:
            for j in self.bricks:
                m.addConstr(x[i, j] <= y[j], name=f'valid_{i}_{j}')
        
        # Track max workload
        for j in self.bricks:
            office_workload = gp.quicksum(self.workload[i] * x[i, j] 
                                         for i in self.bricks)
            m.addConstr(wm >= office_workload, name=f'track_max_{j}')
        
        # Optimize
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = {
                'offices': [j for j in self.bricks if y[j].X > 0.5],
                'assignment': {},
                'workload_max': wm.X,
                'workloads': {},
                'displaced_offices': sum(1 for j in self.initial_offices if y[j].X < 0.5)
            }
            
            for i in self.bricks:
                for j in self.bricks:
                    if x[i, j].X > 0.5:
                        solution['assignment'][i] = j
            
            for j in solution['offices']:
                workload = sum(self.workload[i] for i, office in solution['assignment'].items() 
                              if office == j)
                solution['workloads'][j] = workload
            
            # Calculate total distance
            total_dist = sum(self.distances[(i, j)] 
                           for i, j in solution['assignment'].items())
            solution['total_distance'] = total_dist
            
            print(f"\n✓ Optimal solution found!")
            print(f"  Maximum Workload: {solution['workload_max']:.4f}")
            print(f"  Offices: {solution['offices']}")
            print(f"  Total Distance: {solution['total_distance']:.4f}")
            print(f"  Displaced Offices: {solution['displaced_offices']}/{self.n_srs}")
            
            return m, solution
        else:
            print(f"  ✗ No optimal solution (status: {m.status})")
            return m, None
    
    def model_biobjective(self, alpha=0.5, verbose=True):
        """Model 3: Bi-objective (distance + workload)"""
        print(f"\n{'='*70}")
        print(f"MODEL 3: Bi-Objective (α={alpha:.2f})")
        print(f"{'='*70}")
        
        m = gp.Model("Step3_BiObjective_22x4")
        if not verbose:
            m.setParam('OutputFlag', 0)
        
        x = {}
        for i in self.bricks:
            for j in self.bricks:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        y = {}
        for j in self.bricks:
            y[j] = m.addVar(vtype=GRB.BINARY, name=f'y_{j}')
        
        wm = m.addVar(vtype=GRB.CONTINUOUS, name='workload_max')
        
        m.update()
        
        # Bi-objective: α * distance + (1-α) * workload
        dist_obj = gp.quicksum(self.distances[(i, j)] * x[i, j] 
                               for i in self.bricks for j in self.bricks)
        
        # Normalize objectives (approximate)
        max_dist = 20.0  # Estimated
        max_workload = 2.0  # Estimated
        
        obj = alpha * (dist_obj / max_dist) + (1 - alpha) * (wm / max_workload)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        for i in self.bricks:
            m.addConstr(gp.quicksum(x[i, j] for j in self.bricks) == 1)
        
        m.addConstr(gp.quicksum(y[j] for j in self.bricks) == self.n_srs)
        
        for i in self.bricks:
            for j in self.bricks:
                m.addConstr(x[i, j] <= y[j])
        
        for j in self.bricks:
            office_workload = gp.quicksum(self.workload[i] * x[i, j] 
                                         for i in self.bricks)
            m.addConstr(wm >= office_workload)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            solution = {
                'alpha': alpha,
                'offices': [j for j in self.bricks if y[j].X > 0.5],
                'assignment': {},
                'workload_max': wm.X,
                'workloads': {},
                'displaced_offices': sum(1 for j in self.initial_offices if y[j].X < 0.5)
            }
            
            for i in self.bricks:
                for j in self.bricks:
                    if x[i, j].X > 0.5:
                        solution['assignment'][i] = j
            
            for j in solution['offices']:
                workload = sum(self.workload[i] for i, office in solution['assignment'].items() 
                              if office == j)
                solution['workloads'][j] = workload
            
            total_dist = sum(self.distances[(i, j)] 
                           for i, j in solution['assignment'].items())
            solution['total_distance'] = total_dist
            
            if verbose:
                print(f"  Distance: {solution['total_distance']:.4f}")
                print(f"  Max Workload: {solution['workload_max']:.4f}")
                print(f"  Offices: {solution['offices']}")
            
            return m, solution
        else:
            return m, None
    
    def generate_pareto_frontier(self, num_points=10):
        """Generate Pareto frontier for distance vs workload"""
        print(f"\n{'='*70}")
        print(f"Generating Pareto Frontier ({num_points} points)...")
        print(f"{'='*70}")
        
        results = []
        alphas = np.linspace(0, 1, num_points)
        
        for alpha in alphas:
            m, sol = self.model_biobjective(alpha=alpha, verbose=False)
            if sol:
                results.append({
                    'alpha': alpha,
                    'distance': sol['total_distance'],
                    'workload_max': sol['workload_max'],
                    'displaced': sol['displaced_offices'],
                    'offices': str(sol['offices'])
                })
                print(f"  α={alpha:.2f}: Distance={sol['total_distance']:.4f}, "
                      f"Workload={sol['workload_max']:.4f}, Displaced={sol['displaced_offices']}")
        
        df = pd.DataFrame(results)
        return df


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("PFIZER STEP 3: OFFICE RELOCATION (22 Bricks / 4 SRs)")
    print("="*70)
    
    step3 = Step3_22x4()
    
    # Test Model 1: Minimize distance
    print("\n" + "="*70)
    print("TESTING MODEL 1: Minimize Distance")
    print("="*70)
    m1, sol1 = step3.model_minimize_distance(verbose=True)
    
    # Test Model 2: Minimize max workload
    print("\n" + "="*70)
    print("TESTING MODEL 2: Minimize Maximum Workload")
    print("="*70)
    m2, sol2 = step3.model_minimize_maxworkload(verbose=True)
    
    # Test Model 3: Bi-objective
    print("\n" + "="*70)
    print("TESTING MODEL 3: Bi-Objective with Various Weights")
    print("="*70)
    
    for alpha in [0.0, 0.5, 1.0]:
        m3, sol3 = step3.model_biobjective(alpha=alpha, verbose=True)
    
    # Generate Pareto frontier
    print("\n" + "="*70)
    print("GENERATING PARETO FRONTIER")
    print("="*70)
    
    df_pareto = step3.generate_pareto_frontier(num_points=15)
    
    # Save results
    df_pareto.to_csv('step3_pareto_22x4.csv', index=False)
    print(f"\n✓ Results saved: step3_pareto_22x4.csv")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pareto frontier
    ax1.plot(df_pareto['distance'], df_pareto['workload_max'], 
             'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Maximum Workload', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Frontier: Distance vs Workload\n(Office Relocation Allowed)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Displaced offices
    scatter = ax2.scatter(df_pareto['distance'], df_pareto['workload_max'], 
                         c=df_pareto['displaced'], cmap='RdYlGn_r', 
                         s=150, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Maximum Workload', fontsize=12, fontweight='bold')
    ax2.set_title('Number of Displaced Offices', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Displaced Offices', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('step3_pareto_22x4.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: step3_pareto_22x4.png")
    
    print("\n" + "="*70)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nKey Results:")
    print(f"  Model 1 (Min Distance): {sol1['total_distance']:.4f}")
    print(f"  Model 2 (Min Workload): {sol2['workload_max']:.4f}")
    print(f"  Pareto solutions: {len(df_pareto)}")
    print(f"\nFiles created:")
    print(f"  - step3_pareto_22x4.csv")
    print(f"  - step3_pareto_22x4.png")


if __name__ == "__main__":
    main()
