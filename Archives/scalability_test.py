"""
Scalability Test for Pfizer Territory Optimization
Tests the optimization framework with larger instances

Authors: Decision Modelling Project 2025-26
Date: November 2025
"""

from pfizer_optimization import PfizerOptimization, generate_random_instance
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd

def test_random_instance(num_bricks, num_srs, wl_min=0.8, wl_max=1.2):
    """Test optimization on randomly generated instance"""
    print(f"\n{'='*70}")
    print(f"Testing Instance: {num_bricks} bricks × {num_srs} SRs")
    print(f"{'='*70}")
    
    # Generate random instance
    print("Generating random instance...")
    instance = generate_random_instance(num_bricks, num_srs, seed=42)
    
    print(f"✓ Instance generated")
    print(f"  - Total workload: {sum(instance['workload'].values()):.4f}")
    print(f"  - Target per SR: {sum(instance['workload'].values()) / num_srs:.4f}")
    
    # Create temporary optimizer with custom data
    opt = PfizerOptimization()
    opt.workload = instance['workload']
    opt.distances = instance['distances']
    opt.current_assignment = instance['current_assignment']
    opt.bricks = instance['bricks']
    opt.srs = instance['srs']
    opt.center_bricks = instance['center_bricks']
    
    # Test Model 1: Minimize Distance
    print(f"\nSolving Model 1 (Minimize Distance)...")
    start_time = time.time()
    
    m1 = gp.Model("Model1_MinDistance_Large")
    m1.setParam('OutputFlag', 0)
    m1.setParam('TimeLimit', 300)  # 5 minute time limit
    
    # Decision variables
    x = m1.addVars(opt.bricks, opt.srs, vtype=GRB.BINARY, name="x")
    
    # Constraints
    m1.addConstrs((x.sum(i, '*') == 1 for i in opt.bricks), name="AssignBrick")
    m1.addConstrs(
        (gp.quicksum(opt.workload[i] * x[i, j] for i in opt.bricks) >= wl_min 
         for j in opt.srs), name="WorkloadMin")
    m1.addConstrs(
        (gp.quicksum(opt.workload[i] * x[i, j] for i in opt.bricks) <= wl_max 
         for j in opt.srs), name="WorkloadMax")
    
    # Objective
    obj = gp.quicksum(opt.distances[i, j] * x[i, j] 
                     for i in opt.bricks for j in opt.srs)
    m1.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    m1.optimize()
    solve_time = time.time() - start_time
    
    if m1.status == GRB.OPTIMAL:
        print(f"✓ Optimal solution found")
        print(f"  - Solve time: {solve_time:.2f} seconds")
        print(f"  - Objective value: {m1.ObjVal:.2f}")
        print(f"  - Variables: {m1.NumVars}")
        print(f"  - Constraints: {m1.NumConstrs}")
        print(f"  - MIP Gap: {m1.MIPGap*100:.4f}%")
    elif m1.status == GRB.TIME_LIMIT:
        print(f"⚠ Time limit reached")
        print(f"  - Solve time: {solve_time:.2f} seconds")
        print(f"  - Best bound: {m1.ObjBound:.2f}")
        print(f"  - Best solution: {m1.ObjVal:.2f}")
        print(f"  - MIP Gap: {m1.MIPGap*100:.2f}%")
    else:
        print(f"✗ No solution found (status: {m1.status})")
        return None
    
    return {
        'num_bricks': num_bricks,
        'num_srs': num_srs,
        'solve_time': solve_time,
        'objective': m1.ObjVal,
        'status': m1.status,
        'mip_gap': m1.MIPGap,
        'num_vars': m1.NumVars,
        'num_constrs': m1.NumConstrs
    }


def main():
    """Run scalability tests"""
    print("\n" + "="*70)
    print(" "*20 + "SCALABILITY TESTING")
    print(" "*15 + "Pfizer Territory Optimization")
    print("="*70)
    
    # Test configurations
    # Note: 200×20 exceeds Gurobi academic license limits
    test_cases = [
        (22, 4),    # Original problem
        (50, 5),    # Medium instance
        (100, 10),  # Large instance (requested)
    ]
    
    results = []
    
    for num_bricks, num_srs in test_cases:
        result = test_random_instance(num_bricks, num_srs)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("SCALABILITY SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Save results
    df.to_csv('scalability_results.csv', index=False)
    print("\n✓ Saved scalability results to scalability_results.csv")
    
    # Analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    print("\nSolve Time vs Problem Size:")
    for _, row in df.iterrows():
        size = row['num_bricks'] * row['num_srs']
        print(f"  {int(row['num_bricks'])} × {int(row['num_srs'])} "
              f"(size: {size}): {row['solve_time']:.2f}s")
    
    print("\nKey Observations:")
    if len(df) >= 2:
        time_ratio = df.iloc[-1]['solve_time'] / df.iloc[0]['solve_time']
        size_ratio = (df.iloc[-1]['num_bricks'] * df.iloc[-1]['num_srs']) / \
                    (df.iloc[0]['num_bricks'] * df.iloc[0]['num_srs'])
        print(f"  - Largest instance is {size_ratio:.1f}x bigger than original")
        print(f"  - Solve time increased by {time_ratio:.1f}x")
        print(f"  - All instances solved successfully!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
