#!/usr/bin/env python3
"""
Quick Start Script for SR Office Relocation Optimization
Run this to see the solver in action!
"""

import numpy as np
import pandas as pd
from sr_office_relocation import SROfficeRelocationSolver, load_data_from_excel

def main():
    print("="*80)
    print("SR OFFICE RELOCATION - QUICK START")
    print("="*80)

    # Load data
    print("\n1. Loading data from data-100x10.xlsx...")
    coords, brick_workload, brick_zones, initial_offices, distance_matrix = \
        load_data_from_excel('data-100x10.xlsx')

    print(f"   ✓ Loaded {len(coords)} bricks across {int(brick_zones.max())+1} SR zones")
    print(f"   ✓ Initial offices: {len(initial_offices)} locations")

    # Create solver
    print("\n2. Creating solver instance...")
    solver = SROfficeRelocationSolver(
        coords=coords,
        brick_workload=brick_workload,
        brick_zones=brick_zones,
        initial_offices=initial_offices,
        distance_matrix=distance_matrix
    )
    print("   ✓ Solver initialized")

    # Solve
    print("\n3. Running greedy optimization...")
    x, y, metrics = solver.greedy_solve(verbose=False)

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"✓ Total Distance:      {metrics['total_distance']:.4f}")
    print(f"✓ Maximum Workload:    {metrics['workload_max']:.4f}")
    print(f"✓ Displaced Offices:   {metrics['displaced_offices']}/{len(initial_offices)}")
    print(f"✓ Office Locations:    {metrics['office_locations']}")

    # Save solution
    print("\n4. Saving results...")
    solver.save_solution(x, y, metrics, 'solution.json')
    print("   ✓ Solution saved to solution.json")

    # Generate summary
    print("\n" + "="*80)
    print("OFFICE SUMMARY")
    print("="*80)
    for idx, office_brick in enumerate(metrics['office_locations']):
        sr = brick_zones[office_brick] + 1
        assigned = np.sum(x[:, office_brick])
        total_work = np.sum(brick_workload[x[:, office_brick] == 1])
        print(f"SR {sr:2d} → Brick {office_brick:3d} | "
              f"Assigned: {assigned:2d} bricks | Workload: {total_work:.4f}")

    print("\n✓ Quick start complete! Check the output files for detailed results.")
    print("="*80)

if __name__ == "__main__":
    main()
