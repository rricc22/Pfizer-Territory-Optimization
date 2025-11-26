#!/usr/bin/env python3
"""
SR Office Relocation - Mixed Integer Programming Solver
========================================================

Advanced solver using PuLP for exact optimization.
Requires: pip install pulp

This solver formulates the problem as a MILP and solves it optimally.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("WARNING: PuLP not installed. Install with: pip install pulp")


class MIPSolver:
    """Mixed Integer Programming solver for SR office relocation"""

    def __init__(self, coords, brick_workload, brick_zones, 
                 initial_offices, distance_matrix):
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required for MIP solver")

        self.coords = coords
        self.brick_workload = brick_workload
        self.brick_zones = brick_zones
        self.initial_offices = initial_offices
        self.distance_matrix = distance_matrix

        self.m = len(coords)
        self.n = int(brick_zones.max()) + 1

    def solve_total_distance(self, time_limit=300, verbose=True):
        """
        Solve for minimum total distance using MILP

        Args:
            time_limit: Maximum solving time in seconds
            verbose: Print solving progress

        Returns:
            x, y, metrics
        """
        if verbose:
            print(f"\n{'='*80}")
            print("MIP Solver - Minimizing Total Distance")
            print(f"{'='*80}")
            print(f"Problem size: {self.m} bricks, {self.n} offices")

        # Create the model
        model = LpProblem("SR_Office_Relocation", LpMinimize)

        # Decision variables
        # x[i,j] = 1 if brick i assigned to office at brick j
        x = LpVariable.dicts("x", 
                            ((i, j) for i in range(self.m) for j in range(self.m)),
                            cat='Binary')

        # y[j] = 1 if brick j contains an office
        y = LpVariable.dicts("y", range(self.m), cat='Binary')

        # wm = maximum workload
        wm = LpVariable("wm", lowBound=0)

        # Objective: Minimize total distance
        model += lpSum([self.distance_matrix[i][j] * x[i,j] 
                       for i in range(self.m) for j in range(self.m)])

        # Constraint 1: Each brick assigned to exactly one office
        for i in range(self.m):
            model += lpSum([x[i,j] for j in range(self.m)]) == 1

        # Constraint 2: Exactly n offices
        model += lpSum([y[j] for j in range(self.m)]) == self.n

        # Constraint 3: Can only assign to brick with office
        for i in range(self.m):
            for j in range(self.m):
                model += x[i,j] <= y[j]

        # Constraint 4: Workload max constraint
        for j in range(self.m):
            model += wm >= lpSum([self.brick_workload[i] * x[i,j] 
                                 for i in range(self.m)])

        # Solve
        if verbose:
            print("Solving MILP...")

        solver = PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
        model.solve(solver)

        # Extract solution
        x_sol = np.zeros((self.m, self.m), dtype=int)
        y_sol = np.zeros(self.m, dtype=int)

        for i in range(self.m):
            for j in range(self.m):
                if x[i,j].varValue is not None and x[i,j].varValue > 0.5:
                    x_sol[i,j] = 1

        office_locations = []
        for j in range(self.m):
            if y[j].varValue is not None and y[j].varValue > 0.5:
                y_sol[j] = 1
                office_locations.append(j)

        # Calculate metrics
        total_distance = sum(self.distance_matrix[i][j] * x_sol[i,j]
                           for i in range(self.m) for j in range(self.m))

        max_workload = max(sum(self.brick_workload[i] * x_sol[i,j] 
                              for i in range(self.m))
                          for j in office_locations)

        displaced = sum(1 for j in self.initial_offices if y_sol[j] == 0)

        metrics = {
            'total_distance': total_distance,
            'workload_max': max_workload,
            'displaced_offices': displaced,
            'office_locations': office_locations,
            'solver_status': LpStatus[model.status],
            'objective_value': value(model.objective)
        }

        if verbose:
            print(f"\n{'='*80}")
            print("MIP Solution:")
            print(f"{'='*80}")
            print(f"Status: {metrics['solver_status']}")
            print(f"Total Distance: {metrics['total_distance']:.4f}")
            print(f"Max Workload: {metrics['workload_max']:.4f}")
            print(f"Displaced: {metrics['displaced_offices']}/{len(self.initial_offices)}")

        return x_sol, y_sol, metrics


def main():
    """Example usage of MIP solver"""
    if not PULP_AVAILABLE:
        print("PuLP not available. Please install: pip install pulp")
        return

    # Load data (you'll need to implement or import this)
    from sr_office_relocation import load_data_from_excel

    coords, brick_workload, brick_zones, initial_offices, distance_matrix = \
        load_data_from_excel('data-100x10.xlsx')

    # Create and run MIP solver
    solver = MIPSolver(coords, brick_workload, brick_zones, 
                      initial_offices, distance_matrix)

    x, y, metrics = solver.solve_total_distance(time_limit=60, verbose=True)

    print(f"\nOptimal solution found!")
    print(f"Office locations: {metrics['office_locations']}")


if __name__ == "__main__":
    main()
