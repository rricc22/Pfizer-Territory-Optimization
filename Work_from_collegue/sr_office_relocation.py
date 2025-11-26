#!/usr/bin/env python3
"""
SR Office Relocation Optimization Problem
==========================================

This project solves the SR (Sales Representative) office relocation problem
using integer programming. The goal is to relocate n offices to minimize
total travel distance while balancing workload.

Problem Formulation:
- m: number of bricks (locations)
- n: number of SRs (sales representatives)
- w_i: workload associated with brick i
- d_ij: distance from brick i to brick j

Decision Variables:
- x_ij ∈ {0,1}: brick i is assigned to office at brick j
- y_j ∈ {0,1}: brick j contains an office
- wm: maximum workload across all offices

Constraints:
1. Each brick assigned to exactly one office: Σ_j x_ij = 1 ∀i
2. Exactly n offices: Σ_j y_j = n
3. Can only assign to brick with office: x_ij ≤ y_j ∀i,j
4. Workload constraint: wm ≥ Σ_i w_i*x_ij ∀j

Objectives:
1. Minimize total distance: Σ_ij d_ij*x_ij
2. Minimize max workload: wm
3. Minimize displaced offices from initial configuration

Author: AI Assistant
Date: November 2025
"""

import numpy as np
import pandas as pd
import sys
from typing import Tuple, Dict, List
import json


class SROfficeRelocationSolver:
    """
    Solver for SR Office Relocation Problem

    Implements greedy heuristic and optimization-based approaches
    """

    def __init__(self, coords: np.ndarray, brick_workload: np.ndarray, 
                 brick_zones: np.ndarray, initial_offices: np.ndarray, 
                 distance_matrix: np.ndarray):
        """
        Initialize solver with problem data

        Args:
            coords: (m, 2) array of brick coordinates
            brick_workload: (m,) array of workload for each brick
            brick_zones: (m,) array indicating which SR each brick belongs to
            initial_offices: array of indices of bricks with initial offices
            distance_matrix: (m, m) array of distances between bricks
        """
        self.coords = coords
        self.brick_workload = brick_workload
        self.brick_zones = brick_zones
        self.initial_offices = initial_offices
        self.distance_matrix = distance_matrix

        self.m = len(coords)  # number of bricks
        self.n = int(brick_zones.max()) + 1  # number of SRs

        # Calculate SR workloads
        self.sr_workload = np.zeros(self.n)
        for i in range(self.m):
            self.sr_workload[self.brick_zones[i]] += self.brick_workload[i]

    def calculate_objective_total_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate total distance objective: sum of d_ij * x_ij"""
        total_distance = 0.0
        for i in range(self.m):
            for j in range(self.m):
                if y[j] == 1:  # if brick j has an office
                    total_distance += self.distance_matrix[i, j] * x[i, j]
        return total_distance

    def calculate_workload_max(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate maximum workload across all offices"""
        office_workloads = []
        for j in range(self.m):
            if y[j] == 1:  # if brick j has an office
                workload = sum(self.brick_workload[i] * x[i, j] for i in range(self.m))
                office_workloads.append(workload)
        return max(office_workloads) if office_workloads else 0.0

    def calculate_displaced_offices(self, y: np.ndarray) -> int:
        """Calculate number of displaced offices from initial positions"""
        return sum(1 for j in self.initial_offices if y[j] == 0)

    def greedy_solve(self, objective: str = 'total_distance', 
                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Greedy heuristic: place offices at weighted centroids of each SR zone

        Args:
            objective: 'total_distance', 'workload_max', or 'displaced'
            verbose: whether to print progress

        Returns:
            x: (m, m) assignment matrix
            y: (m,) office location vector
            metrics: dict of objective values
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Greedy Heuristic Solver - Objective: {objective}")
            print(f"{'='*80}")

        y = np.zeros(self.m, dtype=int)
        office_locations = []

        for sr in range(self.n):
            # Find all bricks belonging to this SR
            sr_bricks = np.where(self.brick_zones == sr)[0]

            # Calculate weighted centroid based on workload
            weights = self.brick_workload[sr_bricks]
            centroid = np.average(self.coords[sr_bricks], axis=0, weights=weights)

            # Find brick closest to centroid
            distances_to_centroid = [
                np.linalg.norm(self.coords[brick] - centroid)
                for brick in sr_bricks
            ]
            closest_brick = sr_bricks[np.argmin(distances_to_centroid)]

            y[closest_brick] = 1
            office_locations.append(closest_brick)

            if verbose:
                print(f"SR {sr+1}: Office at brick {closest_brick} "
                      f"(workload: {self.sr_workload[sr]:.4f})")

        # Assign each brick to nearest office
        x = np.zeros((self.m, self.m), dtype=int)
        for i in range(self.m):
            distances = [self.distance_matrix[i, j] for j in office_locations]
            nearest_office = office_locations[np.argmin(distances)]
            x[i, nearest_office] = 1

        # Calculate metrics
        metrics = {
            'total_distance': self.calculate_objective_total_distance(x, y),
            'workload_max': self.calculate_workload_max(x, y),
            'displaced_offices': self.calculate_displaced_offices(y),
            'office_locations': office_locations
        }

        if verbose:
            print(f"\n{'='*80}")
            print(f"Solution Metrics:")
            print(f"{'='*80}")
            print(f"Total Distance: {metrics['total_distance']:.4f}")
            print(f"Maximum Workload: {metrics['workload_max']:.4f}")
            print(f"Displaced: {metrics['displaced_offices']}/{len(self.initial_offices)}")

        return x, y, metrics

    def save_solution(self, x: np.ndarray, y: np.ndarray, metrics: Dict, 
                     filename: str = 'solution.json'):
        """Save solution to JSON file"""
        solution = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'office_locations': [int(loc) for loc in metrics['office_locations']],
            'problem_size': {'bricks': self.m, 'SRs': self.n}
        }

        with open(filename, 'w') as f:
            json.dump(solution, f, indent=2)
        print(f"Solution saved to {filename}")


def load_data_from_excel(filename: str) -> Tuple:
    """Load problem data from Excel file"""
    df = pd.read_excel(filename, header=None, skiprows=4)
    df.columns = ['id', 'x_coord', 'y_coord', 'index', 'current_zone', 'office']
    data = df.dropna()

    coords = data[['x_coord', 'y_coord']].values
    brick_workload = data['index'].values
    brick_zones = data['current_zone'].values.astype(int) - 1
    initial_offices = np.where(data['office'].values == 1)[0]

    # Calculate distance matrix
    m = len(coords)
    distance_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    return coords, brick_workload, brick_zones, initial_offices, distance_matrix


def main():
    """Main execution function"""
    print("SR Office Relocation Optimization")
    print("=" * 80)

    # Load data
    print("Loading data from data-100x10.xlsx...")
    coords, brick_workload, brick_zones, initial_offices, distance_matrix = \
        load_data_from_excel('data-100x10.xlsx')

    print(f"Problem size: {len(coords)} bricks, {int(brick_zones.max()) + 1} SRs")
    print(f"Initial offices: {len(initial_offices)} locations")

    # Create solver
    solver = SROfficeRelocationSolver(
        coords=coords,
        brick_workload=brick_workload,
        brick_zones=brick_zones,
        initial_offices=initial_offices,
        distance_matrix=distance_matrix
    )

    # Solve using greedy heuristic
    x, y, metrics = solver.greedy_solve(verbose=True)

    # Save solution
    solver.save_solution(x, y, metrics, 'solution.json')

    # Generate reports
    print("\nGenerating reports...")
    generate_reports(solver, x, y, metrics)

    print("\nOptimization complete! Check output files for results.")


def generate_reports(solver, x, y, metrics):
    """Generate detailed CSV reports"""
    office_locations = metrics['office_locations']

    # Office details
    office_details = []
    for office_brick in office_locations:
        sr = solver.brick_zones[office_brick]
        assigned_bricks = np.where(x[:, office_brick] == 1)[0]
        office_workload = sum(solver.brick_workload[i] for i in assigned_bricks)
        avg_distance = np.mean([solver.distance_matrix[i, office_brick] 
                               for i in assigned_bricks])

        office_details.append({
            'SR': sr + 1,
            'Office_Brick': office_brick,
            'Num_Assigned': len(assigned_bricks),
            'Workload': office_workload,
            'Avg_Distance': avg_distance,
            'X': solver.coords[office_brick, 0],
            'Y': solver.coords[office_brick, 1]
        })

    pd.DataFrame(office_details).to_csv('office_report.csv', index=False)

    # Brick assignments
    brick_details = []
    for i in range(solver.m):
        assigned_office = office_locations[np.argmax([x[i, j] for j in office_locations])]
        brick_details.append({
            'Brick': i,
            'SR': solver.brick_zones[i] + 1,
            'Workload': solver.brick_workload[i],
            'Office': assigned_office,
            'Distance': solver.distance_matrix[i, assigned_office]
        })

    pd.DataFrame(brick_details).to_csv('brick_report.csv', index=False)
    print("Reports saved: office_report.csv, brick_report.csv")


if __name__ == "__main__":
    main()
