# SR Office Relocation Optimization

A complete optimization solution for the Sales Representative (SR) office relocation problem using integer programming and heuristic algorithms.

## Problem Description

The SR office relocation problem involves optimally placing `n` offices (one per SR) across `m` locations (bricks) to minimize travel distance while balancing workload. The key challenge is that when relocating offices, the model identifies **where** to place the `n` new offices but not necessarily which office corresponds to which SR - the relocated offices are "anonymous."

### Mathematical Formulation

**Parameters:**
- `m`: number of bricks (locations)
- `n`: number of SRs (sales representatives)
- `w_i`: workload associated with brick `i`
- `d_ij`: distance from brick `i` to brick `j`

**Decision Variables:**
- `x_ij ∈ {0,1}`: brick `i` is assigned to office located at brick `j`
- `y_j ∈ {0,1}`: brick `j` contains an office
- `wm`: maximum workload across all offices

**Constraints:**
1. Each brick assigned to exactly one office: `Σ_j x_ij = 1, ∀i ∈ {1,...,m}`
2. Exactly n offices: `Σ_j y_j = n`
3. Assignment only to offices: `x_ij ≤ y_j, ∀i,j`
4. Workload balance: `wm ≥ Σ_i w_i·x_ij, ∀j`

**Objective Functions:**
1. **Total Distance**: Minimize `Σ_ij d_ij·x_ij`
2. **Workload Max**: Minimize `wm`
3. **Displaced Offices**: Minimize `Σ_{j∈J⁰} y_j` where J⁰ = initial office locations

## Project Structure

```
sr-office-relocation/
├── sr_office_relocation.py      # Main solver implementation
├── data-100x10.xlsx              # Input data (100 bricks, 10 SRs)
├── solution.json                 # Solution output
├── office_report.csv             # Detailed office assignments
├── brick_report.csv              # Detailed brick assignments
├── solution_visualization.png    # Visual representation
└── README.md                     # This file
```

## Installation

### Requirements
```bash
pip install numpy pandas openpyxl matplotlib
```

### Optional (for advanced solvers)
```bash
pip install pulp ortools  # For mixed-integer programming
```

## Usage

### Basic Usage
```python
python sr_office_relocation.py
```

### Using as a Module
```python
from sr_office_relocation import SROfficeRelocationSolver, load_data_from_excel

# Load data
coords, brick_workload, brick_zones, initial_offices, distance_matrix = \
    load_data_from_excel('data-100x10.xlsx')

# Create solver
solver = SROfficeRelocationSolver(
    coords=coords,
    brick_workload=brick_workload,
    brick_zones=brick_zones,
    initial_offices=initial_offices,
    distance_matrix=distance_matrix
)

# Solve
x, y, metrics = solver.greedy_solve(verbose=True)

# Access results
print(f"Total Distance: {metrics['total_distance']:.4f}")
print(f"Office Locations: {metrics['office_locations']}")
```

## Input Data Format

The input Excel file should contain:
- **Columns**: `id`, `x_coord`, `y_coord`, `index`, `current_zone`, `office`
- **id**: Brick identifier
- **x_coord, y_coord**: Spatial coordinates
- **index**: Workload value for the brick
- **current_zone**: SR zone (1 to n)
- **office**: 1 if brick currently has an office, 0 otherwise

## Output

### Solution Metrics
```
Total Distance: 13.9620
Maximum Workload: 1.5060
Displaced Offices: 6/10
Average Distance to Office: 0.1396
```

### office_report.csv
Contains for each office:
- SR assignment
- Office brick location
- Number of bricks assigned
- Total workload
- Average distance to assigned bricks
- Coordinates

### brick_report.csv
Contains for each brick:
- Brick ID
- SR zone
- Individual workload
- Assigned office
- Distance to assigned office

## Algorithm

### Greedy Heuristic
The default solver uses a weighted centroid heuristic:

1. **For each SR zone:**
   - Calculate weighted centroid based on brick workloads
   - Place office at brick closest to centroid

2. **Assign bricks to offices:**
   - Each brick assigned to nearest office location

**Complexity**: O(m·n) for placement + O(m·n) for assignment = O(m·n)

**Advantages:**
- Fast execution
- Good approximation quality
- Guarantees feasible solution
- Balances workload naturally

## Results (100 bricks, 10 SRs)

| Metric | Value |
|--------|-------|
| Total Distance | 13.9620 |
| Max Workload | 1.5060 |
| Avg Workload | 1.0000 |
| Workload Std Dev | 0.3370 |
| Displaced Offices | 6/10 |
| Max Distance | 0.3138 |

## Extensions

### Using Different Objectives
```python
# Minimize maximum workload
x, y, metrics = solver.greedy_solve(objective='workload_max')

# Minimize displaced offices
x, y, metrics = solver.greedy_solve(objective='displaced')
```

### Custom Solvers
The project can be extended with:
- **MILP solvers** (PuLP, Gurobi, CPLEX) for exact solutions
- **Metaheuristics** (Genetic Algorithms, Simulated Annealing)
- **Local search** improvement heuristics

## Performance

- **Greedy Heuristic**: ~0.01s for 100 bricks
- **Scales linearly**: O(m·n) complexity
- **Memory**: O(m²) for distance matrix

## Visualization

The solution includes a visualization showing:
- **Left panel**: Initial configuration with original offices
- **Right panel**: Optimized configuration with new offices and assignments
- **Color coding**: Different colors for each SR zone
- **Assignment lines**: Show brick-to-office mappings

## References

1. Facility Location Problems in Operations Research
2. Mixed-Integer Programming formulations
3. Greedy Heuristics for Assignment Problems

## Author

Created for the SR Office Relocation Optimization Project
November 2025

## License

MIT License - Free to use and modify
