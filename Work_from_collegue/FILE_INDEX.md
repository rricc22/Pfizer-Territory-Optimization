# SR Office Relocation Optimization - File Index

## Project Overview
Complete working solution for SR office relocation optimization using integer programming and heuristic algorithms.

## Files Included

### 🐍 Python Source Code
1. **sr_office_relocation.py** (10,142 bytes)
   - Main solver implementation with greedy heuristic
   - Complete mathematical formulation
   - Data loading and preprocessing
   - Solution export and reporting
   - Entry point: `python sr_office_relocation.py`

2. **mip_solver.py** (5,629 bytes)
   - Advanced Mixed Integer Programming solver
   - Requires: `pip install pulp`
   - Finds exact optimal solutions
   - For smaller instances or critical decisions

3. **quickstart.py** (2,300 bytes)
   - Quick demonstration script
   - Best way to test the project
   - Run: `python quickstart.py`

### 📄 Documentation
4. **README.md** (5,646 bytes)
   - Complete usage documentation
   - API reference
   - Examples and tutorials
   - Performance metrics

5. **PROJECT_SUMMARY.md** (9,029 bytes)
   - Executive summary
   - Solution results and analysis
   - Algorithm explanation
   - Extension ideas

6. **requirements.txt** (297 bytes)
   - Python dependencies list
   - Install: `pip install -r requirements.txt`

### 📊 Input Data
7. **data-100x10.xlsx** (15,158 bytes)
   - Test instance: 100 bricks, 10 SRs
   - Contains coordinates, workloads, zone assignments
   - Initial office locations marked

### 📈 Solution Outputs
8. **solution.json** (320 bytes)
   - Solution metrics in JSON format
   - Total distance: 13.9620
   - Office locations: [12, 16, 38, 32, 46, 60, 72, 65, 78, 96]

9. **office_report.csv** (1,217 bytes)
   - Detailed office assignment report
   - Workload per office
   - Average distances

10. **brick_report.csv** (8,499 bytes)
    - Complete brick assignment details
    - Distance to assigned office
    - SR zone mapping

11. **office_assignments.csv** (1,217 bytes)
    - Alternative office format
    - Same data as office_report.csv

12. **brick_assignments.csv** (8,499 bytes)
    - Alternative brick format
    - Same data as brick_report.csv

13. **solution_visualization.png** (228,954 bytes)
    - Dual-panel visualization
    - Before/after comparison
    - Color-coded by SR zone

## Quick Start Guide

### Installation
```bash
pip install -r requirements.txt
```

### Run Demonstration
```bash
python quickstart.py
```

### Run Full Solver
```bash
python sr_office_relocation.py
```

### Use as Library
```python
from sr_office_relocation import SROfficeRelocationSolver, load_data_from_excel

coords, brick_workload, brick_zones, initial_offices, distance_matrix = \
    load_data_from_excel('data-100x10.xlsx')

solver = SROfficeRelocationSolver(
    coords=coords,
    brick_workload=brick_workload,
    brick_zones=brick_zones,
    initial_offices=initial_offices,
    distance_matrix=distance_matrix
)

x, y, metrics = solver.greedy_solve(verbose=True)
print(f"Total Distance: {metrics['total_distance']:.4f}")
```

## Solution Summary

### Metrics
- Total Distance: **13.9620**
- Maximum Workload: **1.5060**
- Average Workload: **1.0000**
- Workload Std Dev: **0.3370**
- Displaced Offices: **6/10**
- Max Distance: **0.3138**

### Office Assignments
| SR | Office Brick | Assigned | Workload | Avg Dist |
|----|-------------|----------|----------|----------|
| 1  | 12          | 11       | 1.0937   | 0.1518   |
| 2  | 16          | 10       | 1.0818   | 0.1488   |
| 3  | 38          | 10       | 0.9995   | 0.1304   |
| 4  | 32          | 14       | 1.4874   | 0.1500   |
| 5  | 46          | 8        | 0.8143   | 0.0967   |
| 6  | 60          | 7        | 0.3763   | 0.1429   |
| 7  | 72          | 11       | 1.0500   | 0.1435   |
| 8  | 65          | 9        | 0.7531   | 0.1380   |
| 9  | 78          | 11       | 1.5060   | 0.1455   |
| 10 | 96          | 9        | 0.8380   | 0.1338   |

## Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- openpyxl >= 3.0.9
- matplotlib >= 3.4.0
- pulp >= 2.6.0 (optional, for MIP solver)

## File Size Summary
- Total Project Size: ~290 KB
- Source Code: ~18 KB
- Documentation: ~15 KB
- Data & Results: ~257 KB

## Key Features
✓ Fast greedy heuristic (< 0.01s)
✓ Optional exact MIP solver
✓ Comprehensive visualization
✓ Detailed CSV reports
✓ JSON solution export
✓ Well-documented code
✓ Production-ready

## Support
- Review README.md for detailed documentation
- Check PROJECT_SUMMARY.md for results analysis
- Run quickstart.py to see example usage
- Examine source code for implementation details

## License
MIT License - Free to use and modify

## Version
1.0 - November 25, 2025
