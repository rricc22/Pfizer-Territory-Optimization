# Pfizer Territory Optimization in Decision Modeling

**Multi-Objective Optimization for Sales Representative Territory Assignment**

A complete optimization framework using Gurobi to assign geographic territories to sales representatives, balancing travel efficiency and assignment stability.

---

## Project Overview

### Problem Statement
Assign 22 geographic territories ("bricks") to 4 Sales Representatives (SRs) to optimize:
1. **Efficiency**: Minimize total travel distance
2. **Stability**: Minimize disruption to current assignments

### Key Features
- Two mono-objective optimization models (distance, disruption)
- Multi-objective Pareto frontier generation
- Three workload balance scenarios
- Comprehensive visualization suite
- Scalability testing framework

---

## Team

**Group 3 - Decision Modelling Project**  
MSc Artificial Intelligence, CentraleSupélec 2025-26


---

## Repository Structure

```
Pfizer_Global/
├── README.md                           # This file
├── MODEL_EXPLANATION.md                # Detailed mathematical explanations
├── COLLABORATION_GUIDE.md              # Git workflow for team
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── pfizer_optimization.py              # Core optimization framework
├── pareto_analysis.py                  # Pareto frontier generation
├── scalability_test.py                 # Performance testing
│
├── data/                               # Input data
│   ├── indexValues.xlsx                # Workload values per brick
│   └── distances.xlsx                  # Distance matrix
│
├── pfizer_complete_analysis.ipynb      # Jupyter notebook analysis
├── gurobi_lab1.ipynb                   # Gurobi introduction
├── expresse_pb_lab1.ipynb              # Express optimization examples
│
└── Projet_Pfitzer_MScAI_2025_26.pdf    # Project specification
```

**Note**: Generated files (`.png`, `.pkl`, `.csv`) are ignored by Git to keep repo clean.

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (Academic license recommended)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Pfizer-Territory-Optimization.git
cd Pfizer-Territory-Optimization
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Gurobi

**Academic License** (Free for students):
1. Register at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Download license file
3. Install:
   ```bash
   grbgetkey YOUR-LICENSE-KEY
   ```

**Verify installation**:
```bash
python -c "import gurobipy; print(gurobipy.__version__)"
```

---

## Usage

### Quick Start: Run Complete Analysis

```bash
# Generate all Pareto frontiers with visualizations
python pareto_analysis.py
```

**Output**:
- 3 Pareto frontiers (different workload scenarios)
- PNG visualizations (frontiers, heatmaps, workload comparisons)
- CSV data files
- Pickle file with complete results

### Test Individual Models

```bash
# Test Model 1 (minimize distance) and Model 2 (minimize disruption)
python pfizer_optimization.py
```

### Run Scalability Tests

```bash
# Test on larger problem instances
python scalability_test.py
```

### Interactive Analysis (Jupyter)

```bash
jupyter notebook pfizer_complete_analysis.ipynb
```

---

## Models Explained

### Model 1: Minimize Distance

**Objective**: Minimize total travel distance
```
Minimize: Σ d_{i,j} × x_{i,j}
```

**Constraints**:
- Each brick to exactly one SR
- Workload balance: `[wl_min, wl_max]` per SR

**Use case**: Optimize operational efficiency

### Model 2: Minimize Disruption

**Objective**: Minimize changes from current assignment
```
Minimize: Σ w_i × |x_{i,j} - A_{i,j}|
```

**Constraints**:
- Same as Model 1
- Uses auxiliary variables for absolute value linearization

**Use case**: Maintain stability, minimize change impact

### Multi-Objective: Epsilon-Constraint Method

Generates Pareto frontier by:
1. Optimizing distance (primary objective)
2. Constraining disruption ≤ ε
3. Varying ε to find trade-off solutions

**Result**: 15-25 Pareto-optimal solutions showing efficiency vs. stability trade-offs

For detailed mathematical explanations with visual matrices, see [`MODEL_EXPLANATION.md`](MODEL_EXPLANATION.md).

---

## Key Results

### Workload Scenarios Tested

| Scenario | Workload Bounds | Flexibility | Pareto Solutions |
|----------|-----------------|-------------|------------------|
| 1        | [0.8, 1.2]      | 40%         | ~25              |
| 2        | [0.85, 1.15]    | 30%         | ~20              |
| 3        | [0.9, 1.1]      | 20%         | ~15              |

### Performance

- **Problem size**: 22 bricks × 4 SRs = 88 binary variables
- **Solve time**: < 2 seconds per model
- **Pareto generation**: ~40 seconds for 20 points
- **Scalability**: Tested up to 100 bricks × 10 SRs

### Sample Results (Scenario 1: [0.8, 1.2])

```
Current Assignment:
  Total Distance: 187.41 km
  Disruption: 0.0 (baseline)

Model 1 (Min Distance):
  Total Distance: 157.23 km (-16.1%)
  Disruption: 1.52

Model 2 (Min Disruption):
  Total Distance: 178.45 km (-4.8%)
  Disruption: 0.23
```

---

## Collaboration Guide

### Git Workflow (Quick Reference)

See [`COLLABORATION_GUIDE.md`](COLLABORATION_GUIDE.md) for complete workflow.

**Daily workflow**:
```bash
# 1. Start working
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 2. Make changes, test locally

# 3. Commit and push
git add <files>
git commit -m "Description of changes"
git push origin feature/your-feature-name

# 4. Create Pull Request on GitHub
# 5. After PR approved, merge and delete branch
```

**Rules**:
- Never push directly to `main`
- Always work in feature branches
- Test code before committing
- Write clear commit messages
- Review teammates' PRs

---

## File Descriptions

### Core Python Files

| File | Description | Lines |
|------|-------------|-------|
| `pfizer_optimization.py` | Main optimization class with Models 1, 2, and epsilon-constraint | 460 |
| `pareto_analysis.py` | Complete Pareto analysis with visualizations | 347 |
| `scalability_test.py` | Performance testing on larger instances | 155 |

### Data Files

| File | Description |
|------|-------------|
| `data/indexValues.xlsx` | Workload (index values) for each brick |
| `data/distances.xlsx` | Distance matrix: brick → SR office |

### Notebooks

| File | Purpose |
|------|---------|
| `pfizer_complete_analysis.ipynb` | Interactive analysis and experimentation |
| `gurobi_lab1.ipynb` | Gurobi basics and examples |
| `expresse_pb_lab1.ipynb` | Express optimization problems |

---

## Dependencies

Main libraries:
- `gurobipy` - Optimization solver
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

See `requirements.txt` for complete list with versions.

---

## Troubleshooting

### Gurobi License Error
```
GurobiError: No Gurobi license found
```
**Solution**: Install academic license (see Setup Step 4)

### Import Error
```
ModuleNotFoundError: No module named 'gurobipy'
```
**Solution**: 
```bash
pip install gurobipy
```

### Data Files Not Found
```
FileNotFoundError: data/indexValues.xlsx
```
**Solution**: Ensure you're running from project root directory

### Merge Conflicts
**Solution**: See `COLLABORATION_GUIDE.md` Section 5

---

## Development

### Running Tests

```bash
# Test optimization models
python pfizer_optimization.py

# Test scalability
python scalability_test.py

# Test Pareto generation (takes ~1 min)
python pareto_analysis.py
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement and test
3. Update documentation if needed
4. Push and create Pull Request
5. Wait for team review

### Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Comment complex logic

---

## Project Timeline

- **Week 1-2**: Problem formulation, Model 1 & 2 implementation
- **Week 3**: Multi-objective optimization (epsilon-constraint)
- **Week 4**: Visualization, analysis, documentation
- **Week 5**: Final report and presentation

---

## References

### Course Materials
- Project Specification: `Projet_Pfitzer_MScAI_2025_26.pdf`
- Course: Decision Modelling, CentraleSupélec MSc AI 2025-26

### Gurobi Documentation
- [Gurobi Python API](https://www.gurobi.com/documentation/current/refman/py_python_api_overview.html)
- [Modeling with Gurobi](https://www.gurobi.com/documentation/current/examples/index.html)

### Optimization Theory
- Multi-objective optimization: Epsilon-constraint method
- Integer programming: Binary variables, linearization techniques

---

## License

Academic project for CentraleSupélec MSc AI 2025-26.  
Not licensed for commercial use.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact team members
- Reach out to course instructors

---

## Acknowledgments

- **CentraleSupélec** - Decision Modelling course
- **Gurobi** - Academic license program
- **Pfizer** - Problem inspiration

---

**Last Updated**: November 2025
