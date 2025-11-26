# Agent Guidelines - Pfizer Territory Optimization

## Build/Run/Test Commands
```bash
# Run main optimization scripts
python STEP1_script/pfizer_optimization.py          # Test Models 1 & 2
python STEP1_script/pareto_analysis.py              # Generate Pareto frontiers (~1 min)
python STEP3_script/step3_office_relocation.py      # Office relocation model
python scalability_test.py                          # Performance testing

# Run single test file
python STEP3_script/step3_22x4_test.py              # Test specific scenario
```

## Code Style & Conventions
- **Imports**: Group stdlib, third-party (gurobipy, pandas, numpy, matplotlib, seaborn), then local. Use `from typing import Dict, List, Tuple, Optional`
- **Formatting**: PEP 8 compliant. Use 4-space indentation. Max line length ~100 chars (flexible for clarity)
- **Types**: Use type hints for function signatures: `def solve(wl_min: float = 0.8) -> Tuple[gp.Model, dict]:`
- **Naming**: Classes in PascalCase, functions/variables in snake_case. Model variables: `x`, `y`, constraints descriptive
- **Docstrings**: Required for classes and public methods. Use triple quotes with description, Args, Returns sections
- **Error Handling**: Suppress warnings with `warnings.filterwarnings('ignore')`. Gurobi: set `m.setParam('OutputFlag', 0)` for quiet mode
- **Comments**: Use section headers with `# ========` for major blocks. Explain mathematical formulations inline
- **Output**: Print progress with checkmarks (✓) and clear formatting. Use `print(f"{'='*70}")` for headers

## Project Context
- Academic project (CentraleSupélec MSc AI 2025-26) using Gurobi for territory optimization
- Data in `data/` directory (Excel files). Generated outputs (PNG, CSV, PKL) are gitignored
- Never push directly to `main` - use feature branches and PRs (see COLLABORATION_GUIDE.md)
