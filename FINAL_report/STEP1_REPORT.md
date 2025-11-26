<style>
body { font-size: 10pt; }      /* Regular text */
h1 { font-size: 18pt; }        /* Main titles */
h2 { font-size: 14pt; }        /* Section titles */
h3 { font-size: 12pt; }        /* Subsections */
code { font-size: 9pt; }       /* Code blocks */
</style>

# STEP 1 REPORT: Sales Representative Territory Assignment
## Fixed Center Bricks - Mono & Multi-Objective Optimization

**Group 3 - Decision Modelling Project**  
**MSc Artificial Intelligence, CentraleSupélec 2025-26**

---

## 1. INTRODUCTION

### 1.1 Problem Context
Pfizer Turkey faces the challenge of assigning 22 geographic territories ("bricks") to 4 Sales Representatives (SRs) while:
- **Minimizing travel distance**: Total distance traveled by all SRs
- **Minimizing disruption**: Changes to existing territory assignments (weighted by brick importance)
- **Balancing workload**: Each SR should handle approximately 1.0 unit of workload

In this first step, we consider SR office locations (center bricks) as **fixed** at: SR1→Brick 4, SR2→Brick 14, SR3→Brick 16, SR4→Brick 22.

### 1.2 Mathematical Notation

**Sets:**
- `I = {1, 2, ..., 22}`: Set of bricks
- `J = {1, 2, 3, 4}`: Set of SRs

**Parameters:**
- `w_i`: Workload (index value) of brick `i`
- `d_ij`: Distance from brick `i` to SR `j`'s office
- `A_ij`: Current assignment (1 if brick `i` assigned to SR `j`, 0 otherwise)
- `[wl_min, wl_max]`: Acceptable workload range per SR

**Decision Variables:**
- `x_ij ∈ {0,1}`: Binary variable (1 if brick `i` assigned to SR `j`)

**Current Assignment:** SR1: {4,5,6,7,8,15}, SR2: {10,11,12,13,14}, SR3: {9,16,17,18}, SR4: {1,2,3,19,20,21,22}


<div style="page-break-after: always;"></div>

## 2. MONO-OBJECTIVE MODELS

### 2.1 Model 1: Minimize Total Distance

**Objective:** Optimize travel efficiency by minimizing total distance traveled.

```python
# Model 1: Minimize Distance
m = gp.Model("Model1_MinDistance")
x = m.addVars(bricks, srs, vtype=GRB.BINARY, name="x")

# Constraint 1: Each brick to exactly one SR
m.addConstrs((x.sum(i, '*') == 1 for i in bricks), name="AssignBrick")

# Constraint 2: Workload balance
m.addConstrs((gp.quicksum(workload[i] * x[i, j] for i in bricks) >= wl_min 
              for j in srs), name="WorkloadMin")
m.addConstrs((gp.quicksum(workload[i] * x[i, j] for i in bricks) <= wl_max 
              for j in srs), name="WorkloadMax")

# Objective: Minimize total distance
obj = gp.quicksum(distances[i, j] * x[i, j] for i in bricks for j in srs)
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
```

**Mathematical Formulation:**
```
Minimize:    Σ_i Σ_j d_ij · x_ij

Subject to:  Σ_j x_ij = 1,  ∀i ∈ I                    (assignment)
             wl_min ≤ Σ_i w_i · x_ij ≤ wl_max,  ∀j ∈ J  (workload)
             x_ij ∈ {0,1},  ∀i,j                      (binary)
```
<div style="page-break-after: always;"></div>

### 2.2 Model 2: Minimize Disruption

**Objective:** Preserve existing relationships by minimizing reassignment disruption, weighted by brick importance (index values).

```python
# Model 2: Minimize Disruption
m = gp.Model("Model2_MinDisruption")
x = m.addVars(bricks, srs, vtype=GRB.BINARY, name="x")
y = m.addVars(bricks, srs, vtype=GRB.CONTINUOUS, name="y")  # |x_ij - A_ij|

A = create_current_assignment_matrix()  # Current assignment

# Constraints 1-2: Same as Model 1
m.addConstrs((x.sum(i, '*') == 1 for i in bricks), name="AssignBrick")
m.addConstrs((gp.quicksum(workload[i] * x[i, j] for i in bricks) >= wl_min 
              for j in srs), name="WorkloadMin")
m.addConstrs((gp.quicksum(workload[i] * x[i, j] for i in bricks) <= wl_max 
              for j in srs), name="WorkloadMax")

# Constraint 3: Linearize absolute value |x_ij - A_ij|
for i in bricks:
    for j in srs:
        m.addConstr(y[i, j] >= x[i, j] - A[(i, j)])
        m.addConstr(y[i, j] >= A[(i, j)] - x[i, j])

# Objective: Minimize weighted disruption
obj = gp.quicksum(workload[i] * y[i, j] for i in bricks for j in srs)
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
```

**Mathematical Formulation:**
```
Minimize:    Σ_i Σ_j w_i · |x_ij - A_ij|

Subject to:  Same assignment & workload constraints as Model 1
             y_ij ≥ x_ij - A_ij,  ∀i,j     (absolute value linearization)
             y_ij ≥ A_ij - x_ij,  ∀i,j     (absolute value linearization)
             y_ij ≥ 0,  ∀i,j
```
<div style="page-break-after: always;"></div>

### 2.3 Results for 22 Bricks × 4 SRs (Workload [0.8, 1.2])

| Metric | Current Assignment | Model 1 (Min Distance) | Model 2 (Min Disruption) |
|--------|-------------------|------------------------|--------------------------|
| **Total Distance** | 187.41 km | **154.60 km** (-17.5%) | 188.95 km (-0.8%) |
| **Disruption** | 0.0000 (baseline) | 0.5864 | **0.1696** |
| **Workload Range** | [0.556, 1.168] | [0.803, 1.115] | [0.874, 1.168] |
| **Solve Time** | - | < 0.5s | < 0.5s |

**Key Observations:**
- **Model 1** achieves 17.5% distance reduction but causes higher disruption (0.5864)
- **Model 2** preserves most relationships (disruption 0.1696) at cost of minimal distance improvement
- Both models balance workload within [0.8, 1.2] constraints
- Clear trade-off between efficiency and stability

<div style="page-break-after: always;"></div>

## 3. MULTI-OBJECTIVE OPTIMIZATION: EPSILON-CONSTRAINT METHOD

### 3.1 Methodology

To explore the trade-off between distance and disruption, we implement the **epsilon-constraint method**:
```python
# Epsilon-Constraint Method
def epsilon_constraint_method(wl_min, wl_max, num_points=20):
    # 1. Find disruption range: [min_disruption, max_disruption]
    m_min, _ = model_2_minimize_disruption(wl_min, wl_max)
    min_disruption = m_min.ObjVal
    
    m_max, sol = model_1_minimize_distance(wl_min, wl_max)
    max_disruption = calculate_disruption(sol)
    
    # 2. Generate epsilon values
    epsilon_values = np.linspace(min_disruption, max_disruption, num_points)
    
    # 3. Solve for each epsilon
    for eps in epsilon_values:
        m = gp.Model("EpsilonConstraint")
        x = m.addVars(bricks, srs, vtype=GRB.BINARY, name="x")
        y = m.addVars(bricks, srs, vtype=GRB.CONTINUOUS, name="y")
        
        # Standard constraints (assignment + workload)
        m.addConstrs((x.sum(i, '*') == 1 for i in bricks))
        m.addConstrs((gp.quicksum(workload[i] * x[i,j] for i in bricks) >= wl_min 
                      for j in srs))
        m.addConstrs((gp.quicksum(workload[i] * x[i,j] for i in bricks) <= wl_max 
                      for j in srs))
        # Absolute value constraints
        for i in bricks:
            for j in srs:
                m.addConstr(y[i, j] >= x[i, j] - A[(i, j)])
                m.addConstr(y[i, j] >= A[(i, j)] - x[i, j])
        # EPSILON CONSTRAINT: Disruption ≤ ε
        disruption = gp.quicksum(workload[i] * y[i, j] for i in bricks for j in srs)
        m.addConstr(disruption <= eps, name="EpsilonConstraint")
        # PRIMARY OBJECTIVE: Minimize distance
        # Add small coefficient (0.0001) to break ties → ensures Pareto efficiency
        distance = gp.quicksum(distances[i, j] * x[i, j] for i in bricks for j in srs)
        m.setObjective(distance + 0.0001 * disruption, GRB.MINIMIZE)
        m.optimize()
```
<div style="page-break-after: always;"></div>

**Mathematical Formulation:**
```
Minimize:    Σ_i Σ_j d_ij · x_ij + 0.0001 · Σ_i Σ_j w_i · y_ij

Subject to:  All constraints from Models 1 & 2
             Σ_i Σ_j w_i · y_ij ≤ ε    (epsilon constraint on disruption)
```

**Key Feature:** The small coefficient (0.0001) on disruption in the objective ensures we find **efficient (Pareto-optimal)** solutions by breaking ties in favor of lower disruption.

### 3.2 Pareto Frontiers: Three Workload Scenarios

We generate Pareto frontiers for three workload flexibility levels:

#### **Scenario 1: [0.8, 1.2] - High Flexibility (±20%)**

| Metric | Value |
|--------|-------|
| Pareto Solutions | 25 |
| Distance Range | [165.96 km, 188.95 km] |
| Disruption Range | [0.1696, 0.5864] |

![Pareto Frontier - Scenario 1](STEP1_Graph/pareto_Scenario_1_0.8_1.2.png)
*Figure 3: Pareto frontier for workload bounds [0.8, 1.2]*

![Workload Distribution - Scenario 1](STEP1_Graph/workload_Scenario_1_0.8_1.2.png)
*Figure 4: Workload distribution across Pareto solutions (Scenario 1)*

#### **Scenario 2: [0.85, 1.15] - Medium Flexibility (±15%)**

| Metric | Value |
|--------|-------|
| Pareto Solutions | 25 |
| Distance Range | [171.68 km, 187.24 km] |
| Disruption Range | [0.2529, 0.4867] |

![Pareto Frontier - Scenario 2](STEP1_Graph/pareto_Scenario_2_0.85_1.15.png)
*Figure 5: Pareto frontier for workload bounds [0.85, 1.15]*

![Workload Distribution - Scenario 2](STEP1_Graph/workload_Scenario_2_0.85_1.15.png)
*Figure 6: Workload distribution across Pareto solutions (Scenario 2)*

#### **Scenario 3: [0.9, 1.1] - Low Flexibility (±10%)**

| Metric | Value |
|--------|-------|
| Pareto Solutions | 25 |
| Distance Range | [171.68 km, 187.24 km] |
| Disruption Range | [0.2529, 0.4867] |

![Pareto Frontier - Scenario 3](STEP1_Graph/pareto_Scenario_3_0.9_1.1.png)
*Figure 7: Pareto frontier for workload bounds [0.9, 1.1]*

![Workload Distribution - Scenario 3](STEP1_Graph/workload_Scenario_3_0.9_1.1.png)
*Figure 8: Workload distribution across Pareto solutions (Scenario 3)*

### 3.3 Comparative Analysis

![Pareto Frontier Comparison](STEP1_Graph/pareto_comparison.png)
*Figure 9: Comparison of Pareto frontiers across three workload scenarios*

**Summary Table:**

| Scenario | Workload Range | Pareto Points | Best Distance | Best Disruption | Distance at Min Disr |
|----------|----------------|---------------|---------------|-----------------|----------------------|
| **1: [0.8, 1.2]** | ±20% | 25 | 165.96 km | 0.1696 | 188.95 km |
| **2: [0.85, 1.15]** | ±15% | 25 | 171.68 km | 0.2529 | 187.24 km |
| **3: [0.9, 1.1]** | ±10% | 25 | 171.68 km | 0.2529 | 187.24 km |

**Key Insights:**

1. **Workload Flexibility Impact:**
   - More flexible bounds ([0.8, 1.2]) enable **5.72 km better** distance optimization (165.96 vs 171.68 km)
   - Tighter bounds restrict solution space, reducing potential for distance savings
   - Scenarios 2 and 3 produce identical frontiers (constraints become equivalently restrictive)

2. **Trade-off Characteristics:**
   - **Steep region (low disruption)**: Small disruption increases yield large distance savings
   - **Flat region (high disruption)**: Diminishing returns on distance as disruption grows
   - Decision-makers can identify "sweet spot" solutions balancing both objectives

3. **Practical Implications:**
   - Solutions at disruption ~0.3 offer good compromise: 10-12% distance reduction with moderate change
   - Extreme solutions (min distance or min disruption) may not be practical
   - Pareto frontier provides 20-25 distinct alternatives for management consideration

<div style="page-break-after: always;"></div>

## 4. DECISION SPACE VISUALIZATION

### 4.1 Solution Characteristics

The generated Pareto solutions provide decision-makers with a spectrum of choices:

- **Left side of frontier** (low disruption): Preserves most current relationships, modest distance improvements
- **Middle region** (balanced): Achieves substantial distance savings (10-15%) with acceptable disruption
- **Right side of frontier** (low distance): Maximum efficiency, but significant organizational change required

### 4.2 Workload Balance Analysis

All Pareto solutions satisfy workload constraints within specified bounds. The visualizations show:
- Scenario 1 ([0.8, 1.2]): Widest workload variation, most solution flexibility
- Scenarios 2 & 3 ([0.85, 1.15] and [0.9, 1.1]): Tighter distributions, more uniform SR workloads

### 4.3 Assignment Pattern Changes

The heatmaps (Figures 1-2) reveal:
- **High-index bricks** (e.g., brick 14: 0.82, brick 15: 0.41): Critical for disruption metric
- **Geographic clustering**: Distance-optimal solutions group nearby bricks
- **Stability patterns**: Min-disruption solutions maintain core territory structures

<div style="page-break-after: always;"></div>

## 5. CONCLUSIONS

### 5.1 Model Performance

Both mono-objective models solve efficiently (< 1 second) for the 22×4 problem:
- **Model 1** identifies 17.5% distance savings potential
- **Model 2** provides stable baseline with minimal change
- Epsilon-constraint method generates comprehensive Pareto frontiers (~25 solutions per scenario)

### 5.2 Key Findings

1. **Clear Trade-off Exists:** Distance and disruption objectives conflict; no single optimal solution
2. **Workload Flexibility Matters:** Relaxing constraints from ±10% to ±20% improves best distance by 3.3%
3. **Multiple Good Solutions:** Pareto frontiers offer 20+ alternatives spanning the efficiency-stability spectrum
4. **Computational Efficiency:** All models solve optimally in under 2 seconds, suitable for interactive decision support

### 5.3 Decision Support

For Pfizer Turkey management, we recommend:
- **Conservative approach:** Solutions near disruption 0.25-0.30 (maintain ~75% of current assignments)
- **Moderate approach:** Solutions near disruption 0.35-0.40 (balance efficiency and stability)
- **Aggressive approach:** Solutions near disruption 0.45-0.50 (maximize distance savings)

The choice depends on organizational priorities:
- If SR-MD relationships are critical → favor low disruption
- If operational costs dominate → favor low distance
- If both matter equally → select middle-ground solutions


<div style="page-break-after: always;"></div>
---

## APPENDIX A: IMPLEMENTATION

**Code Structure:**
- `STEP1_script/pfizer_optimization.py`: Main optimization framework (460 lines)
  - `PfizerOptimization` class with Model 1, Model 2, epsilon-constraint method
  - Data loading, solution extraction, analysis utilities
- `STEP1_script/pareto_analysis.py`: Pareto frontier generation and visualization (347 lines)
  - Generates all three scenarios
  - Creates publication-quality plots
  - Exports CSV data for further analysis

**Data Files:**
- `data/indexValues.xlsx`: Workload (index values) per brick
- `data/distances.xlsx`: Distance matrix (brick → SR office)

**Generated Outputs:**
- `STEP1_Graph/pareto_Scenario_*.png`: Individual Pareto frontiers
- `STEP1_Graph/pareto_comparison.png`: Comparative visualization
- `STEP1_Graph/workload_Scenario_*.png`: Workload distributions
- `STEP1_Graph/assignment_heatmap_*.png`: Territory assignment heatmaps
- `STEP1_Graph/pareto_Scenario_*.csv`: Solution data for each scenario
- `STEP1_Graph/pareto_summary.csv`: Summary statistics

**Reproducibility:**
```bash
# Run both models on 22×4 instance
python STEP1_script/pfizer_optimization.py

# Generate all Pareto frontiers and visualizations
python STEP1_script/pareto_analysis.py
```

---

## APPENDIX B: COMPLETE RESULTS DATA

**Model 1 (Minimize Distance) - Scenario [0.8, 1.2]:**

| SR | Assigned Bricks | Workload | Distance (km) |
|----|----------------|----------|---------------|
| 1 | 4,5,6,7,8,9,12,19,20 | 1.0376 | 64.37 |
| 2 | 11,13,14,18 | 1.0447 | 7.53 |
| 3 | 10,15,16,17 | 1.1149 | 6.57 |
| 4 | 1,2,3,21,22 | 0.8028 | 76.13 |
| **Total** | - | **4.0000** | **154.60** |

**Model 2 (Minimize Disruption) - Scenario [0.8, 1.2]:**

| SR | Assigned Bricks | Workload | Distance (km) |
|----|----------------|----------|---------------|
| 1 | 4,5,6,7,8,15 | 0.9507 | 19.30 |
| 2 | 10,13,14 | 1.1681 | 7.82 |
| 3 | 9,11,12,16,17,18 | 0.8744 | 37.09 |
| 4 | 1,2,3,19,20,21,22 | 1.0068 | 124.74 |
| **Total** | - | **4.0000** | **188.95** |

**Current Assignment (Baseline):**

| SR | Assigned Bricks | Workload | Distance (km) |
|----|----------------|----------|---------------|
| 1 | 4,5,6,7,8,15 | 0.9507 | 19.30 |
| 2 | 10,11,12,13,14 | 1.1680 | 31.82 |
| 3 | 9,16,17,18 | 0.8748 | 12.55 |
| 4 | 1,2,3,19,20,21,22 | 1.0068 | 123.74 |
| **Total** | - | **4.0000** | **187.41** |

**Pareto Frontier Summary (Selected Solutions):**

*Scenario 1: [0.8, 1.2]*

| Solution # | Distance (km) | Disruption | Trade-off |
|------------|---------------|------------|-----------|
| 1 (Min Dist) | 165.96 | 0.5864 | Maximum efficiency |
| 5 | 167.25 | 0.4569 | Balanced |
| 10 | 171.43 | 0.3523 | Moderate |
| 15 | 177.62 | 0.2641 | Conservative |
| 25 (Min Disr) | 188.95 | 0.1696 | Maximum stability |

---

**End of Report**
