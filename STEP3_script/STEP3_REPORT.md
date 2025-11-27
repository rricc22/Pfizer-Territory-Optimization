<style>
body { font-size: 10pt; }
h1 { font-size: 18pt; }
h2 { font-size: 14pt; }
h3 { font-size: 12pt; }
code, pre { font-size: 9pt; }
</style>

# STEP 3: Office Relocation with Variable Center Bricks

## 1. Problem Extension

In Steps 1 and 2, office locations were **fixed** at predefined center bricks. Step 3 relaxes this constraint by treating **office locations as decision variables**. This allows the optimization to select the most efficient center bricks for placing offices, potentially achieving better solutions at the cost of relocating existing offices.

### 1.1 Mathematical Formulation

**Decision Variables:**
- `x[i,j] ∈ {0,1}`: Brick `i` assigned to office at brick `j`
- `y[j] ∈ {0,1}`: Brick `j` contains an SR office (1 = office, 0 = no office)
- `wm ∈ ℝ⁺`: Maximum workload across all offices (for MinMax objective)

**Constraints:**
1. **Single Assignment:** Each brick assigned to exactly one office
   ```
   Σⱼ x[i,j] = 1,  ∀i ∈ Bricks
   ```
2. **Office Count:** Exactly `n` offices must be placed
   ```
   Σⱼ y[j] = n
   ```
3. **Assignment Validity:** Can only assign to bricks with offices
   ```
   x[i,j] ≤ y[j],  ∀i,j
   ```
4. **Workload Tracking:** (for MinMax objective only)
   ```
   wm ≥ Σᵢ w[i]·x[i,j] - M·(1-y[j]),  ∀j
   ```
   where `M` = large constant (Big-M method)

**Objectives:**
1. **Minimize Total Distance:** `Σᵢ,ⱼ d[i,j]·x[i,j]`
2. **Minimize Maximum Workload (MinMax):** `wm`
3. **Minimize Office Relocations:** `Σⱼ∈J₀ (1-y[j])` where `J₀` = initial office locations

### 1.2 Multi-Objective Approach

We use a **weighted sum method** to combine distance and workload objectives:

```
Objective = α·(distance/20) + (1-α)·workload_max
```

- **α = 0:** Pure workload minimization
- **α = 1:** Pure distance minimization  
- **α ∈ (0,1):** Balanced trade-off

By varying `α` from 0 to 1, we generate the Pareto frontier of non-dominated solutions.

<div style="page-break-after: always;"></div>

## 2. Implementation

### 2.1 Model 1: Minimize Total Distance

```python
def model_minimize_distance(self, verbose: bool = True):
    m = gp.Model("Step3_MinDistance")
    m.setParam('OutputFlag', 0 if not verbose else 1)
    
    # Decision variables
    x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
    y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
    
    # Constraints
    m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
    m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
    m.addConstrs(
        (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
        name="AssignOnlyToOffice"
    )
    
    # Objective: Minimize total distance
    obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                     for i in self.bricks for j in self.bricks)
    m.setObjective(obj, GRB.MINIMIZE)
    
    m.optimize()
    return m, self._extract_solution(m, x, y)
```
<div style="page-break-after: always;"></div>

### 2.2 Model 2: Minimize Maximum Workload (MinMax)

```python
def model_minimize_maxworkload(self, verbose: bool = True):
    m = gp.Model("Step3_MinMaxWorkload")
    m.setParam('OutputFlag', 0 if not verbose else 1)
    
    # Decision variables
    x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
    y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
    wm = m.addVar(vtype=GRB.CONTINUOUS, name="wm")  # max workload
    
    # Standard constraints
    m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
    m.addConstr(y.sum('*') == self.n_srs, name="ExactlyNOffices")
    m.addConstrs(
        (x[i, j] <= y[j] for i in self.bricks for j in self.bricks),
        name="AssignOnlyToOffice"
    )
    
    # Workload tracking (Big-M method)
    M = sum(self.workload.values()) + 1
    m.addConstrs(
        (wm >= gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) 
         - M * (1 - y[j]) for j in self.bricks),
        name="MaxWorkload"
    )
    
    # Objective: Minimize maximum workload
    m.setObjective(wm, GRB.MINIMIZE)
    
    m.optimize()
    return m, self._extract_solution(m, x, y)
```
<div style="page-break-after: always;"></div>

### 2.3 Bi-Objective Model (Weighted Sum)

```python
def model_biobjective(self, alpha: float = 0.5, verbose: bool = True):
    m = gp.Model("Step3_Biobjective")
    m.setParam('OutputFlag', 0 if not verbose else 1)
    
    # Decision variables
    x = m.addVars(self.bricks, self.bricks, vtype=GRB.BINARY, name="x")
    y = m.addVars(self.bricks, vtype=GRB.BINARY, name="y")
    wm = m.addVar(vtype=GRB.CONTINUOUS, name="wm")
    
    # Constraints (same as above)
    # ...
    
    # Bi-objective: weighted sum with normalization
    distance_obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                              for i in self.bricks for j in self.bricks)
    distance_norm = distance_obj / 20  # Expected range ~10-20
    workload_norm = wm                 # Expected range ~0.8-1.5
    
    obj = alpha * distance_norm + (1 - alpha) * workload_norm
    m.setObjective(obj, GRB.MINIMIZE)
    
    m.optimize()
    return m, self._extract_solution(m, x, y)
```

---

## 3. Results: 22 Bricks / 4 SRs

### 3.1 Pareto Frontier Analysis

We generated **15 Pareto-optimal solutions** by varying weight parameter `α` from 0 to 1:

![Pareto Frontier](STEP3_Graph/pareto_22x4_frontier.png)

**Key Observations:**
- **Distance range:** 16.57 - 149.62 (9× difference)
- **Workload range:** 1.0001 - 3.0000 (3× difference)
- Clear convex trade-off curve indicating efficient frontier
- Two extreme solutions dominate:
  - **Min Distance (α=1.0):** Distance = 16.57, Workload = 3.00
  - **Min Workload (α=0.0):** Distance = 149.62, Workload = 1.0001

<div style="page-break-after: always;"></div>

### 3.2 Trade-off Analysis

![Trade-off](STEP3_Graph/pareto_22x4_tradeoff.png)

The dual-axis plot shows how objectives evolve as `α` increases:
- **α < 0.3:** Workload improves rapidly with minimal distance increase
- **α ∈ [0.3, 0.7]:** Balanced region with moderate trade-offs (best practical solutions)
- **α > 0.7:** Distance improves at the expense of severe workload imbalance

**Recommended solution:** `α = 0.36` achieves:
- Distance: 20.44 (23% improvement over fixed offices)
- Max Workload: 1.38 (acceptable balance)
- Relocated offices: 4/4

<div style="page-break-after: always;"></div>

### 3.3 Relocation Impact

![Relocation](STEP3_Graph/pareto_22x4_relocation.png)

**Displaced Offices Distribution:**
- **3 displaced:** 2 solutions (13%)
- **4 displaced:** 13 solutions (87%)

**Key Insight:** Most efficient solutions require relocating **all 4 offices**. Only the first two solutions (α=0.0, α=0.07) keep 1 office in place, but these have extremely poor distance performance (≥29.68).

### 3.4 Effect of Weight Parameter α

![Alpha Effect](STEP3_Graph/pareto_22x4_alpha.png)

Normalized objectives (0-1 scale) show:
- **Workload** (red): Relatively flat for α > 0.3, sharp increase only at α=1.0
- **Distance** (blue): Smooth monotonic decrease as α increases
- **Sweet spot:** α ∈ [0.3, 0.5] balances both objectives effectively

<div style="page-break-after: always;"></div>

## 4. Comparison: Fixed vs Relocatable Offices

| Metric | Step 1 (Fixed) | Step 3 (Min Distance) | Improvement |
|--------|----------------|----------------------|-------------|
| **Total Distance** | 27.50 | 16.57 | **-39.7%** |
| **Max Workload** | 1.18 | 3.00 | +153.4% |
| **Office Locations** | [6, 13, 17, 21] | [5, 11, 12, 22] | 4/4 relocated |
| **Solve Time** | ~0.3s | ~1.2s | +300% |

| Metric | Step 1 (Fixed) | Step 3 (Balanced α=0.36) | Improvement |
|--------|----------------|--------------------------|-------------|
| **Total Distance** | 27.50 | 20.44 | **-25.7%** |
| **Max Workload** | 1.18 | 1.38 | +16.9% |
| **Office Locations** | [6, 13, 17, 21] | [5, 11, 12, 22] | 4/4 relocated |

**Key Findings:**
1. **Distance:** Allowing relocation reduces distance by **25-40%** depending on workload tolerance
2. **Workload:** Can achieve near-perfect balance (1.0001) but at 5× distance penalty
3. **Relocation Cost:** Most efficient solutions require relocating **all offices**
4. **Practical Choice:** Balanced solution (α=0.36) provides substantial distance improvement (+25.7%) with acceptable workload increase (+16.9%)

<div style="page-break-after: always;"></div>

## 5. Computational Performance

**Test Environment:** Gurobi 11.0, Intel i7-10750H, 16GB RAM

| Model | Variables | Constraints | Solve Time | Status |
|-------|-----------|-------------|------------|--------|
| Model 1 (Min Distance) | 484 binary, 22 binary | ~2,000 | 1.2s | Optimal |
| Model 2 (Min Workload) | 484 binary, 22 binary, 1 cont | ~2,500 | 1.8s | Optimal |
| Bi-objective (α=0.5) | 484 binary, 22 binary, 1 cont | ~2,500 | 1.5s | Optimal |
| Full Pareto (15 solves) | - | - | ~18s | All Optimal |

**Performance Notes:**
- Decision variables scale as `O(n²)` where `n` = number of bricks
- Constraint count scales as `O(n²)` due to assignment validity constraints
- All 22×4 instances solve in < 2 seconds
- **Scalability concern:** 100×10 instances take ~30-60s per solve (see Step 2 for details)

## 6. Practical Considerations

### 6.1 When to Use Office Relocation

**Advantages:**
- Significant distance reduction (25-40%)
- Flexibility to achieve near-perfect workload balance if needed
- Can adapt to changing demand patterns

**Disadvantages:**
- High disruption cost (most solutions relocate all offices)
- Requires physical office moves (real estate, equipment, morale impact)
- Longer solve times (~4× slower than fixed-office models)

**Recommendation:** Use office relocation when:
1. Existing office locations were chosen arbitrarily (not optimized)
2. Demand patterns have changed significantly since initial placement
3. Long-term distance savings justify one-time relocation cost
4. Organization can tolerate operational disruption

<div style="page-break-after: always;"></div>

### 6.2 Choosing the Right α

| α Range | Use Case | Typical Results |
|---------|----------|----------------|
| **0.0 - 0.2** | Workload fairness critical (union contracts, equity concerns) | Poor distance |
| **0.3 - 0.5** | **Balanced (RECOMMENDED)** | 20-30% distance improvement, acceptable workload |
| **0.6 - 0.8** | Distance critical (high travel costs) | Good distance, moderate imbalance |
| **0.9 - 1.0** | Pure distance optimization | Best distance, severe imbalance (not practical) |



## 7. Extensions and Future Work

### 7.1 Three-Objective Model

The code includes a three-objective epsilon-constraint method to simultaneously optimize:
1. Total distance
2. Maximum workload
3. Number of relocated offices

This allows decision-makers to **limit relocation count** while optimizing distance/workload. For example:
- "Find best solution relocating at most 2 offices"
- "Compare 0, 1, 2, 3, 4 relocation scenarios on Pareto curve"
<div style="page-break-after: always;"></div>

Implementation sketch:
```python
def model_three_objective_epsilon(self, num_relocated: int, 
                                 max_workload_eps: float = None):
    # Standard model setup
    # ...
    
    # Epsilon constraint: Exactly num_relocated offices moved
    initial_kept = self.n_srs - num_relocated
    m.addConstr(
        gp.quicksum(y[j] for j in self.initial_offices) == initial_kept,
        name="RelocatedOffices"
    )
    
    # Optional: Bound on max workload
    if max_workload_eps is not None:
        m.addConstr(wm <= max_workload_eps, name="MaxWorkloadEpsilon")
    
    # Primary objective: Minimize distance
    m.setObjective(distance_obj, GRB.MINIMIZE)
```

### 7.2 Incremental Relocation

For practical deployment, consider **staged relocation**:
1. Phase 1: Relocate 1-2 offices with highest impact
2. Observe results, measure actual travel time
3. Phase 2: Relocate remaining offices if justified

This reduces risk and allows validation of model assumptions.

### 7.3 Soft Relocation Penalties

Instead of hard constraints, penalize relocations in the objective:
```python
relocation_penalty = 1000  # Cost per office moved
obj = distance_obj + relocation_penalty * gp.quicksum(1 - y[j] for j in initial_offices)
```

This produces a single optimal solution balancing all three objectives automatically.

<div style="page-break-after: always;"></div>

## 8. Conclusions

**Step 3 demonstrates:**
1. **Flexibility of office relocation** yields 25-40% distance improvements over fixed offices
2. **Multi-objective optimization** reveals trade-offs between distance, workload, and disruption
3. **Pareto frontier** provides decision-makers with range of efficient solutions (no single "best" answer)
4. **Practical balance** (α=0.36) achieves substantial distance reduction with acceptable workload increase

**Model Complexity Evolution:**
- **Step 1:** Fixed offices, mono-objective → Simple, fast, limited flexibility
- **Step 2:** Fixed offices, scalability (100×10) → Larger problem size
- **Step 3:** Variable offices, multi-objective → High flexibility, more complex trade-offs

**Final Recommendation:**
- Use **Step 1 (fixed offices)** for routine reassignments and minor demand changes
- Use **Step 3 (relocatable offices)** for major reorganizations or initial office placement
- For Step 3, choose **α ∈ [0.3, 0.5]** for practical balance between distance and workload

https://github.com/rricc22/Pfizer-Territory-Optimization.git