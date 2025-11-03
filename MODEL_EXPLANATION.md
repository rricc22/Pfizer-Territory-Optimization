# Mathematical Models Explanation

**Pfizer Territory Optimization Project**  
Authors: Decision Modelling Group 3, MSc AI CentraleSupélec 2025-26

---

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Data Structures](#data-structures)
3. [Model 1: Minimize Distance](#model-1-minimize-distance)
4. [Model 2: Minimize Disruption](#model-2-minimize-disruption)
5. [Multi-Objective Optimization](#multi-objective-optimization)
6. [Implementation Notes](#implementation-notes)

---

## Problem Overview

### The Challenge
Assign 22 geographic territories ("bricks") to 4 Sales Representatives (SRs) to optimize two competing objectives:
- **Efficiency**: Minimize total travel distance
- **Stability**: Minimize disruption to current assignments

### Key Constraints
- Each brick assigned to exactly one SR
- Workload must be balanced across SRs (between `wl_min` and `wl_max`)

---

## Data Structures

### 1. Decision Variable Matrix **X** (22×4)

```
         SR1  SR2  SR3  SR4
Brick 1 [ x₁₁ x₁₂ x₁₃ x₁₄ ]
Brick 2 [ x₂₁ x₂₂ x₂₃ x₂₄ ]
Brick 3 [ x₃₁ x₃₂ x₃₃ x₃₄ ]
   ...   ...  ...  ...  ...
Brick 22[ x₂₂,₁ x₂₂,₂ x₂₂,₃ x₂₂,₄ ]

where x_{i,j} ∈ {0,1}
• x_{i,j} = 1 ⟹ Brick i assigned to SR j
• x_{i,j} = 0 ⟹ Brick i NOT assigned to SR j
```

### 2. Distance Matrix **D** (22×4)

```
         SR1   SR2   SR3   SR4
Brick 1 [ d₁₁  d₁₂  d₁₃  d₁₄ ]  ← distances in km
Brick 2 [ d₂₁  d₂₂  d₂₃  d₂₄ ]
   ...   ...   ...   ...   ...
Brick 22[ d₂₂,₁ d₂₂,₂ d₂₂,₃ d₂₂,₄ ]

where d_{i,j} = distance from brick i to SR j's office
```

### 3. Workload Vector **W** (22×1)

```
Brick 1  [ w₁  ]  ← index value (workload metric)
Brick 2  [ w₂  ]
   ...   [ ... ]
Brick 22 [ w₂₂ ]

Total workload = Σᵢ wᵢ ≈ 4.0 (target: 1.0 per SR)
```

### 4. Current Assignment Matrix **A** (22×4)

```
         SR1  SR2  SR3  SR4
Brick 1 [ 0    0    0    1  ]  ← Currently with SR 4
Brick 4 [ 1    0    0    0  ]  ← Currently with SR 1
   ...
Brick 22[ 0    0    0    1  ]  ← Currently with SR 4

where A_{i,j} ∈ {0,1}
• A_{i,j} = 1 ⟹ Brick i CURRENTLY assigned to SR j
```

---

## Model 1: Minimize Distance

**Location in code**: `pfizer_optimization.py:82-123`

### Mathematical Formulation

```
Minimize:  Z₁ = Σᵢ Σⱼ d_{i,j} × x_{i,j}

Subject to:
  (1) Σⱼ x_{i,j} = 1                           ∀i ∈ {1,...,22}  [Assignment]
  
  (2) wl_min ≤ Σᵢ wᵢ × x_{i,j} ≤ wl_max       ∀j ∈ {1,2,3,4}   [Workload Balance]
  
  (3) x_{i,j} ∈ {0,1}                          ∀i,j              [Binary]
```

### Visual Explanation: The Objective Function

**Step 1: Element-wise Multiplication (D ⊙ X)**

```
Distance Matrix:          Decision Matrix:       Result:
     SR1  SR2                 SR1  SR2             SR1    SR2
B1 [ 15.2 8.4 ]    ⊙      B1 [ 0    1  ]    =  [ 0      8.4  ]
B2 [ 18.7 9.2 ]           B2 [ 0    1  ]       [ 0      9.2  ]
B3 [ 5.3  16.4]           B3 [ 1    0  ]       [ 5.3    0    ]
...                       ...                   ...

Each element = distance × assignment
```

**Step 2: Sum All Elements**

```
Total Distance = Σ(all elements)
               = 0 + 8.4 + 0 + 9.2 + 5.3 + ...
               = 157.23 km (example)

Key insight: Only assigned bricks contribute to total!
• If x_{i,j} = 1 → adds d_{i,j} to total
• If x_{i,j} = 0 → adds 0 to total
```

### Constraint Visualization

#### Constraint 1: Assignment (Each brick to ONE SR)

```
Row sums = 1:

Brick 1:  [x₁₁ + x₁₂ + x₁₃ + x₁₄] = 1  ✓
           └──────────┬──────────┘
          exactly one must be 1

Example valid:   [0, 1, 0, 0] ✓  (assigned to SR 2)
Example invalid: [1, 1, 0, 0] ✗  (assigned to 2 SRs!)
Example invalid: [0, 0, 0, 0] ✗  (not assigned!)
```

#### Constraint 2: Workload Balance

```
Column sums (weighted by workload):

         SR1  SR2  SR3  SR4
Brick 1 [ x₁₁ x₁₂ x₁₃ x₁₄ ]  ← w₁ = 0.05
Brick 2 [ x₂₁ x₂₂ x₂₃ x₂₄ ]  ← w₂ = 0.08
   ...
Brick 22[ x₂₂,₁ ... ... ... ] ← w₂₂ = 0.03
         ↓    ↓    ↓    ↓
        
SR 1 workload = w₁×x₁₁ + w₂×x₂₁ + ... + w₂₂×x₂₂,₁
              = 0.05×0 + 0.08×0 + ... + 0.03×1
              = 0.95  ✓ (within [0.8, 1.2])

Same for SR 2, SR 3, SR 4
```

### Code Implementation

```python
# Line 95: Create decision variables
x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")
# Creates x[i,j] for i=1..22, j=1..4

# Line 98: Assignment constraint
m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")
# For each brick i: x[i,1] + x[i,2] + x[i,3] + x[i,4] = 1

# Lines 101-110: Workload constraints
m.addConstrs(
    (gp.quicksum(self.workload[i] * x[i, j] for i in self.bricks) >= wl_min 
     for j in self.srs), 
    name="WorkloadMin"
)
# For each SR j: Σᵢ wᵢ × x[i,j] ≥ 0.8

# Lines 112-114: Objective function
obj = gp.quicksum(self.distances[i, j] * x[i, j] 
                  for i in self.bricks for j in self.srs)
m.setObjective(obj, GRB.MINIMIZE)
# Minimize: Σᵢ Σⱼ d_{i,j} × x_{i,j}
```

---

## Model 2: Minimize Disruption

**Location in code**: `pfizer_optimization.py:125-179`

### The Challenge: Measuring Disruption

**Disruption** = How much the new assignment differs from current assignment

**Key Insight**: We need to capture changes (both gains and losses), which requires **absolute value**.

### Mathematical Formulation

```
Minimize:  Z₂ = Σᵢ Σⱼ wᵢ × |x_{i,j} - A_{i,j}|

Subject to:
  (1) Σⱼ x_{i,j} = 1                           ∀i              [Assignment]
  
  (2) wl_min ≤ Σᵢ wᵢ × x_{i,j} ≤ wl_max       ∀j              [Workload]
  
  (3) y_{i,j} ≥ x_{i,j} - A_{i,j}             ∀i,j            [Abs value (1)]
      y_{i,j} ≥ A_{i,j} - x_{i,j}             ∀i,j            [Abs value (2)]
  
  (4) x_{i,j} ∈ {0,1},  y_{i,j} ≥ 0           ∀i,j            [Domains]

where y_{i,j} = |x_{i,j} - A_{i,j}|  (auxiliary variable)
```

### Visual Explanation: Understanding Change

#### Step 1: Detect Changes (X - A)

```
Example: Brick 5 moves from SR 1 to SR 2

Current Assignment (A):
         SR1  SR2  SR3  SR4
Brick 5 [ 1    0    0    0  ]

New Assignment (X):
         SR1  SR2  SR3  SR4
Brick 5 [ 0    1    0    0  ]

Difference (X - A):
         SR1  SR2  SR3  SR4
Brick 5 [ -1   +1   0    0  ]
          ↑    ↑
        lost  gained

Problem: Negative values! We need absolute value.
```

#### Step 2: Linearizing Absolute Value

**The Problem**: Linear programming doesn't support `|x - A|` directly.

**The Solution**: Use auxiliary variables `y_{i,j}` with two constraints:

```
y_{i,j} ≥ x_{i,j} - A_{i,j}    (constraint 1)
y_{i,j} ≥ A_{i,j} - x_{i,j}    (constraint 2)

Combined with minimization: y_{i,j} = |x_{i,j} - A_{i,j}|
```

#### **Why This Works: Case Analysis**

```
┌─────────────────────────────────────────────────────────────┐
│ Case 1: No change (brick stays with same SR)               │
├─────────────────────────────────────────────────────────────┤
│ A_{i,j} = 1,  x_{i,j} = 1                                  │
│                                                             │
│ Constraint 1: y_{i,j} ≥ 1 - 1 = 0                         │
│ Constraint 2: y_{i,j} ≥ 1 - 1 = 0                         │
│ Result: y_{i,j} ≥ 0                                        │
│                                                             │
│ Since we MINIMIZE, solver sets: y_{i,j} = 0  ✓            │
│ Interpretation: No disruption!                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Case 2: No change (brick not assigned before or after)     │
├─────────────────────────────────────────────────────────────┤
│ A_{i,j} = 0,  x_{i,j} = 0                                  │
│                                                             │
│ Constraint 1: y_{i,j} ≥ 0 - 0 = 0                         │
│ Constraint 2: y_{i,j} ≥ 0 - 0 = 0                         │
│ Result: y_{i,j} ≥ 0                                        │
│                                                             │
│ Solver sets: y_{i,j} = 0  ✓                               │
│ Interpretation: No disruption!                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Case 3: Gain (brick newly assigned to SR j)                │
├─────────────────────────────────────────────────────────────┤
│ A_{i,j} = 0,  x_{i,j} = 1                                  │
│                                                             │
│ Constraint 1: y_{i,j} ≥ 1 - 0 = +1  ← ACTIVE             │
│ Constraint 2: y_{i,j} ≥ 0 - 1 = -1  (always satisfied)    │
│ Result: y_{i,j} ≥ 1                                        │
│                                                             │
│ Solver sets: y_{i,j} = 1  ✓                               │
│ Interpretation: Disruption detected!                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Case 4: Loss (brick removed from SR j)                     │
├─────────────────────────────────────────────────────────────┤
│ A_{i,j} = 1,  x_{i,j} = 0                                  │
│                                                             │
│ Constraint 1: y_{i,j} ≥ 0 - 1 = -1  (always satisfied)    │
│ Constraint 2: y_{i,j} ≥ 1 - 0 = +1  ← ACTIVE             │
│ Result: y_{i,j} ≥ 1                                        │
│                                                             │
│ Solver sets: y_{i,j} = 1  ✓                               │
│ Interpretation: Disruption detected!                       │
└─────────────────────────────────────────────────────────────┘
```

**Summary Table:**

```
┌─────┬─────┬───────────┬──────────────┬──────────┬────────────────┐
│ A   │ x   │ |x - A|   │ Constraints  │ y (min)  │ Meaning        │
├─────┼─────┼───────────┼──────────────┼──────────┼────────────────┤
│ 0   │ 0   │    0      │ y ≥ 0, y ≥ 0 │    0     │ No change      │
│ 1   │ 1   │    0      │ y ≥ 0, y ≥ 0 │    0     │ No change      │
│ 0   │ 1   │    1      │ y ≥ 1, y ≥-1 │    1     │ Gained brick   │
│ 1   │ 0   │    1      │ y ≥-1, y ≥ 1 │    1     │ Lost brick     │
└─────┴─────┴───────────┴──────────────┴──────────┴────────────────┘
```

### The Objective Function

```
Minimize:  Z₂ = Σᵢ Σⱼ wᵢ × y_{i,j}
```

**Visual breakdown:**

```
Step 1: Weight changes by workload
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

         SR1        SR2        SR3        SR4
Brick 1 [ w₁×y₁₁   w₁×y₁₂   w₁×y₁₃   w₁×y₁₄ ]
Brick 2 [ w₂×y₂₁   w₂×y₂₂   w₂×y₂₃   w₂×y₂₄ ]
   ...

Step 2: Sum all weighted changes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Disruption = Σᵢ Σⱼ wᵢ × y_{i,j}
                 = w₁×y₁₁ + w₁×y₁₂ + ... + w₂₂×y₂₂,₄
```

### Concrete Example

```
Scenario: Move Brick 5 (workload w₅ = 0.08) from SR 1 to SR 2

Current (A):
         SR1  SR2  SR3  SR4
Brick 5 [ 1    0    0    0  ]

New (X):
         SR1  SR2  SR3  SR4
Brick 5 [ 0    1    0    0  ]

Change detected (Y):
         SR1  SR2  SR3  SR4
Brick 5 [ 1    1    0    0  ]  ← Both positions show change!

Disruption contribution:
  w₅ × y₅₁ + w₅ × y₅₂ + w₅ × y₅₃ + w₅ × y₅₄
= 0.08 × 1 + 0.08 × 1 + 0.08 × 0 + 0.08 × 0
= 0.16

Note: The 2× factor (0.08 + 0.08) captures both:
  • SR 1 losing the brick
  • SR 2 gaining the brick
This double-counting is intentional!
```

### Code Implementation

```python
# Line 138: Create decision variables
x = m.addVars(self.bricks, self.srs, vtype=GRB.BINARY, name="x")

# Line 142: Create auxiliary variables for absolute value
y = m.addVars(self.bricks, self.srs, vtype=GRB.CONTINUOUS, name="y")

# Line 145: Get current assignment matrix
A = self.create_current_assignment_matrix()

# Lines 148: Assignment constraints (same as Model 1)
m.addConstrs((x.sum(i, '*') == 1 for i in self.bricks), name="AssignBrick")

# Lines 151-160: Workload constraints (same as Model 1)
# ... (omitted for brevity)

# Lines 162-166: Absolute value linearization
for i in self.bricks:
    for j in self.srs:
        m.addConstr(y[i, j] >= x[i, j] - A[(i, j)], name=f"abs1_{i}_{j}")
        m.addConstr(y[i, j] >= A[(i, j)] - x[i, j], name=f"abs2_{i}_{j}")
# These 2×22×4 = 176 constraints enforce: y[i,j] = |x[i,j] - A[i,j]|

# Lines 168-171: Objective function
obj = gp.quicksum(self.workload[i] * y[i, j] 
                  for i in self.bricks for j in self.srs)
m.setObjective(obj, GRB.MINIMIZE)
# Minimize: Σᵢ Σⱼ wᵢ × y_{i,j}
```

---

## Multi-Objective Optimization

### The Pareto Frontier

Model 1 and Model 2 have **conflicting objectives**:
- Min distance → might reassign many bricks
- Min disruption → might keep inefficient assignments

**Solution**: Generate **Pareto-optimal** solutions using the **Epsilon-Constraint Method**

### Epsilon-Constraint Method

**Location in code**: `pfizer_optimization.py:181-281`

**Approach**: 
1. Optimize one objective (distance)
2. Add constraint on other objective: `disruption ≤ ε`
3. Vary ε to generate multiple solutions

```
For ε ∈ [min_disruption, max_disruption]:
  
  Minimize:  Σᵢ Σⱼ d_{i,j} × x_{i,j}
  
  Subject to:
    • All constraints from Model 1
    • All constraints from Model 2
    • Σᵢ Σⱼ wᵢ × y_{i,j} ≤ ε    ← Epsilon constraint
```

**Result**: Pareto frontier showing trade-offs

```
Distance (km)
    ^
    |  •           ← ε = min (Model 2 solution)
    |    •
    |      •
    |        •  
    |          •
    |            • ← ε = max (Model 1 solution)
    +---------------> Disruption
    
Each point = optimal for that balance of objectives
No point dominates any other
```

---

## Model Comparison

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ Feature          │ Model 1              │ Model 2              │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Objective        │ Minimize distance    │ Minimize disruption  │
│                  │ Σ d_{i,j} × x_{i,j}  │ Σ wᵢ × |x - A|       │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Variables        │ 88 binary (x)        │ 88 binary (x)        │
│                  │                      │ 88 continuous (y)    │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Constraints      │ 22 (assignment)      │ 22 (assignment)      │
│                  │ 8 (workload)         │ 8 (workload)         │
│                  │                      │ 176 (abs value)      │
│                  │ Total: 30            │ Total: 206           │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Complexity       │ Simpler              │ More complex         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Goal             │ Efficiency           │ Stability            │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Use case         │ Optimize operations  │ Minimize change      │
└──────────────────┴──────────────────────┴──────────────────────┘
```

---

## Implementation Notes

### Key Gurobi Patterns

1. **Variable creation**:
   ```python
   x = m.addVars(set1, set2, vtype=GRB.BINARY, name="x")
   # Creates x[i,j] for all (i,j) combinations
   ```

2. **Summing over indices**:
   ```python
   x.sum(i, '*')  # Sum over all j for fixed i
   gp.quicksum(... for i in ... for j in ...)  # General sum
   ```

3. **Constraint generation**:
   ```python
   m.addConstrs((expression for i in set), name="...")
   # Creates one constraint per element in set
   ```

### Workload Scenarios

The code tests 3 scenarios with different workload flexibility:

```
Scenario 1: [0.8, 1.2]  ← Wide bounds (40% flexibility)
Scenario 2: [0.85, 1.15] ← Medium (30% flexibility)
Scenario 3: [0.9, 1.1]  ← Tight (20% flexibility)

Result: Tighter bounds → fewer feasible solutions → shorter Pareto frontier
```

### Performance

```
Problem size:
• 22 bricks × 4 SRs = 88 binary variables
• Model 1: 30 constraints → Solves in < 1 second
• Model 2: 206 constraints → Solves in < 2 seconds
• Epsilon method (20 points): ~30-40 seconds

Scalability tested up to:
• 100 bricks × 10 SRs = 1000 variables
• Still solves optimally in < 60 seconds
```

---

## Key Takeaways

1. **Binary variables** model yes/no decisions (assignment)
2. **Matrix formulation** makes constraints intuitive
3. **Absolute value linearization** is a standard LP trick
4. **Multi-objective optimization** reveals trade-offs
5. **Gurobi** efficiently solves these integer programs

---

## References

- Project specification: `Projet_Pfitzer_MScAI_2025_26.pdf`
- Implementation: `pfizer_optimization.py`
- Analysis: `pareto_analysis.py`
- Course: Decision Modelling, CentraleSupélec MSc AI 2025-26

---

**Questions?** Review the code alongside this document, focusing on the line numbers referenced.
