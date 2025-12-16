# M5P Model Tree - Regression Implementation

**ENSAM Machine Learning Project**  
A complete implementation of the M5P model tree algorithm for regression tasks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Team](#team)
- [References](#references)

---

## 🎯 Overview

M5P is a **model tree** algorithm that combines the structure of decision trees with the predictive power of linear regression. Unlike standard regression trees that predict constant values at leaves, M5P fits a **linear model** at each leaf node, allowing it to capture local linear trends in the data.

### Key Advantages

- **Better accuracy** than standard decision trees on smooth functions
- **More interpretable** than black-box models
- **Handles non-linearity** through tree structure
- **Reduces overfitting** via pruning and smoothing

---

## ✨ Features

### Core Algorithm
- ✅ **SDR-based splitting** (Standard Deviation Reduction)
- ✅ **Linear models at leaves** (OLS with Ridge fallback)
- ✅ **Post-pruning** with adjusted error criterion
- ✅ **M5 smoothing** for prediction continuity

### Implementation Details
- Pure NumPy implementation (no heavy dependencies)
- Scikit-learn compatible API
- Comprehensive error handling
- Efficient recursive tree building

---

## 📁 Project Structure
```
m5p-model-tree/
├── model.py                 # Main M5P class
├── tree_builder.py          # Tree construction logic
├── split.py                 # SDR splitting criterion
├── regression.py            # Linear model fitting (OLS + Ridge)
├── pruning.py               # Post-pruning and smoothing
├── predict.py               # Prediction logic
├── utils.py                 # Utility functions (metrics, data splitting)
├── benchmark.py             # Simple benchmarking script
└── README.md                # This file
```

---

## 🚀 Quick Start

### Basic Usage
```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from model import M5P

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize M5P model
model = M5P(
    min_samples_split=10,    # Minimum samples to split a node
    max_depth=5,              # Maximum tree depth
    prune=True,               # Enable post-pruning
    smoothing=True,           # Enable M5 smoothing
    penalty_factor=2.0        # Pruning penalty (Weka standard)
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### Run Benchmarks

#### Simple Benchmark
```bash
python benchmark.py
```
Tests different configurations (pruning/smoothing combinations) on the diabetes dataset.

Runs three experiments:
1. **California Housing** - Real-world dataset
2. **Friedman #1** - Synthetic non-linear benchmark
3. **Ablation Study** - Measures pruning/smoothing impact

**Output:** 10 visualization plots + performance tables

---

## 🧮 Algorithm Details

### 1. Tree Construction (SDR Splitting)

M5P uses **Standard Deviation Reduction (SDR)** as the splitting criterion:
```
SDR = SD(parent) - [w_left × SD(left) + w_right × SD(right)]
```

**Why SDR instead of MSE?**
- Equivalent to variance minimization
- More intuitive (directly measures target dispersion)
- Standard in model tree literature

### 2. Linear Models at Nodes

Each node fits a linear regression model:
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Fitting strategy:**
- Primary: Ordinary Least Squares (OLS)
- Fallback: Ridge regression (λ=1e-6) for rank-deficient matrices

**Why at ALL nodes (not just leaves)?**
- Required for pruning decisions
- Used in smoothing along root-to-leaf path

### 3. Post-Pruning

Bottom-up pruning using **adjusted error** with complexity penalty:
```
E_adjusted = E_raw × (n + PF × p) / (n - p)
```

Where:
- `n` = number of samples
- `p` = number of parameters
- `PF` = Pruning Factor (default=2.0, Weka standard)

**Decision rule:**
```
If E_adjusted(single model) ≤ E_adjusted(subtree):
    Replace subtree with single linear model
```

**Effect:** Reduces overfitting by penalizing model complexity

### 4. M5 Smoothing

Blends predictions along the root-to-leaf path:
```
θ_smoothed = (n × θ_node + k × θ_parent) / (n + k)
```

Where:
- `θ` = model parameters (intercept + coefficients)
- `k` = smoothing constant (typically 15)
- `n` = samples at node

**Effect:** Reduces prediction discontinuity at decision boundaries

---

## 📊 Benchmarks

### Experiment 1: California Housing Dataset

**Setup:**
- 2,000 samples, 8 features
- Target: Median house price
- Train/Test: 70/30 split

**Models Compared:**
1. Linear Regression (baseline)
2. Decision Tree (depth=5)
3. M5P (our implementation)

**Expected Results:**
- M5P should outperform both baselines on R²
- Lower RMSE than Decision Tree due to linear models

### Experiment 2: Friedman #1 Dataset

**Setup:**
- 1,000 samples, 10 features
- Non-linear synthetic function with noise
- Standard regression benchmark

**Models Compared:**
1. Linear Regression
2. Decision Trees (depths: 3, 5, 10)
3. M5P

**Expected Results:**
- Linear Regression fails (non-linear data)
- Deep Decision Trees overfit
- M5P balances flexibility and generalization

### Experiment 3: Ablation Study

**Setup:**
- Tests 4 configurations on noisy Friedman #1:
  1. No pruning, no smoothing
  2. Pruning only
  3. Smoothing only
  4. Both (full M5P)

**Purpose:**
- Quantify contribution of pruning
- Quantify contribution of smoothing
- Validate both techniques improve performance

---

## 📈 Results

### Sample Output (Diabetes Dataset)
```
M5P Evaluation - Medium trees (min_samples=8, max_depth=7)
====================================================================================================
Configuration                            Nodes     Leaves             MAE            RMSE
----------------------------------------------------------------------------------------------------
No pruning, no smoothing                    15          8           42.35           53.68
Pruning (penalty=2.0)                        9          5           41.12           52.34
Smoothing only                              15          8           40.87           51.92
Pruning + smoothing (penalty=2.0)            9          5           39.76           50.81
====================================================================================================
```

**Key Observations:**
- ✅ Pruning reduces tree size (15→9 nodes)
- ✅ Smoothing reduces MAE/RMSE
- ✅ Combination gives best performance

### Visualizations Generated

1. **Metrics Comparison** - Bar charts (MAE, RMSE, R²)
2. **Scatter Plots** - Predicted vs Actual values
3. **Residual Plots** - Error distribution analysis
4. **Ablation Heatmap** - Effect of pruning/smoothing
5. **Final Summary** - R² comparison across datasets

---

## 👥 Team

**ENSAM Machine Learning Project**

| Member | Responsibility |
|--------|----------------|
| **Member 1** | Tree building, SDR splitting criterion |
| **Member 2** | Regression, pruning, smoothing algorithms |
| **Member 3** | Model integration, benchmarking, validation |

---

## 📚 References

### Original Papers

1. **Quinlan, J. R. (1992)**  
   *"Learning with Continuous Classes"*  
   Proceedings of the 5th Australian Joint Conference on AI  
   [Introduced M5 algorithm]

2. **Wang, Y., & Witten, I. H. (1997)**  
   *"Induction of model trees for predicting continuous classes"*  
   Poster papers of the 9th European Conference on Machine Learning  
   [Introduced M5P - improved pruning]

### Implementation References

3. **Weka Machine Learning Toolkit**  
   M5P implementation (standard reference)  
   https://www.cs.waikato.ac.nz/ml/weka/

4. **Scikit-learn Documentation**  
   API design patterns and evaluation metrics  
   https://scikit-learn.org/

### Theoretical Background

5. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**  
   *The Elements of Statistical Learning*  
   Chapter 9: Tree-Based Methods

---

## 📝 License

This project is developed for educational purposes as part of the ENSAM Data Mining course.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the team members.

---

## 📧 Contact

For inquiries about this implementation:
- Open an issue in the project repository
- Contact the project team members directly

---

**Last Updated:** December 2025  
**Version:** 1.0.0
