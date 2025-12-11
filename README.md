# M5P Model Tree - Python Implementation from Scratch

A complete implementation of the M5P model tree algorithm in Python, built from scratch without relying on existing machine learning libraries (except NumPy for numerical operations).

## What is M5P?

M5P is a decision tree algorithm for regression that combines the interpretability of decision trees with the predictive power of linear regression models. Unlike standard regression trees that predict constant values at leaves, M5P builds linear regression models at each leaf node.

### Key Features

- **Model Trees**: Linear regression models at leaf nodes instead of constant predictions
- **Splitting Criterion**: Uses Standard Deviation Reduction (SDR) to find optimal splits
- **Pruning**: Optional reduced error pruning to prevent overfitting
- **Pure Python**: Implemented from scratch with only NumPy dependency

## Installation

No installation required! Just make sure you have NumPy installed:

```bash
pip install numpy scikit-learn  # scikit-learn only needed for examples
```

## Quick Start

```python
import numpy as np
from m5p_tree import M5PTree

# Create some sample data
X = np.random.randn(100, 3)
y = 2*X[:, 0] + X[:, 1]**2 - X[:, 2] + np.random.normal(0, 0.1, 100)

# Train the M5P tree
model = M5PTree(
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=5,
    use_pruning=True
)

model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Check R² score
r2_score = model.score(X, y)
print(f"R² Score: {r2_score:.4f}")

# Visualize the tree structure
model.print_tree()
```

## Algorithm Overview

### 1. Splitting Criterion

M5P uses **Standard Deviation Reduction (SDR)** to evaluate splits:

```
SDR = sd(T) - (|T_left|/|T| * sd(T_left) + |T_right|/|T| * sd(T_right))
```

Where:
- `sd(T)` is the standard deviation of target values in node T
- `T_left` and `T_right` are the left and right child nodes
- The split with maximum SDR is chosen

### 2. Tree Building Process

1. Start with all training data at the root
2. For each node:
   - If stopping criteria met → create leaf with linear model
   - Otherwise → find best split using SDR
   - Recursively build left and right subtrees
3. Fit linear regression models at each leaf node

### 3. Pruning (Optional)

Reduced error pruning:
- Bottom-up traversal of the tree
- For each internal node, compare:
  - Error with current split
  - Error if converted to leaf
- Keep the configuration with lower error

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_samples_split` | int | 10 | Minimum samples required to split a node |
| `min_samples_leaf` | int | 5 | Minimum samples required at a leaf |
| `max_depth` | int | None | Maximum tree depth (None = unlimited) |
| `use_pruning` | bool | True | Whether to apply pruning after building |

## API Reference

### M5PTree Class

#### Methods

- **`fit(X, y)`**: Build the tree from training data
  - `X`: Feature matrix (n_samples, n_features)
  - `y`: Target values (n_samples,)
  - Returns: self

- **`predict(X)`**: Predict target values
  - `X`: Feature matrix (n_samples, n_features)
  - Returns: Predictions (n_samples,)

- **`score(X, y)`**: Calculate R² score
  - `X`: Feature matrix
  - `y`: True target values
  - Returns: R² score (float)

- **`print_tree()`**: Print tree structure to console

## Examples

Run the example script to see various use cases:

```bash
python example.py
```

The examples include:
1. **Synthetic Regression**: Multi-feature synthetic dataset
2. **Non-linear Function**: Learning quadratic relationships
3. **Multidimensional Data**: Complex function with interactions
4. **Hyperparameter Comparison**: Comparing different settings

## How It Works

### Linear Models at Leaves

Each leaf node contains a linear regression model:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Coefficients are computed using ordinary least squares:

```
β = (XᵀX)⁻¹Xᵀy
```

### Decision Process

When predicting for a new sample:
1. Start at root node
2. Follow split decisions down the tree
3. At leaf node, use linear model to predict
4. Return prediction

## Advantages of M5P

1. **Better Predictions**: Linear models capture local trends better than constants
2. **Interpretable**: Tree structure is easy to understand
3. **Handles Non-linearity**: Tree partitions space, models capture local linear trends
4. **Smoothness**: Linear models provide smoother predictions than step functions

## Limitations

- Requires sufficient data at each leaf to fit linear models
- Can be sensitive to outliers (linear regression assumption)
- May not perform well with very high-dimensional data
- Computational cost higher than standard regression trees

## File Structure

```
M5P-tree-py-from-scratch/
├── m5p_tree.py          # Main implementation
├── example.py           # Usage examples
└── README.md            # This file
```

## Implementation Details

### Classes

1. **`LinearModel`**: Simple linear regression for leaf nodes
2. **`Node`**: Tree node (internal or leaf)
3. **`M5PTree`**: Main M5P algorithm implementation

### Key Algorithms

- **Split Finding**: Exhaustive search over all features and split points
- **SDR Calculation**: Weighted standard deviation reduction
- **Linear Regression**: Normal equation solver with fallback
- **Pruning**: Bottom-up reduced error pruning

## Performance Tips

1. **Adjust `min_samples_split`**: Larger values → simpler trees, faster training
2. **Use `max_depth`**: Limit depth to prevent overfitting and speed up training
3. **Enable pruning**: Usually improves generalization
4. **Scale features**: Linear models at leaves benefit from scaled features

## Comparison with Standard Trees

| Feature | M5P Tree | Standard Regression Tree |
|---------|----------|-------------------------|
| Leaf predictions | Linear models | Constant values |
| Smoothness | Smooth within regions | Step function |
| Complexity | Higher | Lower |
| Accuracy | Generally better | Simpler baseline |

## Future Enhancements

Potential improvements:
- [ ] Smoothing predictions across leaf boundaries
- [ ] Feature importance calculation
- [ ] Parallel split finding
- [ ] Support for categorical features
- [ ] Cross-validation utilities
- [ ] Visualization tools

## License

This is a educational implementation. Feel free to use and modify as needed.

## References

- Wang, Y., & Witten, I. H. (1997). Induction of model trees for predicting continuous classes.
- Quinlan, J. R. (1992). Learning with continuous classes.

## Contributing

This is a from-scratch implementation for learning purposes. Contributions welcome!

---

**Note**: This implementation prioritizes clarity and educational value over performance. For production use, consider scikit-learn or other optimized libraries.
