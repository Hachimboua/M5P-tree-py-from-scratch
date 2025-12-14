# M5P Model Tree - From Scratch

Clean implementation of the M5P algorithm using only NumPy.

## Usage

```python
import numpy as np
from model import M5P
from predict import predict

# Generate data
X = np.random.randn(100, 3)
y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.5

# Fit model
model = M5P(min_samples_split=10, max_depth=5)
model.fit(X, y)

# Make predictions
predictions = predict(model, X)

# Score
score = model.score(X, y)
```

## Files

- `model.py` - Main M5P model class
- `predict.py` - Prediction functions
- `regression.py` - Linear regression with OLS
- `split.py` - Splitting criterion (SDR)
- `tree_builder.py` - Tree construction
- `pruning.py` - Pruning and smoothing

## Run Tests

```bash
python example.py
python test_complete.py
```
