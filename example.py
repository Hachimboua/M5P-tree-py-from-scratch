import numpy as np
from model import M5P
from predict import predict


np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.5

model = M5P(min_samples_split=10, max_depth=5)
model.fit(X, y)

predictions = predict(model, X[:5])
print("Predictions:", predictions)
print("Actual:", y[:5])
print("R² score:", model.score(X, y))
