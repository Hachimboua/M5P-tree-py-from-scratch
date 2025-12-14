import numpy as np
from model import M5P
from sklearn.datasets import make_regression

np.random.seed(42)

X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)

print("Testing pruning on deep tree...")
print("=" * 60)

model = M5P(min_samples_split=8, max_depth=7, prune=False, smoothing=False)
model.fit(X, y)
nodes_before = model.tree.count_nodes()
leaves_before = len(model.tree.get_leaves())

print(f"Before pruning: {nodes_before} nodes, {leaves_before} leaves")

model_pruned = M5P(min_samples_split=8, max_depth=7, prune=True, smoothing=False)
model_pruned.fit(X, y)
nodes_after = model_pruned.tree.count_nodes()
leaves_after = len(model_pruned.tree.get_leaves())

print(f"After pruning:  {nodes_after} nodes, {leaves_after} leaves")
print(f"Reduction:      {nodes_before - nodes_after} nodes removed ({100*(nodes_before - nodes_after)/nodes_before:.1f}%)")

from predict import predict

y_pred_no_prune = predict(model, X)
y_pred_pruned = predict(model_pruned, X)

mae_no_prune = np.mean(np.abs(y - y_pred_no_prune))
mae_pruned = np.mean(np.abs(y - y_pred_pruned))

print(f"\nTraining MAE without pruning: {mae_no_prune:.2f}")
print(f"Training MAE with pruning:    {mae_pruned:.2f}")
print(f"Error change:                 {mae_pruned - mae_no_prune:+.2f}")
print("=" * 60)
