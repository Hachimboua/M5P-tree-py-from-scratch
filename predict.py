import numpy as np
from regression import predict_linear


def predict(model, X):
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    predictions = np.array([_predict_single(model.tree.root, x, model.smoothing) for x in X])
    return predictions


def _predict_single(node, x, use_smoothing):
    leaf = _traverse_to_leaf(node, x)
    model = leaf.smoothed_model if (use_smoothing and hasattr(leaf, 'smoothed_model')) else leaf.linear_model
    return predict_linear(model, x)[0]


def _traverse_to_leaf(node, x):
    if node.is_leaf:
        return node
    
    if x[node.feature] <= node.threshold:
        return _traverse_to_leaf(node.left, x)
    else:
        return _traverse_to_leaf(node.right, x)
