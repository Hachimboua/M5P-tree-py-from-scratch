import numpy as np
from regression import predict_linear

def predict(model, X):
    """
    Make predictions for multiple samples.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
        
    Returns:
    --------
    predictions : ndarray, shape (n_samples,)
        Predicted values
    """
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    predictions = np.array([_predict_single(model.tree.root, x, model.smoothing) for x in X])
    return predictions

def _predict_single(node, x, use_smoothing):
    """
    Predict a single sample by traversing tree to leaf.
    
    Uses smoothed model if available and enabled, otherwise uses leaf's linear model.
    """
    leaf = _traverse_to_leaf(node, x)
    
    if leaf is None:
        return 0.0
    
    if use_smoothing and hasattr(leaf, 'smoothed_model') and leaf.smoothed_model is not None:
        linear_model = leaf.smoothed_model
    elif hasattr(leaf, 'linear_model') and leaf.linear_model is not None:
        linear_model = leaf.linear_model
    else:
        return leaf.node_mean if hasattr(leaf, 'node_mean') else 0.0
    
    result = predict_linear(linear_model, x)
    
    if isinstance(result, np.ndarray):
        return result[0] if len(result) > 0 else 0.0
    else:
        return float(result)

def _traverse_to_leaf(node, x):
    """
    Traverse tree from node to appropriate leaf for sample x.
    
    Follows split rules: go left if x[feature] <= threshold, else right.
    """
    if node is None:
        return None
    
    if node.is_leaf:
        return node
    
    if node.left is None and node.right is None:
        return node
    
    if x[node.feature] <= node.threshold:
        if node.left is not None:
            return _traverse_to_leaf(node.left, x)
        else:
            return node
    else:
        if node.right is not None:
            return _traverse_to_leaf(node.right, x)
        else:
            return node
