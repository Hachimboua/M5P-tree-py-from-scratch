"""
M5P Model Tree - Prediction Module
===================================

This module handles prediction for fitted M5P model trees.
It is designed to be independent from the training logic,
receiving only the fitted tree structure.

The prediction process:
1. For each sample, traverse the tree from root to leaf
2. At each internal node, follow left/right based on split condition
3. At the leaf, use the linear regression model to predict

Optional smoothing combines predictions from ancestor nodes
for more robust estimates.

Author: Member 3 (ZAyoub)
Project: ENSAM M5P Implementation
"""

import numpy as np
from typing import Optional, List, Any

# Type alias for the node class (avoid circular import)
# The actual M5PNode class is defined in model.py
M5PNode = Any


def predict_single(
    tree: M5PNode,
    x: np.ndarray,
    smoothing: bool = True
) -> float:
    """
    Predict the target value for a single sample.
    
    Traverses the tree from root to leaf following split conditions,
    then uses the leaf's linear model for the final prediction.
    
    Parameters
    ----------
    tree : M5PNode
        Root node of the fitted M5P tree
    x : np.ndarray of shape (n_features,)
        Single sample feature vector
    smoothing : bool, default=True
        If True, apply smoothing using ancestor predictions
    
    Returns
    -------
    prediction : float
        Predicted target value
    
    Notes
    -----
    When smoothing is enabled, the prediction at each node is
    combined with its parent's prediction using the formula:
    
        smoothed = (node_pred * n_node + parent_pred * k) / (n_node + k)
    
    where k is a smoothing constant (typically 15) and n_node
    is the number of training samples at that node.
    """
    if smoothing:
        return _predict_with_smoothing(tree, x)
    else:
        return _predict_simple(tree, x)


def _predict_simple(node: M5PNode, x: np.ndarray) -> float:
    """
    Simple prediction without smoothing.
    
    Traverse to leaf and use the leaf's linear model.
    
    Parameters
    ----------
    node : M5PNode
        Current node in traversal
    x : np.ndarray
        Feature vector
    
    Returns
    -------
    float
        Prediction from the leaf's linear model
    """
    # Base case: reached a leaf node
    if node.is_leaf:
        return _predict_with_linear_model(node, x)
    
    # Recursive case: follow the appropriate branch
    if x[node.feature_index] <= node.split_value:
        # Go left: feature value <= split threshold
        return _predict_simple(node.left, x)
    else:
        # Go right: feature value > split threshold
        return _predict_simple(node.right, x)


def _predict_with_smoothing(
    node: M5PNode,
    x: np.ndarray,
    smoothing_constant: float = 15.0
) -> float:
    """
    Prediction with smoothing from ancestor nodes.
    
    Smoothing combines the leaf prediction with predictions
    from ancestor nodes, weighted by sample counts. This
    helps reduce variance, especially for leaves with few samples.
    
    Parameters
    ----------
    node : M5PNode
        Current node (start with root)
    x : np.ndarray
        Feature vector
    smoothing_constant : float
        Smoothing parameter k (higher = more smoothing)
    
    Returns
    -------
    float
        Smoothed prediction
    """
    # Collect path from root to leaf
    path = _get_path_to_leaf(node, x)
    
    if len(path) == 0:
        return 0.0
    
    # Start with the leaf prediction
    leaf_node = path[-1]
    prediction = _predict_with_linear_model(leaf_node, x)
    
    # Apply smoothing from leaf back to root
    # We traverse the path in reverse (leaf -> root)
    for i in range(len(path) - 2, -1, -1):
        parent_node = path[i]
        child_node = path[i + 1]
        
        # Get parent's linear model prediction (if available)
        # For internal nodes, use the mean as fallback
        parent_pred = parent_node.node_mean
        if hasattr(parent_node, 'linear_model') and parent_node.linear_model is not None:
            parent_pred = _predict_with_linear_model(parent_node, x)
        
        # Smoothing formula from M5P paper
        n_child = child_node.n_samples
        k = smoothing_constant
        
        # Weighted combination of child and parent predictions
        prediction = (
            prediction * n_child + parent_pred * k
        ) / (n_child + k)
    
    return prediction


def _get_path_to_leaf(
    node: M5PNode,
    x: np.ndarray
) -> List[M5PNode]:
    """
    Get the path from root to the appropriate leaf.
    
    Parameters
    ----------
    node : M5PNode
        Root node to start from
    x : np.ndarray
        Feature vector determining the path
    
    Returns
    -------
    path : list of M5PNode
        Ordered list of nodes from root to leaf
    """
    path = []
    current = node
    
    while current is not None:
        path.append(current)
        
        if current.is_leaf:
            break
        
        # Determine which child to follow
        if x[current.feature_index] <= current.split_value:
            current = current.left
        else:
            current = current.right
    
    return path


def _predict_with_linear_model(
    node: M5PNode,
    x: np.ndarray
) -> float:
    """
    Use a node's linear model to make a prediction.
    
    Falls back to the node mean if no linear model is available.
    
    Parameters
    ----------
    node : M5PNode
        Node with the linear model
    x : np.ndarray
        Feature vector
    
    Returns
    -------
    float
        Prediction from the linear model
    """
    if node.linear_model is None:
        # Fallback: use the mean of training samples at this node
        return node.node_mean
    
    # Use the regression module's predict method
    # The linear model should have a predict method
    try:
        # If the linear model has a predict method for single samples
        if hasattr(node.linear_model, 'predict_single'):
            return node.linear_model.predict_single(x)
        elif hasattr(node.linear_model, 'predict'):
            # Reshape for batch prediction interface
            pred = node.linear_model.predict(x.reshape(1, -1))
            return float(pred[0])
        else:
            # Direct computation: y = X @ coefficients + intercept
            coefficients = getattr(node.linear_model, 'coefficients', None)
            intercept = getattr(node.linear_model, 'intercept', 0.0)
            
            if coefficients is not None:
                return float(np.dot(x, coefficients) + intercept)
            else:
                return node.node_mean
    except Exception:
        # Ultimate fallback
        return node.node_mean


def predict_batch(
    tree: M5PNode,
    X: np.ndarray,
    smoothing: bool = True
) -> np.ndarray:
    """
    Predict target values for multiple samples.
    
    This is the main entry point for batch prediction,
    called by M5PModel.predict().
    
    Parameters
    ----------
    tree : M5PNode
        Root node of the fitted M5P tree
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix where each row is a sample
    smoothing : bool, default=True
        Whether to apply prediction smoothing
    
    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Predicted target values
    
    Examples
    --------
    >>> predictions = predict_batch(fitted_tree, X_test, smoothing=True)
    """
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float64)
    
    # Predict each sample
    for i in range(n_samples):
        predictions[i] = predict_single(
            tree=tree,
            x=X[i],
            smoothing=smoothing
        )
    
    return predictions


def predict_with_paths(
    tree: M5PNode,
    X: np.ndarray,
    smoothing: bool = True
) -> tuple:
    """
    Predict values and return the paths taken for each sample.
    
    Useful for model interpretation and debugging.
    
    Parameters
    ----------
    tree : M5PNode
        Root node of the fitted M5P tree
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    smoothing : bool
        Whether to apply smoothing
    
    Returns
    -------
    predictions : np.ndarray
        Predicted values
    paths : list of list
        For each sample, the list of nodes traversed
    """
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float64)
    paths = []
    
    for i in range(n_samples):
        x = X[i]
        path = _get_path_to_leaf(tree, x)
        paths.append(path)
        
        if smoothing:
            predictions[i] = _predict_with_smoothing(tree, x)
        else:
            predictions[i] = _predict_simple(tree, x)
    
    return predictions, paths


def get_leaf_indices(
    tree: M5PNode,
    X: np.ndarray
) -> np.ndarray:
    """
    Get the leaf node index for each sample.
    
    Useful for understanding which samples end up in which
    regions of the feature space.
    
    Parameters
    ----------
    tree : M5PNode
        Root node of the fitted tree
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    
    Returns
    -------
    leaf_indices : np.ndarray of shape (n_samples,)
        Integer index of the leaf node for each sample.
        Leaves are numbered in left-to-right order.
    """
    # First, enumerate all leaves
    leaves = []
    _collect_leaves(tree, leaves)
    leaf_to_index = {id(leaf): i for i, leaf in enumerate(leaves)}
    
    # For each sample, find which leaf it reaches
    n_samples = X.shape[0]
    indices = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        path = _get_path_to_leaf(tree, X[i])
        leaf = path[-1]
        indices[i] = leaf_to_index.get(id(leaf), -1)
    
    return indices


def _collect_leaves(node: M5PNode, leaves: list) -> None:
    """
    Recursively collect all leaf nodes in left-to-right order.
    
    Parameters
    ----------
    node : M5PNode
        Current node
    leaves : list
        List to append leaves to (modified in place)
    """
    if node is None:
        return
    
    if node.is_leaf:
        leaves.append(node)
    else:
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)


# =============================================================================
# Utility Functions for Analysis
# =============================================================================

def compute_leaf_statistics(
    tree: M5PNode,
    X: np.ndarray,
    y: np.ndarray
) -> dict:
    """
    Compute prediction statistics for each leaf.
    
    Useful for model diagnostics and understanding
    which regions of the feature space are well-modeled.
    
    Parameters
    ----------
    tree : M5PNode
        Fitted tree root
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True target values
    
    Returns
    -------
    stats : dict
        Dictionary with leaf statistics including:
        - 'leaf_counts': samples per leaf
        - 'leaf_mse': MSE per leaf
        - 'leaf_mae': MAE per leaf
    """
    leaf_indices = get_leaf_indices(tree, X)
    predictions = predict_batch(tree, X, smoothing=False)
    
    unique_leaves = np.unique(leaf_indices)
    
    stats = {
        'leaf_counts': {},
        'leaf_mse': {},
        'leaf_mae': {},
        'leaf_r2': {}
    }
    
    for leaf_idx in unique_leaves:
        mask = leaf_indices == leaf_idx
        y_true = y[mask]
        y_pred = predictions[mask]
        
        n = np.sum(mask)
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² score for this leaf
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        stats['leaf_counts'][leaf_idx] = int(n)
        stats['leaf_mse'][leaf_idx] = float(mse)
        stats['leaf_mae'][leaf_idx] = float(mae)
        stats['leaf_r2'][leaf_idx] = float(r2)
    
    return stats
