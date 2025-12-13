import numpy as np
from regression import fit_linear_model, predict_linear


def compute_error(node, model):
    predictions = predict_linear(model, node.X)
    return np.mean(np.abs(node.y - predictions))


def subtree_error(node):
    if node.is_leaf:
        return compute_error(node, node.linear_model)
    
    left_error = subtree_error(node.left) * len(node.left.y)
    right_error = subtree_error(node.right) * len(node.right.y)
    total_samples = len(node.left.y) + len(node.right.y)
    
    return (left_error + right_error) / total_samples


def prune_tree(node):
    if node.is_leaf:
        return
    
    prune_tree(node.left)
    prune_tree(node.right)
    
    subtree_err = subtree_error(node)
    linear_err = compute_error(node, node.linear_model)

    if linear_err <= subtree_err:
        node.is_leaf = True
        node.left = None
        node.right = None


def smooth_predictions(node, parent_model=None, k=15):
    if parent_model is None:
        node.smoothed_model = node.linear_model
    else:
        n = len(node.y)
        
        intercept = (n * node.linear_model['intercept'] + k * parent_model['intercept']) / (n + k)
        coeffs = (n * node.linear_model['coefficients'] + k * parent_model['coefficients']) / (n + k)
        
        node.smoothed_model = {
            'intercept': intercept,
            'coefficients': coeffs
        }
    
    if not node.is_leaf:
        smooth_predictions(node.left, node.smoothed_model, k)
        smooth_predictions(node.right, node.smoothed_model, k)
