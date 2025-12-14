"""
Integration test to verify all components work together
"""
import numpy as np
from tree_builder import TreeNode
from regression import fit_linear_model, predict_linear
from pruning import compute_error, subtree_error, prune_tree, smooth_predictions


def add_data_to_nodes(node, X, y):
    """Add training data to each node for pruning/smoothing"""
    node.X = X
    node.y = y
    
    if not node.is_leaf and node.left is not None and node.right is not None:
        left_mask = X[:, node.feature] <= node.threshold
        right_mask = ~left_mask
        
        add_data_to_nodes(node.left, X[left_mask], y[left_mask])
        add_data_to_nodes(node.right, X[right_mask], y[right_mask])


def add_linear_models(node):
    """Fit linear models at all nodes"""
    if hasattr(node, 'X') and hasattr(node, 'y') and len(node.X) > 0:
        node.linear_model = fit_linear_model(node.X, node.y)
    
    if not node.is_leaf:
        if node.left is not None:
            add_linear_models(node.left)
        if node.right is not None:
            add_linear_models(node.right)


def test_integration():
    print("Testing M5P components integration...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.5
    
    # Build tree
    from tree_builder import M5PTree
    tree = M5PTree(min_samples_split=10, max_depth=3)
    tree.fit(X, y)
    print(f"✓ Tree built: {tree.count_nodes()} nodes, {len(tree.get_leaves())} leaves")
    
    # Add training data to nodes
    add_data_to_nodes(tree.root, X, y)
    print("✓ Training data added to nodes")
    
    # Fit linear models at all nodes
    add_linear_models(tree.root)
    print("✓ Linear models fitted at all nodes")
    
    # Test regression functions
    sample_node = tree.get_leaves()[0]
    if hasattr(sample_node, 'linear_model'):
        pred = predict_linear(sample_node.linear_model, sample_node.X[:1])
        print(f"✓ Regression prediction works: {pred[0]:.3f}")
    
    # Test error computation
    if hasattr(sample_node, 'linear_model'):
        error = compute_error(sample_node, sample_node.linear_model)
        print(f"✓ Error computation works: MAE = {error:.3f}")
    
    # Test pruning
    initial_nodes = tree.count_nodes()
    prune_tree(tree.root)
    final_nodes = tree.count_nodes()
    print(f"✓ Pruning works: {initial_nodes} → {final_nodes} nodes")
    
    # Test smoothing
    smooth_predictions(tree.root)
    if hasattr(tree.root, 'smoothed_model'):
        print("✓ Smoothing works: smoothed_model created")
    
    print("\n✓ All components integrated successfully!")
    return tree


if __name__ == "__main__":
    test_integration()
