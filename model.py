import numpy as np
from tree_builder import M5PTree
from regression import fit_linear_model
from pruning import prune_tree, smooth_predictions


class M5P:
    def __init__(self, min_samples_split=10, max_depth=None, prune=True, smoothing=True, prune_threshold=0.95):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune = prune
        self.smoothing = smoothing
        self.prune_threshold = prune_threshold
        self.tree = None
    
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        self.tree = M5PTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        self.tree.fit(X, y)
        
        self._add_data_to_nodes(self.tree.root, X, y)
        self._add_linear_models(self.tree.root)
        
        if self.prune:
            prune_tree(self.tree.root, threshold=self.prune_threshold)
        
        if self.smoothing:
            smooth_predictions(self.tree.root)
        
        return self
    
    def _add_data_to_nodes(self, node, X, y):
        node.X = X
        node.y = y
        
        if not node.is_leaf and node.left is not None and node.right is not None:
            left_mask = X[:, node.feature] <= node.threshold
            right_mask = ~left_mask
            
            self._add_data_to_nodes(node.left, X[left_mask], y[left_mask])
            self._add_data_to_nodes(node.right, X[right_mask], y[right_mask])
    
    def _add_linear_models(self, node):
        if hasattr(node, 'X') and hasattr(node, 'y') and len(node.X) > 0:
            node.linear_model = fit_linear_model(node.X, node.y)
        
        if not node.is_leaf:
            if node.left is not None:
                self._add_linear_models(node.left)
            if node.right is not None:
                self._add_linear_models(node.right)
    
    def score(self, X, y):
        from predict import predict
        
        y_pred = predict(self, X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
