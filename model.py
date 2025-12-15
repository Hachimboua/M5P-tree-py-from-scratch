import numpy as np
from tree_builder import M5PTree
from regression import fit_linear_model
from pruning import prune_tree, smooth_predictions


class M5P:
    """
    M5P Model Tree for regression.
    Builds a tree with linear models at leaves instead of constant values.
    """
    def __init__(self, min_samples_split=10, max_depth=None, prune=True, smoothing=True, 
                 penalty_factor=2.0, use_weka_formula=False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.prune = prune  # Apply post-pruning
        self.smoothing = smoothing  # Apply M5 smoothing
        self.penalty_factor = penalty_factor  # Complexity penalty factor
        self.use_weka_formula = use_weka_formula  # Use original Weka formula vs AIC
        self.tree = None
    
    def fit(self, X, y):
        """Build M5P model tree from training data."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Build initial tree using SDR splitting
        self.tree = M5PTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        self.tree.fit(X, y)
        
        # Attach data to nodes for pruning and model building
        self._add_data_to_nodes(self.tree.root, X, y)
        
        # Fit linear models at all nodes
        self._add_linear_models(self.tree.root)
        
        # Apply bottom-up pruning if requested
        if self.prune:
            prune_tree(self.tree.root, self.penalty_factor, self.use_weka_formula)
        
        # Apply M5 smoothing if requested
        if self.smoothing:
            smooth_predictions(self.tree.root)
        
        return self
    
    def _add_data_to_nodes(self, node, X, y):
        """Recursively attach training data to each node."""
        if node is None:
            return
        
        node.X = X
        node.y = y
        node.n_samples = len(y)
        node.node_mean = np.mean(y) if len(y) > 0 else 0.0
        
        # Recursively split data for child nodes
        if not node.is_leaf and node.left is not None and node.right is not None:
            left_mask = X[:, node.feature] <= node.threshold
            right_mask = ~left_mask
            
            self._add_data_to_nodes(node.left, X[left_mask], y[left_mask])
            self._add_data_to_nodes(node.right, X[right_mask], y[right_mask])
    
    def _add_linear_models(self, node):
        """Recursively fit linear models at all nodes."""
        if node is None:
            return
        
        # Fit linear model using OLS with ridge fallback
        if hasattr(node, 'X') and hasattr(node, 'y') and len(node.X) > 0:
            node.linear_model = fit_linear_model(node.X, node.y)
        else:
            node.linear_model = None
        
        # Recursively fit models in subtrees
        if not node.is_leaf:
            if node.left is not None:
                self._add_linear_models(node.left)
            if node.right is not None:
                self._add_linear_models(node.right)
    
    def predict(self, X):
        """Make predictions on new data."""
        from predict import predict
        X = np.array(X, dtype=float)
        return predict(self, X)
    
    def score(self, X, y):
        """Compute R² score on given data."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
