"""
M5P Model Tree Implementation from Scratch

M5P is a decision tree algorithm that uses linear regression models at the leaf nodes.
It combines the interpretability of decision trees with the predictive power of linear models.
"""

import numpy as np
from typing import Optional, Union, Tuple


class LinearModel:
    """Simple linear regression model for leaf nodes."""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression using ordinary least squares.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        try:
            # Solve using normal equation: (X^T X)^-1 X^T y
            params = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            self.intercept = params[0]
            self.coefficients = params[1:]
        except np.linalg.LinAlgError:
            # Fallback to mean prediction if matrix is singular
            self.intercept = np.mean(y)
            self.coefficients = np.zeros(X.shape[1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the linear model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet")
        return self.intercept + X @ self.coefficients


class Node:
    """Node in the M5P tree."""
    
    def __init__(self):
        self.is_leaf = False
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.model = None  # Linear model for leaf nodes
        self.prediction = None  # Mean prediction for leaf nodes
        
    def predict_sample(self, x: np.ndarray) -> float:
        """
        Predict for a single sample.
        
        Args:
            x: Feature vector of shape (n_features,)
            
        Returns:
            Predicted value
        """
        if self.is_leaf:
            if self.model is not None:
                return self.model.predict(x.reshape(1, -1))[0]
            else:
                return self.prediction
        else:
            if x[self.split_feature] <= self.split_value:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)


class M5PTree:
    """
    M5P Model Tree implementation.
    
    Parameters:
    -----------
    min_samples_split : int, default=10
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=5
        Minimum number of samples required at a leaf node.
    max_depth : int, default=None
        Maximum depth of the tree. If None, nodes expand until all leaves
        contain less than min_samples_split samples.
    use_pruning : bool, default=True
        Whether to use pruning after building the tree.
    """
    
    def __init__(
        self,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_depth: Optional[int] = None,
        use_pruning: bool = True
    ):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.use_pruning = use_pruning
        self.root = None
        
    def _calculate_standard_deviation(self, y: np.ndarray) -> float:
        """Calculate standard deviation of target values."""
        if len(y) == 0:
            return 0.0
        return np.std(y)
    
    def _calculate_sdr(
        self,
        y: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """
        Calculate Standard Deviation Reduction (SDR).
        
        SDR = sd(T) - (|T_left|/|T| * sd(T_left) + |T_right|/|T| * sd(T_right))
        
        Args:
            y: Target values for current node
            y_left: Target values for left child
            y_right: Target values for right child
            
        Returns:
            Standard deviation reduction
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        sd_current = self._calculate_standard_deviation(y)
        sd_left = self._calculate_standard_deviation(y_left)
        sd_right = self._calculate_standard_deviation(y_right)
        
        sdr = sd_current - (n_left / n * sd_left + n_right / n * sd_right)
        return sdr
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split point for the current node.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            Tuple of (best_feature, best_value, best_sdr)
        """
        n_samples, n_features = X.shape
        best_sdr = 0.0
        best_feature = None
        best_value = None
        
        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try splits between consecutive unique values
            for i in range(len(unique_values) - 1):
                split_value = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split the data
                left_mask = feature_values <= split_value
                right_mask = ~left_mask
                
                # Check minimum samples constraint
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate SDR
                y_left = y[left_mask]
                y_right = y[right_mask]
                sdr = self._calculate_sdr(y, y_left, y_right)
                
                # Update best split
                if sdr > best_sdr:
                    best_sdr = sdr
                    best_feature = feature_idx
                    best_value = split_value
        
        return best_feature, best_value, best_sdr
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:
        """
        Recursively build the M5P tree.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            depth: Current depth of the tree
            
        Returns:
            Root node of the (sub)tree
        """
        node = Node()
        n_samples = len(y)
        
        # Check stopping criteria
        should_stop = (
            n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1
        )
        
        if should_stop:
            # Create leaf node
            node.is_leaf = True
            node.prediction = np.mean(y)
            
            # Fit linear model at leaf
            if n_samples >= 2 and X.shape[1] > 0:
                try:
                    model = LinearModel()
                    model.fit(X, y)
                    node.model = model
                except:
                    # Fallback to mean prediction
                    node.model = None
            
            return node
        
        # Find best split
        best_feature, best_value, best_sdr = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None or best_sdr <= 0:
            node.is_leaf = True
            node.prediction = np.mean(y)
            
            # Fit linear model at leaf
            if n_samples >= 2 and X.shape[1] > 0:
                try:
                    model = LinearModel()
                    model.fit(X, y)
                    node.model = model
                except:
                    node.model = None
            
            return node
        
        # Split the data
        left_mask = X[:, best_feature] <= best_value
        right_mask = ~left_mask
        
        # Store split information
        node.split_feature = best_feature
        node.split_value = best_value
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _prune_tree(self, node: Node, X: np.ndarray, y: np.ndarray) -> Node:
        """
        Prune the tree using reduced error pruning.
        
        Args:
            node: Current node to consider for pruning
            X: Feature matrix for this node
            y: Target values for this node
            
        Returns:
            Pruned node
        """
        if node.is_leaf:
            return node
        
        # Split data for children
        left_mask = X[:, node.split_feature] <= node.split_value
        right_mask = ~left_mask
        
        # Recursively prune children
        node.left = self._prune_tree(node.left, X[left_mask], y[left_mask])
        node.right = self._prune_tree(node.right, X[right_mask], y[right_mask])
        
        # Calculate error with current split
        predictions = np.array([node.predict_sample(x) for x in X])
        error_with_split = np.mean((y - predictions) ** 2)
        
        # Calculate error if we convert to leaf
        temp_node = Node()
        temp_node.is_leaf = True
        temp_node.prediction = np.mean(y)
        
        # Try fitting linear model
        if len(y) >= 2 and X.shape[1] > 0:
            try:
                model = LinearModel()
                model.fit(X, y)
                temp_node.model = model
            except:
                temp_node.model = None
        
        predictions_leaf = np.array([temp_node.predict_sample(x) for x in X])
        error_as_leaf = np.mean((y - predictions_leaf) ** 2)
        
        # Prune if leaf is better or similar (with tolerance)
        if error_as_leaf <= error_with_split * 1.01:  # 1% tolerance
            return temp_node
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build the M5P tree from training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Prune if requested
        if self.use_pruning:
            self.root = self._prune_tree(self.root, X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        predictions = np.array([self.root.predict_sample(x) for x in X])
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True target values of shape (n_samples,)
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def _print_tree(self, node: Node, depth: int = 0, prefix: str = ""):
        """Print tree structure for visualization."""
        if node.is_leaf:
            if node.model is not None:
                print(f"{prefix}Leaf: Linear Model (coef={node.model.coefficients})")
            else:
                print(f"{prefix}Leaf: Mean = {node.prediction:.4f}")
        else:
            print(f"{prefix}Split: X[{node.split_feature}] <= {node.split_value:.4f}")
            print(f"{prefix}├── Left:")
            self._print_tree(node.left, depth + 1, prefix + "│   ")
            print(f"{prefix}└── Right:")
            self._print_tree(node.right, depth + 1, prefix + "    ")
    
    def print_tree(self):
        """Print the tree structure."""
        if self.root is None:
            print("Tree has not been fitted yet")
        else:
            print("M5P Tree Structure:")
            print("=" * 50)
            self._print_tree(self.root)
