"""
M5P Model Tree - Main Model Interface
======================================

This module implements the main M5P model class, providing a scikit-learn-like
API for training and prediction on regression tasks.

The M5P algorithm (Quinlan, 1992) builds a model tree where:
1. Internal nodes contain split conditions (like a decision tree)
2. Leaf nodes contain linear regression models (unlike standard regression trees)

This hybrid approach combines the interpretability of decision trees with
the predictive power of linear models.

Author: Member 3 (ZAyoub)
Project: ENSAM M5P Implementation
"""

import numpy as np
from typing import Optional, Union, Any

# Import teammate modules
from tree_builder import build_tree
from regression import fit_linear_model, LinearRegressionModel
from pruning import prune_tree
from predict import predict_batch


class M5PNode:
    """
    Represents a node in the M5P model tree.
    
    This class serves as the fundamental building block for the tree structure.
    Internal nodes contain split information, while leaf nodes contain
    linear regression models.
    
    Attributes
    ----------
    is_leaf : bool
        True if this node is a leaf (terminal) node
    feature_index : int or None
        Index of the feature used for splitting (None for leaves)
    split_value : float or None
        Threshold value for the split (None for leaves)
    left : M5PNode or None
        Left child node (samples where feature <= split_value)
    right : M5PNode or None
        Right child node (samples where feature > split_value)
    linear_model : LinearRegressionModel or None
        Linear regression model for prediction (only for leaves)
    n_samples : int
        Number of training samples that reached this node
    node_mean : float
        Mean of target values at this node (used as fallback)
    """
    
    def __init__(self):
        """Initialize an empty M5P node."""
        self.is_leaf: bool = False
        self.feature_index: Optional[int] = None
        self.split_value: Optional[float] = None
        self.left: Optional['M5PNode'] = None
        self.right: Optional['M5PNode'] = None
        self.linear_model: Optional[Any] = None
        self.n_samples: int = 0
        self.node_mean: float = 0.0
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.is_leaf:
            return f"M5PNode(leaf, n_samples={self.n_samples})"
        return f"M5PNode(split: X[{self.feature_index}] <= {self.split_value:.4f})"


class M5PModel:
    """
    M5P Model Tree for Regression.
    
    M5P (Model 5 Prime) is an algorithm that builds a tree structure where
    each leaf node contains a linear regression model. This combines the
    piecewise partitioning of decision trees with local linear models,
    often achieving better accuracy than either approach alone.
    
    The algorithm proceeds in three phases:
    1. Tree Construction: Build a regression tree using variance reduction
    2. Model Fitting: Fit linear models at each leaf using the training data
    3. Pruning (optional): Simplify the tree by replacing subtrees with
       linear models when this reduces estimated error
    
    Parameters
    ----------
    min_samples_split : int, default=10
        Minimum number of samples required to split an internal node.
        Higher values prevent overfitting but may underfit.
    
    min_samples_leaf : int, default=5
        Minimum number of samples required at each leaf node.
        Ensures each linear model has enough data for stable fitting.
    
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited depth.
        Used to prevent overly complex trees.
    
    pruning : bool, default=True
        Whether to apply post-pruning to simplify the tree.
        Pruning replaces subtrees with linear models when beneficial.
    
    smoothing : bool, default=True
        Whether to apply smoothing to predictions.
        Smoothing combines predictions from ancestor nodes.
    
    Attributes
    ----------
    tree_ : M5PNode or None
        The root node of the fitted tree. None before fitting.
    
    n_features_ : int
        Number of features seen during fit.
    
    is_fitted_ : bool
        Whether the model has been fitted.
    
    Examples
    --------
    >>> import numpy as np
    >>> from model import M5PModel
    >>> 
    >>> # Generate sample data
    >>> X = np.random.randn(100, 3)
    >>> y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    >>> 
    >>> # Fit the model
    >>> model = M5PModel(min_samples_leaf=5, pruning=True)
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)
    
    References
    ----------
    Quinlan, J.R. (1992). Learning with Continuous Classes. 
    Proceedings of the 5th Australian Joint Conference on AI, pp. 343-348.
    
    Wang, Y. & Witten, I.H. (1997). Induction of Model Trees for 
    Predicting Continuous Classes.
    """
    
    def __init__(
        self,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_depth: Optional[int] = None,
        pruning: bool = True,
        smoothing: bool = True
    ):
        """
        Initialize the M5P model with hyperparameters.
        
        Parameters
        ----------
        min_samples_split : int
            Minimum samples required to attempt a split
        min_samples_leaf : int
            Minimum samples required in each leaf
        max_depth : int or None
            Maximum tree depth (None = unlimited)
        pruning : bool
            Whether to apply post-pruning
        smoothing : bool
            Whether to smooth predictions using ancestor models
        """
        # Validate parameters
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")
        
        # Store hyperparameters
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.pruning = pruning
        self.smoothing = smoothing
        
        # Initialize model state
        self.tree_: Optional[M5PNode] = None
        self.n_features_: int = 0
        self.is_fitted_: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'M5PModel':
        """
        Fit the M5P model tree to the training data.
        
        This method executes the three phases of M5P:
        1. Build the tree structure using variance reduction splits
        2. Fit linear regression models at each leaf node
        3. Optionally prune the tree to reduce complexity
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix. Each row is a sample.
        
        y : np.ndarray of shape (n_samples,)
            Target values for training.
        
        Returns
        -------
        self : M5PModel
            The fitted model instance (for method chaining).
        
        Raises
        ------
        ValueError
            If X and y have incompatible shapes.
        """
        # ===== Input Validation =====
        X, y = self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # ===== Phase 1: Build Tree Structure =====
        # The tree_builder module handles finding optimal splits
        # using standard deviation reduction (SDR) criterion
        self.tree_ = build_tree(
            X=X,
            y=y,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth
        )
        
        # ===== Phase 2: Fit Linear Models at Leaves =====
        # Traverse the tree and fit a linear regression model
        # at each leaf node using the samples that reach it
        self._fit_leaf_models(self.tree_, X, y)
        
        # ===== Phase 3: Pruning (Optional) =====
        # Prune subtrees where a linear model performs better
        # than the subtree, reducing complexity
        if self.pruning:
            self.tree_ = prune_tree(
                node=self.tree_,
                X=X,
                y=y
            )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        For each sample, traverses the tree from root to leaf,
        then uses the leaf's linear model to compute the prediction.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict. Must have same number of features
            as the training data.
        
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values for each sample.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If X has wrong number of features.
        """
        # Check if model is fitted
        if not self.is_fitted_:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )
        
        # Validate input
        X = self._validate_predict_input(X)
        
        # Delegate to predict module for batch prediction
        return predict_batch(
            tree=self.tree_,
            X=X,
            smoothing=self.smoothing
        )
    
    def _validate_input(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple:
        """
        Validate and prepare training input data.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
            
        Returns
        -------
        X, y : tuple of np.ndarray
            Validated and converted arrays
        """
        # Convert to numpy arrays if needed
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Check dimensions
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got shape {X.shape}"
            )
        
        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D array, got shape {y.shape}"
            )
        
        # Check sample count consistency
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different sample counts: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")
        
        return X, y
    
    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input data for prediction.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction
            
        Returns
        -------
        X : np.ndarray
            Validated array
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got shape {X.shape}"
            )
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained "
                f"with {self.n_features_} features"
            )
        
        return X
    
    def _fit_leaf_models(
        self,
        node: M5PNode,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Recursively fit linear regression models at leaf nodes.
        
        This method traverses the tree and, for each leaf node,
        fits a linear model using the samples that reach that leaf.
        
        Parameters
        ----------
        node : M5PNode
            Current node in the recursion
        X : np.ndarray
            Feature matrix for samples reaching this node
        y : np.ndarray
            Target values for samples reaching this node
        """
        if node is None:
            return
        
        # Store sample statistics at this node
        node.n_samples = len(y)
        node.node_mean = np.mean(y) if len(y) > 0 else 0.0
        
        if node.is_leaf:
            # ===== Fit Linear Model at Leaf =====
            # Use the regression module to fit a linear model
            # The model will be used for predictions
            node.linear_model = fit_linear_model(X, y)
        else:
            # ===== Recurse on Children =====
            # Split the data according to the node's split condition
            # and continue fitting on child nodes
            
            left_mask = X[:, node.feature_index] <= node.split_value
            right_mask = ~left_mask
            
            # Fit models in left subtree
            self._fit_leaf_models(
                node.left,
                X[left_mask],
                y[left_mask]
            )
            
            # Fit models in right subtree
            self._fit_leaf_models(
                node.right,
                X[right_mask],
                y[right_mask]
            )
    
    def get_n_leaves(self) -> int:
        """
        Count the number of leaf nodes in the fitted tree.
        
        Returns
        -------
        n_leaves : int
            Number of leaf nodes
        """
        if not self.is_fitted_:
            return 0
        return self._count_leaves(self.tree_)
    
    def _count_leaves(self, node: Optional[M5PNode]) -> int:
        """Recursively count leaf nodes."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return (
            self._count_leaves(node.left) + 
            self._count_leaves(node.right)
        )
    
    def get_depth(self) -> int:
        """
        Get the maximum depth of the fitted tree.
        
        Returns
        -------
        depth : int
            Maximum depth (root = depth 0)
        """
        if not self.is_fitted_:
            return 0
        return self._get_max_depth(self.tree_)
    
    def _get_max_depth(self, node: Optional[M5PNode]) -> int:
        """Recursively compute maximum depth."""
        if node is None or node.is_leaf:
            return 0
        return 1 + max(
            self._get_max_depth(node.left),
            self._get_max_depth(node.right)
        )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        if self.is_fitted_:
            return (
                f"M5PModel({status}, "
                f"n_leaves={self.get_n_leaves()}, "
                f"depth={self.get_depth()})"
            )
        return f"M5PModel({status})"
