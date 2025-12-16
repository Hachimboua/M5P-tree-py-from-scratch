import numpy as np
from split import find_best_split, std_deviation


class TreeNode:
    """
    Node in M5P regression tree.
    
    Can be either:
    - Internal node: contains split rule (feature, threshold)
    - Leaf node: contains linear regression model (added later)
    """
    
    def __init__(self):
        # Split information (for internal nodes)
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        
        # Node statistics
        self.n_samples = 0
        self.std = 0.0
        self.is_leaf = False
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(samples={self.n_samples})"
        return f"Node(feature={self.feature}, threshold={self.threshold:.3f})"


class M5PTree:
    """
    M5P regression tree builder using SDR splitting criterion.
    
    Builds initial tree structure - linear models added later by M5P class.
    """
    
    def __init__(self, min_samples_split=4, max_depth=None):
        """
        Parameters:
        -----------
        min_samples_split : int
            Minimum samples required to attempt a split
        max_depth : int or None
            Maximum tree depth (None = unlimited)
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        """Build tree structure using recursive SDR-based splitting."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir la même longueur")
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build tree using greedy top-down approach.
        
        Stopping criteria:
        1. Too few samples (< min_samples_split)
        2. Maximum depth reached
        3. No variance to reduce (std ≈ 0)
        4. No valid split found
        """
        n_samples = len(y)
        
        node = TreeNode()
        node.n_samples = n_samples
        node.std = std_deviation(y)
        
        should_stop = False
        
        # Check stopping criteria
        if n_samples < self.min_samples_split:
            should_stop = True
        
        if self.max_depth is not None and depth >= self.max_depth:
            should_stop = True
        
        if node.std < 1e-7:  # All targets nearly identical
            should_stop = True
        
        if should_stop:
            node.is_leaf = True
            return node
        
        # Find best split using SDR criterion
        split = find_best_split(X, y, self.min_samples_split)
        
        # No valid split found (all features constant or min_samples violated)
        if split is None:
            node.is_leaf = True
            return node
        
        # Create internal node with split rule
        node.feature = split['feature']
        node.threshold = split['threshold']
        
        # Partition data and recursively build children
        left_mask = split['left_mask']
        right_mask = split['right_mask']
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def _traverse_to_leaf(self, x, node=None):
        """Navigate tree to find leaf for sample x."""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return node
        
        # Follow split rule
        if x[node.feature] <= node.threshold:
            return self._traverse_to_leaf(x, node.left)
        else:
            return self._traverse_to_leaf(x, node.right)
    
    def get_leaves(self):
        """Collect all leaf nodes (used for model fitting)."""
        leaves = []
        
        def collect_leaves(node):
            if node is None:
                return
            if node.is_leaf:
                leaves.append(node)
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)
        
        collect_leaves(self.root)
        return leaves
    
    def count_nodes(self):
        """Count total nodes (internal + leaves)."""
        def count(node):
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)
        
        return count(self.root)
    
    def get_depth(self):
        """Compute tree depth (longest root-to-leaf path)."""
        def depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        
        return depth(self.root)
    
    def print_tree(self, node=None, prefix="", is_left=True):
        """Visualize tree structure (for debugging/validation)."""
        if node is None:
            node = self.root
            print("M5P Tree:")
        
        if node is None:
            return
        
        connector = "L-- " if is_left else "R-- "
        
        if node.is_leaf:
            print(f"{prefix}{connector}Leaf (n={node.n_samples}, std={node.std:.3f})")
        else:
            print(f"{prefix}{connector}X[{node.feature}] <= {node.threshold:.3f} (n={node.n_samples})")
            
            new_prefix = prefix + ("|   " if is_left else "    ")
            if node.left:
                self.print_tree(node.left, new_prefix, True)
            if node.right:
                self.print_tree(node.right, new_prefix, False)


def test_tree():
    """Sanity check for tree building."""
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    
    tree = M5PTree(min_samples_split=10, max_depth=3)
    tree.fit(X, y)
    
    print(f"Arbre construit avec {tree.count_nodes()} noeuds")
    print(f"Profondeur: {tree.get_depth()}")
    print(f"Nombre de feuilles: {len(tree.get_leaves())}")
    print("\nStructure:")
    tree.print_tree()

if __name__ == "__main__":
    test_tree()
