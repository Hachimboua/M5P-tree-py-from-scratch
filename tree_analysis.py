import numpy as np
import matplotlib.pyplot as plt
from tree_builder import M5PTree
from sklearn.datasets import fetch_california_housing

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class TreeStructureAnalyzer:
    def __init__(self, tree, X_train, y_train):
        self.tree = tree.root if hasattr(tree, 'root') else tree
        self.X_train = X_train
        self.y_train = y_train
        
    def analyze_tree_structure(self):
        structure = {
            'total_nodes': self._count_nodes(self.tree),
            'total_leaves': self._count_leaves(self.tree),
            'max_depth': self._get_max_depth(self.tree),
            'avg_leaf_samples': self._get_avg_leaf_samples(self.tree),
            'leaf_distribution': self._get_leaf_distribution(self.tree)
        }
        return structure
    
    def _count_nodes(self, node):
        if node is None or node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def _get_max_depth(self, node, current_depth=0):
        if node is None or node.is_leaf:
            return current_depth
        return max(
            self._get_max_depth(node.left, current_depth + 1),
            self._get_max_depth(node.right, current_depth + 1)
        )
    
    def _get_avg_leaf_samples(self, node):
        leaves = self._collect_leaves(node)
        if not leaves:
            return 0
        return np.mean([leaf.n_samples for leaf in leaves])
    
    def _collect_leaves(self, node):
        if node is None:
            return []
        if node.is_leaf:
            return [node]
        return self._collect_leaves(node.left) + self._collect_leaves(node.right)
    
    def _get_leaf_distribution(self, node):
        leaves = self._collect_leaves(node)
        if not leaves:
            return []
        return [leaf.n_samples for leaf in leaves]
    
    def validate_sdr_splits(self):
        results = {
            'splits': [],
            'sdr_values': [],
            'split_quality': []
        }
        self._validate_node_splits(self.tree, results)
        return results
    
    def _validate_node_splits(self, node, results, depth=0, path="Root"):
        if node is None or node.is_leaf:
            return
        
        samples_mask = np.ones(len(self.y_train), dtype=bool)
        X_node = self.X_train[samples_mask]
        y_node = self.y_train[samples_mask]
        
        actual_sdr = self._calculate_sdr(
            y_node,
            X_node[:, node.feature] <= node.threshold
        )
        
        alt_splits = self._evaluate_alternative_splits(X_node, y_node, node.feature)
        
        results['splits'].append({
            'depth': depth,
            'feature': node.feature,
            'threshold': node.threshold,
            'samples': len(y_node),
            'path': path
        })
        
        results['sdr_values'].append(actual_sdr)
        
        max_alt_sdr = max(alt_splits) if alt_splits else actual_sdr
        quality = actual_sdr / (max_alt_sdr + 1e-10)
        results['split_quality'].append(quality)
        
        if node.left:
            self._validate_node_splits(node.left, results, depth + 1, path + "->L")
        if node.right:
            self._validate_node_splits(node.right, results, depth + 1, path + "->R")
    
    def _calculate_sdr(self, y, split_mask):
        sd_parent = np.std(y)
        y_left = y[split_mask]
        y_right = y[~split_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        sd_left = np.std(y_left)
        sd_right = np.std(y_right)
        
        n = len(y)
        sdr = sd_parent - (len(y_left) / n) * sd_left - (len(y_right) / n) * sd_right
        
        return sdr
    
    def _evaluate_alternative_splits(self, X, y, exclude_feature=None):
        sdr_values = []
        
        for feature_idx in range(min(X.shape[1], 3)):
            if feature_idx == exclude_feature:
                continue
            
            threshold = np.median(X[:, feature_idx])
            split_mask = X[:, feature_idx] <= threshold
            
            if split_mask.sum() > 0 and (~split_mask).sum() > 0:
                sdr = self._calculate_sdr(y, split_mask)
                sdr_values.append(sdr)
        
        return sdr_values
    
    def interpret_splits(self, feature_names=None):
        interpretation = {
            'overall_strategy': "",
            'split_interpretations': [],
            'leaf_characteristics': []
        }
        
        structure = self.analyze_tree_structure()
        interpretation['overall_strategy'] = (
            f"Tree with {structure['total_nodes']} nodes, {structure['total_leaves']} leaves, "
            f"max depth {structure['max_depth']}. "
            f"Average {structure['avg_leaf_samples']:.1f} samples per leaf."
        )
        
        self._interpret_node_splits(self.tree, interpretation, feature_names)
        
        return interpretation
    
    def _interpret_node_splits(self, node, results, feature_names=None, depth=0):
        if node is None:
            return
        
        if node.is_leaf:
            results['leaf_characteristics'].append({
                'depth': depth,
                'samples': node.n_samples,
                'model': ""
            })
            return
        
        feat_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
        
        results['split_interpretations'].append({
            'depth': depth,
            'condition': f"{feat_name} <= {node.threshold:.4f}",
            'samples': node.n_samples,
            'explanation': f"Split separates data on {feat_name}"
        })
        
        if node.left:
            self._interpret_node_splits(node.left, results, feature_names, depth + 1)
        if node.right:
            self._interpret_node_splits(node.right, results, feature_names, depth + 1)
    
    def print_analysis_report(self):
        print("\n" + "="*80)
        print(" TREE STRUCTURE ANALYSIS")
        print("="*80)
        
        structure = self.analyze_tree_structure()
        print("\n[1] TREE STRUCTURE")
        print("-" * 80)
        print(f"Total Nodes:              {structure['total_nodes']}")
        print(f"Total Leaves:             {structure['total_leaves']}")
        print(f"Maximum Depth:            {structure['max_depth']}")
        print(f"Avg Samples per Leaf:     {structure['avg_leaf_samples']:.2f}")
        print(f"Leaf Distribution:        {structure['leaf_distribution']}")
        
        print("\n[2] SDR VALIDATION")
        print("-" * 80)
        sdr_results = self.validate_sdr_splits()
        
        if sdr_results['sdr_values']:
            print(f"Number of splits:         {len(sdr_results['sdr_values'])}")
            print(f"Mean SDR:                 {np.mean(sdr_results['sdr_values']):.4f}")
            print(f"Std Dev SDR:              {np.std(sdr_results['sdr_values']):.4f}")
            print(f"Mean Split Quality:       {np.mean(sdr_results['split_quality']):.4f}")
            
            print("\nTop 5 Splits by SDR:")
            top_indices = np.argsort(sdr_results['sdr_values'])[-5:][::-1]
            for i, idx in enumerate(top_indices, 1):
                if idx < len(sdr_results['splits']):
                    split = sdr_results['splits'][idx]
                    print(f"  {i}. Feature {split['feature']}, Threshold {split['threshold']:.4f}, "
                          f"SDR={sdr_results['sdr_values'][idx]:.4f}")
        
        print("\n[3] SPLIT INTERPRETATION")
        print("-" * 80)
        interpretation = self.interpret_splits()
        print(f"Overall Strategy: {interpretation['overall_strategy']}")
        
        if interpretation['split_interpretations']:
            print(f"\nInternal nodes: {len(interpretation['split_interpretations'])}")
            for i, split in enumerate(interpretation['split_interpretations'][:3], 1):
                print(f"  {i}. Depth {split['depth']}: {split['condition']} (n={split['samples']})")
        
        print("\n" + "="*80)


def load_california_housing():
    """Charge le jeu de données California Housing depuis sklearn et renvoie X, y.

    Retourne:
        X: ndarray shape (n_samples, n_features)
        y: ndarray shape (n_samples,)
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y


def visualize_tree_splits(analyzer, X, y, feature_names=None):
    if feature_names is None:
        feature_names = [f"X[{i}]" for i in range(X.shape[1])]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Tree Splits', fontsize=16)
    
    feature_pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    
    for ax, (f1, f2) in zip(axes.flat, feature_pairs):
        scatter = ax.scatter(X[:, f1], X[:, f2], c=y, cmap='viridis', alpha=0.6, s=30)
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(f'{feature_names[f1]} vs {feature_names[f2]}')
        plt.colorbar(scatter, ax=ax)
        
        tree = analyzer.tree
        _draw_splits(ax, tree, f1, f2, X)
    
    plt.tight_layout()
    plt.savefig('tree_splits_visualization.png', dpi=300)
    print("Visualization saved")
    plt.show()


def _draw_splits(ax, node, f1, f2, X, depth=0):
    if node is None or node.is_leaf or depth > 3:
        return
    
    if node.feature == f1:
        ax.axvline(x=node.threshold, color='red', linestyle='--', alpha=0.5)
    elif node.feature == f2:
        ax.axhline(y=node.threshold, color='blue', linestyle='--', alpha=0.5)
    
    if node.left:
        _draw_splits(ax, node.left, f1, f2, X, depth + 1)
    if node.right:
        _draw_splits(ax, node.right, f1, f2, X, depth + 1)


def main():
    print("\n" + "="*80)
    print(" M5P TREE ANALYSIS")
    print("="*80)
    
    print("\n[Step 1] Loading California Housing dataset...")
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    
    print("\n[Step 2] Training M5P model...")
    model = M5PTree(min_samples_split=10, max_depth=5)
    model.fit(X_train, y_train)
    print("  Model trained")
    
    print("\n[Step 3] Analyzing tree...")
    analyzer = TreeStructureAnalyzer(model, X_train, y_train)
    analyzer.print_analysis_report()
    
    print("\n[Step 4] Generating visualizations...")
    feature_names = [f"X[{i}]" for i in range(X_train.shape[1])]
    try:
        visualize_tree_splits(analyzer, X_train, y_train, feature_names)
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
