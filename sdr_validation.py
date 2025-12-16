import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from tree_builder import M5PTree

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class DetailedSDRValidator:
    def __init__(self, model, X, y):
        self.tree = model.root if hasattr(model, 'root') else model
        self.X = X
        self.y = y
        
    def comprehensive_split_analysis(self):
        analysis = {
            'splits': [],
            'sdr_comparison': [],
            'feature_importance': {},
            'split_statistics': {}
        }
        self._analyze_splits_recursive(self.tree, analysis, np.ones(len(self.y), dtype=bool), depth=0)
        return analysis
    
    def _analyze_splits_recursive(self, node, analysis, mask, depth=0):
        if node is None:
            return
        
        X_node = self.X[mask]
        y_node = self.y[mask]
        
        if node.is_leaf:
            analysis['splits'].append({
                'type': 'leaf',
                'depth': depth,
                'samples': len(y_node),
                'target_mean': np.mean(y_node),
                'target_std': np.std(y_node)
            })
            return
        
        feature_idx = node.feature
        threshold = node.threshold
        
        actual_sdr = self._calculate_sdr_detailed(y_node, X_node[:, feature_idx], threshold)
        alternatives = self._evaluate_alternative_splits(X_node, y_node)
        
        all_sdrs = [actual_sdr] + alternatives['sdr_values']
        rank = len([s for s in all_sdrs if s > actual_sdr]) + 1
        
        split_info = {
            'type': 'internal',
            'depth': depth,
            'feature': feature_idx,
            'threshold': threshold,
            'samples': len(y_node),
            'sdr': actual_sdr,
            'num_alternatives': len(alternatives['sdr_values']),
            'best_alternative_sdr': max(alternatives['sdr_values']) if alternatives['sdr_values'] else 0,
            'rank_among_alternatives': rank,
            'split_quality': actual_sdr / (max(alternatives['sdr_values']) + 1e-10) if alternatives['sdr_values'] else 1.0,
            'left_samples': np.sum(X_node[:, feature_idx] <= threshold),
            'right_samples': np.sum(X_node[:, feature_idx] > threshold)
        }
        
        analysis['splits'].append(split_info)
        
        if feature_idx not in analysis['feature_importance']:
            analysis['feature_importance'][feature_idx] = {'count': 0, 'total_sdr': 0, 'depths': []}
        analysis['feature_importance'][feature_idx]['count'] += 1
        analysis['feature_importance'][feature_idx]['total_sdr'] += actual_sdr
        analysis['feature_importance'][feature_idx]['depths'].append(depth)
        
        analysis['sdr_comparison'].append({
            'depth': depth,
            'feature': feature_idx,
            'actual_sdr': actual_sdr,
            'alternative_sdrs': alternatives['sdr_values']
        })
        
        left_mask = mask & (self.X[:, feature_idx] <= threshold)
        right_mask = mask & (self.X[:, feature_idx] > threshold)
        
        if node.left:
            self._analyze_splits_recursive(node.left, analysis, left_mask, depth + 1)
        if node.right:
            self._analyze_splits_recursive(node.right, analysis, right_mask, depth + 1)
    
    def _calculate_sdr_detailed(self, y, feature_values, threshold):
        parent_std = np.std(y)
        n_total = len(y)
        
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0.0
        
        left_std = np.std(y_left)
        right_std = np.std(y_right)
        
        n_left = len(y_left)
        n_right = len(y_right)
        
        sdr = parent_std - (n_left / n_total) * left_std - (n_right / n_total) * right_std
        
        return max(0.0, sdr)
    
    def _evaluate_alternative_splits(self, X, y):
        alternatives = {'sdr_values': [], 'features': [], 'thresholds': []}
        
        n_features = min(X.shape[1], 5)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) > 1:
                threshold = np.median(feature_values)
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    sdr = self._calculate_sdr_detailed(y, feature_values, threshold)
                    alternatives['sdr_values'].append(sdr)
                    alternatives['features'].append(feature_idx)
                    alternatives['thresholds'].append(threshold)
        
        return alternatives
    
    def print_detailed_report(self):
        analysis = self.comprehensive_split_analysis()
        
        print("\n" + "="*100)
        print(" SDR SPLIT VALIDATION")
        print("="*100)
        
        print("\n[1] OVERALL TREE STATISTICS")
        print("-" * 100)
        
        internal_splits = [s for s in analysis['splits'] if s['type'] == 'internal']
        leaves = [s for s in analysis['splits'] if s['type'] == 'leaf']
        
        print(f"Total nodes:              {len(analysis['splits'])}")
        print(f"Internal splits:          {len(internal_splits)}")
        print(f"Leaf nodes:               {len(leaves)}")
        
        if internal_splits:
            sdrs = [s['sdr'] for s in internal_splits]
            print(f"\nSDR Statistics:")
            print(f"  Mean SDR:               {np.mean(sdrs):.6f}")
            print(f"  Std Dev:                {np.std(sdrs):.6f}")
            print(f"  Min/Max:                {np.min(sdrs):.6f} / {np.max(sdrs):.6f}")
            
            quality_scores = [s['split_quality'] for s in internal_splits]
            print(f"\nSplit Quality:")
            print(f"  Mean Quality:           {np.mean(quality_scores):.4f}")
            print(f"  Min/Max:                {np.min(quality_scores):.4f} / {np.max(quality_scores):.4f}")
            print(f"  % Optimal (≥0.95):      {100 * np.mean(np.array(quality_scores) >= 0.95):.1f}%")
        
        print("\n[2] FEATURE IMPORTANCE")
        print("-" * 100)
        
        if analysis['feature_importance']:
            print(f"{'Feature':<15} {'Count':<10} {'Avg SDR':<15} {'Depths':<30}")
            print("-" * 100)
            
            for feat_idx in sorted(analysis['feature_importance'].keys()):
                info = analysis['feature_importance'][feat_idx]
                avg_sdr = info['total_sdr'] / info['count']
                depths = str(info['depths'])
                print(f"X[{feat_idx}]          {info['count']:<10} {avg_sdr:<15.6f} {depths:<30}")
        
        print("\n[3] SPLIT ANALYSIS")
        print("-" * 100)
        
        for i, split in enumerate(internal_splits):
            print(f"\nSplit #{i+1} (Depth {split['depth']}):")
            print(f"  Feature:                X[{split['feature']}]")
            print(f"  Threshold:              {split['threshold']:.6f}")
            print(f"  Samples:                {split['samples']} (L:{split['left_samples']}, R:{split['right_samples']})")
            print(f"  SDR:                    {split['sdr']:.6f}")
            print(f"  Split Quality:          {split['split_quality']:.4f} {'✓' if split['split_quality'] >= 0.95 else '✗'}")
            print(f"  Rank:                   {split['rank_among_alternatives']}/{split['num_alternatives']+1}")
            
            if split['num_alternatives'] > 0:
                print(f"  Best Alternative:       {split['best_alternative_sdr']:.6f}")
        
        print("\n[4] LEAVES")
        print("-" * 100)
        
        if leaves:
            print(f"{'Leaf #':<10} {'Depth':<10} {'Samples':<10} {'Target Mean':<15} {'Target Std':<15}")
            print("-" * 100)
            
            for i, leaf in enumerate(leaves, 1):
                print(f"{i:<10} {leaf['depth']:<10} {leaf['samples']:<10} "
                      f"{leaf['target_mean']:<15.6f} {leaf['target_std']:<15.6f}")
            
            print(f"\nLeaf Distribution:")
            print(f"  Mean:                   {np.mean([l['samples'] for l in leaves]):.2f}")
            print(f"  Std Dev:                {np.std([l['samples'] for l in leaves]):.2f}")
            print(f"  Imbalance Ratio:        {max([l['samples'] for l in leaves]) / (min([l['samples'] for l in leaves]) + 1e-10):.2f}x")
        
        print("\n[5] INTERPRETATION")
        print("-" * 100)
        
        if internal_splits:
            avg_quality = np.mean([s['split_quality'] for s in internal_splits])
            
            if avg_quality >= 0.95:
                quality_assessment = "Excellent"
            elif avg_quality >= 0.90:
                quality_assessment = "Good"
            elif avg_quality >= 0.85:
                quality_assessment = "Fair"
            else:
                quality_assessment = "Poor"
            
            print(f"\nOverall Split Quality: {quality_assessment}")
            print(f"Average Quality Score: {avg_quality:.4f}")
            
            if analysis['feature_importance']:
                most_important = max(analysis['feature_importance'].items(), 
                                    key=lambda x: x[1]['count'])
                print(f"\nMost Important Feature: X[{most_important[0]}] "
                      f"(used {most_important[1]['count']} times)")
        
        print("\n" + "="*100)
    
    def compare_with_alternatives(self):
        analysis = self.comprehensive_split_analysis()
        
        internal_splits = [s for s in analysis['splits'] if s['type'] == 'internal']
        
        if not internal_splits:
            return
        
        actual_sdrs = np.array([s['sdr'] for s in internal_splits])
        best_alternatives = np.array([s['best_alternative_sdr'] for s in internal_splits])
        
        improvement = ((actual_sdrs - best_alternatives) / (best_alternatives + 1e-10)) * 100
        
        print("\n[6] COMPARISON WITH ALTERNATIVES")
        print("-" * 100)
        print(f"{'Metric':<40} {'Value':<15} {'Interpretation'}")
        print("-" * 100)
        print(f"Mean Improvement over Best Alternative {np.mean(improvement):>13.2f}% {'Significantly better' if np.mean(improvement) > 5 else 'Slightly better'}")
        print(f"Median Improvement                      {np.median(improvement):>13.2f}%")
        print(f"% Splits Better than Alternatives       {100 * np.mean(actual_sdrs > best_alternatives):>13.1f}%")


def validate_on_diabetes():
    print("\n" + "="*100)
    print(" VALIDATION ON DIABETES DATASET")
    print("="*100)
    
    data = load_diabetes()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = M5PTree(min_samples_split=8, max_depth=5)
    model.fit(X_train, y_train)
    
    validator = DetailedSDRValidator(model, X_train, y_train)
    validator.print_detailed_report()
    validator.compare_with_alternatives()


def validate_on_synthetic():
    print("\n" + "="*100)
    print(" VALIDATION ON SYNTHETIC DATASET")
    print("="*100)
    
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (500, 4))
    y = np.zeros(500)
    
    mask1 = X[:, 0] <= 0
    y[mask1] = 3 * X[mask1, 0] + 2 * X[mask1, 1] - X[mask1, 2] + 10
    
    mask2 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    y[mask2] = -2 * X[mask2, 0] + 4 * X[mask2, 2] + X[mask2, 3] - 5
    
    mask3 = (X[:, 0] > 0) & (X[:, 1] > 0)
    y[mask3] = X[mask3, 0] + X[mask3, 1] + 2 * X[mask3, 2] - X[mask3, 3]
    
    y += np.random.normal(0, 0.3, 500)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = M5PTree(min_samples_split=10, max_depth=5)
    model.fit(X_train, y_train)
    
    validator = DetailedSDRValidator(model, X_train, y_train)
    validator.print_detailed_report()
    validator.compare_with_alternatives()


if __name__ == "__main__":
    print("\n" + "="*100)
    print(" SDR VALIDATION")
    print("="*100)
    
    try:
        validate_on_synthetic()
    except Exception as e:
        print(f"Error in synthetic validation: {e}")
    
    try:
        validate_on_diabetes()
    except Exception as e:
        print(f"Error in diabetes validation: {e}")
    
    print("\n" + "="*100)
    print(" VALIDATION COMPLETE")
    print("="*100)
