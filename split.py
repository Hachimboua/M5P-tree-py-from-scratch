import numpy as np

def std_deviation(values):
    """
    Compute standard deviation of values.
    
    Returns 0 for single value (no variance to reduce).
    """
    if len(values) <= 1:
        return 0.0
    return np.std(values)

def calculate_sdr(parent_vals, left_vals, right_vals):
    """
    Calculate Standard Deviation Reduction (SDR).
    
    SDR = SD(parent) - [w_left * SD(left) + w_right * SD(right)]
    
    Why SDR instead of MSE?
    - SDR measures variance reduction directly
    - Equivalent to MSE minimization but more intuitive
    - Higher SDR = better split (more homogeneous children)
    
    SDR > 0: split reduces target variance (good)
    SDR = 0: split provides no benefit
    SDR < 0: should not occur with proper splitting
    """
    if len(parent_vals) == 0:
        return 0.0
    
    parent_std = std_deviation(parent_vals)
    left_std = std_deviation(left_vals)
    right_std = std_deviation(right_vals)
    
    # Weighted average of child standard deviations
    n_total = len(parent_vals)
    n_left = len(left_vals)
    n_right = len(right_vals)
    
    weighted_std = (n_left/n_total) * left_std + (n_right/n_total) * right_std
    
    # Reduction in standard deviation
    return parent_std - weighted_std

def find_split_for_feature(X, y, feature_idx, min_samples=2):
    """
    Find best split point for a single feature using SDR criterion.
    
    Strategy:
    - Test midpoints between consecutive unique values
    - Ensures min_samples constraint on both sides
    - Returns split with highest SDR
    """
    feature_vals = X[:, feature_idx]
    unique_vals = np.unique(feature_vals)
    
    # Cannot split if all values are identical
    if len(unique_vals) < 2:
        return None
    
    best_sdr = -np.inf
    best_threshold = None
    best_left_idx = None
    best_right_idx = None
    
    # Test all possible split points (midpoints between unique values)
    for i in range(len(unique_vals) - 1):
        threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0
        
        left_mask = feature_vals <= threshold
        right_mask = ~left_mask
        
        # Enforce minimum samples per child (prevents overfitting)
        if np.sum(left_mask) < min_samples or np.sum(right_mask) < min_samples:
            continue
        
        # Compute SDR for this split
        sdr = calculate_sdr(y, y[left_mask], y[right_mask])
        
        # Track best split
        if sdr > best_sdr:
            best_sdr = sdr
            best_threshold = threshold
            best_left_idx = left_mask
            best_right_idx = right_mask
    
    # No valid split found
    if best_threshold is None:
        return None
    
    result = {
        'feature': feature_idx,
        'threshold': best_threshold,
        'sdr': best_sdr,
        'left_mask': best_left_idx,
        'right_mask': best_right_idx
    }
    
    return result

def find_best_split(X, y, min_samples=2):
    """
    Find globally best split across all features.
    
    Greedy approach:
    - Test each feature independently
    - Select feature and threshold with highest SDR
    - This is the standard decision tree splitting strategy
    """
    n_samples, n_features = X.shape
    
    if n_samples < min_samples:
        return None
    
    best_split = None
    best_sdr = -np.inf
    
    # Evaluate all features
    for feat_idx in range(n_features):
        split = find_split_for_feature(X, y, feat_idx, min_samples)
        
        # Update best split if this feature is better
        if split and split['sdr'] > best_sdr:
            best_sdr = split['sdr']
            best_split = split
    
    return best_split
