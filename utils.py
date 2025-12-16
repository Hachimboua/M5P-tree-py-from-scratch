import numpy as np
from typing import Tuple

def standard_deviation_reduction(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Calculate Standard Deviation Reduction (SDR) for a split.
    
    SDR = SD(parent) - weighted_average[SD(left), SD(right)]
    
    This is M5P's splitting criterion - higher SDR = better split.
    """
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0
    
    sd_parent = np.std(y)
    sd_left = np.std(y_left)
    sd_right = np.std(y_right)
    
    weighted_sd = (n_left * sd_left + n_right * sd_right) / n
    
    return sd_parent - weighted_sd

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error - penalizes large errors more heavily.
    
    MSE = (1/n) Σ(y_true - y_pred)²
    """
    return float(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error - robust to outliers.
    
    MAE = (1/n) Σ|y_true - y_pred|
    """
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² coefficient of determination.
    
    R² = 1 - (SS_residual / SS_total)
    
    Interpretation:
    - R² = 1: perfect predictions
    - R² = 0: model = mean baseline
    - R² < 0: model worse than mean
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Returns: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance.
    
    Z-score normalization: X_norm = (X - μ) / σ
    
    Returns: X_normalized, mean, std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero for constant features
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std
