"""
Utility Functions for M5P Implementation
"""

import numpy as np
from typing import Tuple


def standard_deviation_reduction(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
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
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    
    X_normalized = (X - mean) / std
    
    return X_normalized, mean, std
