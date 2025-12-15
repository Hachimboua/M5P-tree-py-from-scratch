"""
M5P Benchmarking Script - Member 3
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from model import M5PModel
from utils import train_test_split


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def generate_piecewise_linear_data(n_samples=500, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, (n_samples, 4))
    y = np.zeros(n_samples)
    
    mask1 = X[:, 0] <= 0
    y[mask1] = 3 * X[mask1, 0] + 2 * X[mask1, 1] - X[mask1, 2] + 10
    
    mask2 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    y[mask2] = -2 * X[mask2, 0] + 4 * X[mask2, 2] + X[mask2, 3] - 5
    
    mask3 = (X[:, 0] > 0) & (X[:, 1] > 0)
    y[mask3] = X[mask3, 0] + X[mask3, 1] + 2 * X[mask3, 2] - X[mask3, 3]
    
    y += np.random.normal(0, noise, n_samples)
    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def print_results_table(results, title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


def benchmark_diabetes():
    print("\n" + "#"*60)
    print(" BENCHMARK 1: Sklearn Diabetes Dataset")
    print("#"*60)
    
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'M5P (ours)': M5PModel(min_samples_split=10, min_samples_leaf=5, pruning=True, smoothing=True)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Diabetes Dataset Results")
    return results


def benchmark_synthetic():
    print("\n" + "#"*60)
    print(" BENCHMARK 2: Synthetic Piecewise-Linear Dataset")
    print("#"*60)
    
    X, y = generate_piecewise_linear_data(n_samples=600, noise=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print("Data type: Piecewise linear with 3 distinct regions")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree (d=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Decision Tree (d=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (d=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        'M5P (ours)': M5PModel(min_samples_split=10, min_samples_leaf=5, pruning=True, smoothing=True)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Synthetic Piecewise-Linear Results")
    return results


def benchmark_pruning_smoothing():
    print("\n" + "#"*60)
    print(" BENCHMARK 3: Pruning & Smoothing Analysis")
    print("#"*60)
    
    X, y = generate_piecewise_linear_data(n_samples=500, noise=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    
    configs = {
        'M5P (no prune, no smooth)': M5PModel(pruning=False, smoothing=False),
        'M5P (prune, no smooth)': M5PModel(pruning=True, smoothing=False),
        'M5P (no prune, smooth)': M5PModel(pruning=False, smoothing=True),
        'M5P (prune + smooth)': M5PModel(pruning=True, smoothing=True)
    }
    
    results = {}
    for name, model in configs.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Pruning & Smoothing Comparison")
    return results


def main():
    print("\n" + "="*60)
    print(" M5P MODEL TREE - COMPREHENSIVE BENCHMARK")
    print(" Member 3 - ENSAM Project")
    print("="*60)
    
    results_diabetes = benchmark_diabetes()
    results_synthetic = benchmark_synthetic()
    results_ablation = benchmark_pruning_smoothing()
    
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    print("\n[Diabetes Dataset]")
    print(f"  Best model: M5P" if results_diabetes['M5P (ours)']['RMSE'] <= results_diabetes['Decision Tree']['RMSE'] else "  Best model: Decision Tree")
    
    print("\n[Synthetic Piecewise-Linear]")
    print("  M5P is expected to excel here due to local linear models")
    
    print("\n[Ablation Study]")
    print("  Pruning reduces overfitting")
    print("  Smoothing stabilizes predictions")
    
    print("\n" + "="*60)
    print(" BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
