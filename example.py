"""
Example usage of the M5P Tree implementation.
"""

import numpy as np
from m5p_tree import M5PTree
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def example_synthetic_data():
    """Example 1: Using synthetic regression data."""
    print("=" * 60)
    print("Example 1: Synthetic Regression Data")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=500,
        n_features=5,
        n_informative=3,
        noise=10,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train M5P tree
    m5p = M5PTree(
        min_samples_split=20,
        min_samples_leaf=10,
        max_depth=5,
        use_pruning=True
    )
    
    print(f"\nTraining on {len(X_train)} samples...")
    m5p.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = m5p.predict(X_train)
    y_pred_test = m5p.predict(X_test)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nResults:")
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Test MSE:     {test_mse:.4f}")
    print(f"  Training R²:  {train_r2:.4f}")
    print(f"  Test R²:      {test_r2:.4f}")
    
    print("\nTree Structure:")
    m5p.print_tree()
    print()


def example_simple_function():
    """Example 2: Learning a simple non-linear function."""
    print("=" * 60)
    print("Example 2: Non-linear Function (y = x² + 2x + noise)")
    print("=" * 60)
    
    # Generate data from a quadratic function
    np.random.seed(42)
    X = np.linspace(-5, 5, 200).reshape(-1, 1)
    y = X.ravel() ** 2 + 2 * X.ravel() + np.random.normal(0, 2, 200)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train M5P tree
    m5p = M5PTree(
        min_samples_split=15,
        min_samples_leaf=7,
        max_depth=4,
        use_pruning=True
    )
    
    print(f"\nTraining on {len(X_train)} samples...")
    m5p.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = m5p.score(X_train, y_train)
    test_r2 = m5p.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²:     {test_r2:.4f}")
    
    # Show some predictions
    print(f"\nSample predictions:")
    sample_indices = [0, 10, 20, 30]
    for idx in sample_indices:
        if idx < len(X_test):
            print(f"  X={X_test[idx, 0]:6.2f} -> Predicted: {m5p.predict(X_test[idx:idx+1])[0]:7.2f}, Actual: {y_test[idx]:7.2f}")
    print()


def example_multidimensional():
    """Example 3: Multidimensional data."""
    print("=" * 60)
    print("Example 3: Multidimensional Data")
    print("=" * 60)
    
    # Generate complex data
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 3)
    
    # Complex function: y = 3*x1 + x2² - 2*x3 + interaction terms
    y = (3 * X[:, 0] + 
         X[:, 1] ** 2 - 
         2 * X[:, 2] + 
         0.5 * X[:, 0] * X[:, 1] + 
         np.random.normal(0, 0.5, n_samples))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Train M5P tree
    m5p = M5PTree(
        min_samples_split=10,
        min_samples_leaf=5,
        max_depth=6,
        use_pruning=True
    )
    
    print(f"\nTraining on {len(X_train)} samples with {X_train.shape[1]} features...")
    m5p.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = m5p.predict(X_train)
    y_pred_test = m5p.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nResults:")
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Test MSE:     {test_mse:.4f}")
    print(f"  Training R²:  {train_r2:.4f}")
    print(f"  Test R²:      {test_r2:.4f}")
    print()


def compare_hyperparameters():
    """Example 4: Comparing different hyperparameters."""
    print("=" * 60)
    print("Example 4: Hyperparameter Comparison")
    print("=" * 60)
    
    # Generate data
    X, y = make_regression(
        n_samples=400,
        n_features=4,
        n_informative=3,
        noise=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different configurations
    configs = [
        {"name": "Shallow tree", "max_depth": 3, "min_samples_split": 30},
        {"name": "Deep tree", "max_depth": 10, "min_samples_split": 10},
        {"name": "No pruning", "max_depth": 5, "use_pruning": False},
        {"name": "With pruning", "max_depth": 5, "use_pruning": True},
    ]
    
    print(f"\nTraining {len(configs)} different configurations...\n")
    
    for config in configs:
        name = config.pop("name")
        m5p = M5PTree(**config)
        m5p.fit(X_train, y_train)
        
        test_r2 = m5p.score(X_test, y_test)
        print(f"{name:20s} -> Test R²: {test_r2:.4f}")
    
    print()


if __name__ == "__main__":
    # Run all examples
    example_synthetic_data()
    example_simple_function()
    example_multidimensional()
    compare_hyperparameters()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
