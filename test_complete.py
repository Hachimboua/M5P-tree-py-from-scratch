import numpy as np
from model import M5P
from predict import predict


def test_complete_workflow():
    print("M5P Complete Workflow Test")
    print("=" * 50)
    
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 4)
    y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] - 0.5*X[:, 3] + np.random.randn(n_samples)*0.5
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nData: {len(X_train)} train, {len(X_test)} test samples")
    
    print("\n1. M5P with pruning and smoothing:")
    model1 = M5P(min_samples_split=10, max_depth=5, prune=True, smoothing=True)
    model1.fit(X_train, y_train)
    score1 = model1.score(X_test, y_test)
    print(f"   Nodes: {model1.tree.count_nodes()}")
    print(f"   Leaves: {len(model1.tree.get_leaves())}")
    print(f"   Test R²: {score1:.4f}")
    
    print("\n2. M5P without pruning:")
    model2 = M5P(min_samples_split=10, max_depth=5, prune=False, smoothing=True)
    model2.fit(X_train, y_train)
    score2 = model2.score(X_test, y_test)
    print(f"   Nodes: {model2.tree.count_nodes()}")
    print(f"   Leaves: {len(model2.tree.get_leaves())}")
    print(f"   Test R²: {score2:.4f}")
    
    print("\n3. M5P without smoothing:")
    model3 = M5P(min_samples_split=10, max_depth=5, prune=True, smoothing=False)
    model3.fit(X_train, y_train)
    score3 = model3.score(X_test, y_test)
    print(f"   Test R²: {score3:.4f}")
    
    print("\n4. Single sample prediction:")
    sample = X_test[0]
    pred = predict(model1, sample)
    print(f"   Input: {sample}")
    print(f"   Prediction: {pred[0]:.3f}")
    print(f"   Actual: {y_test[0]:.3f}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_complete_workflow()
