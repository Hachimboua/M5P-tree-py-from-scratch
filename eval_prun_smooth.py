import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import M5P

# Fix random seed for reproducibility across runs
np.random.seed(42)

# Load the diabetes dataset from sklearn
# This is a standard regression dataset with 10 features and continuous target
data = load_diabetes()
X, y = data.data, data.target

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define experimental configurations to test pruning and smoothing effects
# Each configuration tests a different combination of pruning/smoothing with varying penalties
configs = [
    {"prune": False, "smoothing": False, "penalty": 2.0, "name": "No pruning, no smoothing"},
    {"prune": True, "smoothing": False, "penalty": 1.0, "name": "Pruning (penalty=1.0)"},
    {"prune": True, "smoothing": False, "penalty": 2.0, "name": "Pruning (penalty=2.0)"},
    {"prune": True, "smoothing": False, "penalty": 3.0, "name": "Pruning (penalty=3.0)"},
    {"prune": False, "smoothing": True, "penalty": 2.0, "name": "Smoothing only"},
    {"prune": True, "smoothing": True, "penalty": 2.0, "name": "Pruning + smoothing (penalty=2.0)"},
]

# Define tree complexity parameter sets
# Tests how tree depth and minimum samples affect model performance
param_sets = [
    {"min_samples": 15, "depth": 5, "name": "Shallow trees"},
    {"min_samples": 8, "depth": 7, "name": "Medium trees"},
    {"min_samples": 5, "depth": 8, "name": "Deep trees"},
]

# Run benchmarking experiments for each parameter configuration
for params in param_sets:
    results = []
    
    # Test each pruning/smoothing configuration
    for config in configs:
        # Initialize M5P model with current parameters
        model = M5P(
            min_samples_split=params["min_samples"],
            max_depth=params["depth"],
            prune=config["prune"],
            smoothing=config["smoothing"],
            penalty_factor=config["penalty"]
        )
        
        # Train the model on training data
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        from predict import predict
        y_pred = predict(model, X_test)
        
        # Compute error metrics
        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
        
        # Extract tree statistics
        n_nodes = model.tree.count_nodes()  # Total number of nodes (internal + leaves)
        n_leaves = len(model.tree.get_leaves())  # Number of leaf nodes (linear models)
        
        # Store results for this configuration
        results.append({
            "config": config["name"],
            "mae": mae,
            "rmse": rmse,
            "nodes": n_nodes,
            "leaves": n_leaves
        })
    
    # Display results table for current parameter set
    print(f"\nM5P Evaluation - {params['name']} (min_samples={params['min_samples']}, max_depth={params['depth']})")
    print("=" * 100)
    print(f"{'Configuration':<40} {'Nodes':>10} {'Leaves':>10} {'MAE':>15} {'RMSE':>15}")
    print("-" * 100)
    
    # Print each configuration's results
    for r in results:
        print(f"{r['config']:<40} {r['nodes']:>10} {r['leaves']:>10} {r['mae']:>15.2f} {r['rmse']:>15.2f}")
    
    print("=" * 100)
