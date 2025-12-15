import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import M5P

np.random.seed(42)

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

configs = [
    {"prune": False, "smoothing": False, "penalty": 2.0, "name": "No pruning, no smoothing"},
    {"prune": True, "smoothing": False, "penalty": 1.0, "name": "Pruning (penalty=1.0)"},
    {"prune": True, "smoothing": False, "penalty": 2.0, "name": "Pruning (penalty=2.0)"},
    {"prune": True, "smoothing": False, "penalty": 3.0, "name": "Pruning (penalty=3.0)"},
    {"prune": False, "smoothing": True, "penalty": 2.0, "name": "Smoothing only"},
    {"prune": True, "smoothing": True, "penalty": 2.0, "name": "Pruning + smoothing (penalty=2.0)"},
]

param_sets = [
    {"min_samples": 15, "depth": 5, "name": "Shallow trees"},
    {"min_samples": 8, "depth": 7, "name": "Medium trees"},
    {"min_samples": 5, "depth": 8, "name": "Deep trees"},
]

for params in param_sets:
    results = []
    
    for config in configs:
        model = M5P(
            min_samples_split=params["min_samples"],
            max_depth=params["depth"],
            prune=config["prune"],
            smoothing=config["smoothing"],
            penalty_factor=config["penalty"]
        )
        
        model.fit(X_train, y_train)
        
        from predict import predict
        y_pred = predict(model, X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        n_nodes = model.tree.count_nodes()
        n_leaves = len(model.tree.get_leaves())
        
        results.append({
            "config": config["name"],
            "mae": mae,
            "rmse": rmse,
            "nodes": n_nodes,
            "leaves": n_leaves
        })
    
    print(f"\nM5P Evaluation - {params['name']} (min_samples={params['min_samples']}, max_depth={params['depth']})")
    print("=" * 100)
    print(f"{'Configuration':<40} {'Nodes':>10} {'Leaves':>10} {'MAE':>15} {'RMSE':>15}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['config']:<40} {r['nodes']:>10} {r['leaves']:>10} {r['mae']:>15.2f} {r['rmse']:>15.2f}")
    
    print("=" * 100)

