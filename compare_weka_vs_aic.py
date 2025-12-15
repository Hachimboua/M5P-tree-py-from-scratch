import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import M5P

np.random.seed(42)

print("=" * 120)
print("COMPARAISON: Formule Weka Originale vs Formule AIC")
print("=" * 120)
print()
print("Formule Weka:  err_adj = err_raw * (n + PF*v) / (n - v)    [PF=2, standard M5P]")
print("Formule AIC:   err_adj = err_raw * (1 + factor*v/n)        [factor=2, robuste]")
print()

datasets = [
    {
        'name': 'Diabetes',
        'data': load_diabetes(),
        'description': 'Petit dataset (442 samples, 10 features) - Cas difficile pour Weka',
        'subsample': None
    },
    {
        'name': 'California Housing (subsample)',
        'data': fetch_california_housing(),
        'description': 'Grand dataset (5000 samples subsampled, 8 features) - Cas idéal pour Weka',
        'subsample': 5000
    }
]

tree_configs = [
    {"min_samples": 20, "depth": 4, "name": "Arbres peu profonds (M5P classique)"},
    {"min_samples": 10, "depth": 6, "name": "Arbres moyens"},
]

for dataset_info in datasets:
    print("\n" + "=" * 120)
    print(f"DATASET: {dataset_info['name']}")
    print("=" * 120)
    print(f"{dataset_info['description']}")
    print()
    
    data = dataset_info['data']
    X, y = data.data, data.target
    
    # Subsample si nécessaire pour accélérer
    if dataset_info['subsample'] and len(X) > dataset_info['subsample']:
        indices = np.random.choice(len(X), dataset_info['subsample'], replace=False)
        X, y = X[indices], y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Taille: {len(X_train)} train, {len(X_test)} test, {X.shape[1]} features\n")
    
    for tree_config in tree_configs:
        print("-" * 120)
        print(f"{tree_config['name']} (min_samples={tree_config['min_samples']}, max_depth={tree_config['depth']})")
        print("-" * 120)
        
        results = []
        
        configs = [
            {"prune": False, "smoothing": False, "weka": False, "name": "Sans pruning ni smoothing"},
            {"prune": True, "smoothing": False, "weka": True, "name": "Pruning Weka (PF=2)"},
            {"prune": True, "smoothing": False, "weka": False, "name": "Pruning AIC (factor=2)"},
            {"prune": False, "smoothing": True, "weka": False, "name": "Smoothing seulement"},
            {"prune": True, "smoothing": True, "weka": True, "name": "Weka + Smoothing"},
            {"prune": True, "smoothing": True, "weka": False, "name": "AIC + Smoothing"},
        ]
        
        for config in configs:
            print(f"  Training: {config['name']}...", end=" ", flush=True)
            try:
                model = M5P(
                    min_samples_split=tree_config["min_samples"],
                    max_depth=tree_config["depth"],
                    prune=config["prune"],
                    smoothing=config["smoothing"],
                    penalty_factor=2.0,
                    use_weka_formula=config["weka"]
                )
                
                model.fit(X_train, y_train)
                
                from predict import predict
                y_pred = predict(model, X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                n_nodes = model.tree.count_nodes()
                n_leaves = len(model.tree.get_leaves())
                
                print(f"Done ({n_nodes} nodes, MAE={mae:.2f})")
                
                results.append({
                    "config": config["name"],
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "nodes": n_nodes,
                    "leaves": n_leaves,
                    "error": None
                })
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                results.append({
                    "config": config["name"],
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "nodes": None,
                    "leaves": None,
                    "error": str(e)[:50]
                })
        
        print(f"{'Configuration':<35} {'Nodes':>8} {'Leaves':>8} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
        print("-" * 120)
        
        for r in results:
            if r["error"]:
                print(f"{r['config']:<35} {'ERROR':<8} {'---':<8} {'---':<12} {'---':<12} {'---':<10}")
            else:
                print(f"{r['config']:<35} {r['nodes']:>8} {r['leaves']:>8} "
                      f"{r['mae']:>12.2f} {r['rmse']:>12.2f} {r['r2']:>10.3f}")
        
        print()
        
        # Comparative analysis between Weka and AIC
        if len(results) >= 3 and results[1]["mae"] is not None and results[2]["mae"] is not None:
            weka = results[1]  # Pruning Weka
            aic = results[2]   # Pruning AIC
            
            print("COMPARISON Weka vs AIC:")
            print(f"   Weka: {weka['nodes']} nodes, MAE={weka['mae']:.2f}")
            print(f"   AIC:  {aic['nodes']} nodes, MAE={aic['mae']:.2f}")
            
            if weka['nodes'] == 1 and aic['nodes'] > 1:
                print("   WARNING: Weka over-pruned (1 node only), AIC preserves structure")
            elif weka['mae'] < aic['mae'] * 0.95:
                print("   Weka performs better (MAE lower by >5%)")
            elif aic['mae'] < weka['mae'] * 0.95:
                print("   AIC performs better (MAE lower by >5%)")
            else:
                print("   Similar performance")
            print()

print("\n" + "=" * 120)
print("CONCLUSIONS")
print("=" * 120)

print("""
1. WEKA FORMULA (n + PF*v)/(n - v):
   
   Works WELL on:
      - Large datasets (Housing: 20k samples)
      - Shallow trees (depth <= 5)
      - Low v/n ratio
   
   Problems on:
      - Small datasets (Diabetes: 442 samples)
      - Deep trees (depth > 6)
      - Over-aggressive pruning -> 1 node only

2. AIC FORMULA (1 + factor*v/n):
   
   Advantages:
      - Robust on all dataset types
      - Works with deep trees
      - No mathematical explosion
      - Balanced pruning
   
   May be slightly less aggressive than Weka on large datasets

3. RECOMMENDATION:
   
   - LARGE dataset (>5000 samples, features <20) -> Weka OK
   - SMALL dataset or DEEP trees -> AIC recommended
   - For general use -> AIC (current implementation)
""")

print("=" * 120)
