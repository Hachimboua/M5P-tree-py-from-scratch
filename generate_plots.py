import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import M5P
from predict import predict

np.random.seed(42)

datasets = [
    {
        'name': 'Diabetes',
        'data': load_diabetes(),
        'subsample': None,
        'color': '#FF6B6B'
    },
    {
        'name': 'California Housing',
        'data': fetch_california_housing(),
        'subsample': 5000,
        'color': '#4ECDC4'
    }
]

tree_configs = [
    {"min_samples": 20, "depth": 4, "label": "Shallow (depth=4)"},
    {"min_samples": 10, "depth": 6, "label": "Medium (depth=6)"},
]

pruning_methods = [
    {"name": "No Pruning", "prune": False, "smoothing": False, "weka": False, "marker": 'o'},
    {"name": "Weka Formula", "prune": True, "smoothing": False, "weka": True, "marker": 's'},
    {"name": "AIC Formula", "prune": True, "smoothing": False, "weka": False, "marker": '^'},
    {"name": "Weka + Smoothing", "prune": True, "smoothing": True, "weka": True, "marker": 'D'},
    {"name": "AIC + Smoothing", "prune": True, "smoothing": True, "weka": False, "marker": 'v'},
]

results = {ds['name']: {tc['label']: [] for tc in tree_configs} for ds in datasets}

print("Collecting data for plots...")

for dataset_info in datasets:
    print(f"\nProcessing {dataset_info['name']}...")
    data = dataset_info['data']
    X, y = data.data, data.target
    
    if dataset_info['subsample'] and len(X) > dataset_info['subsample']:
        indices = np.random.choice(len(X), dataset_info['subsample'], replace=False)
        X, y = X[indices], y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    for tree_config in tree_configs:
        print(f"  {tree_config['label']}...")
        
        for method in pruning_methods:
            model = M5P(
                min_samples_split=tree_config["min_samples"],
                max_depth=tree_config["depth"],
                prune=method["prune"],
                smoothing=method["smoothing"],
                penalty_factor=2.0,
                use_weka_formula=method["weka"]
            )
            
            model.fit(X_train, y_train)
            y_pred = predict(model, X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            n_nodes = model.tree.count_nodes()
            n_leaves = len(model.tree.get_leaves())
            
            results[dataset_info['name']][tree_config['label']].append({
                'method': method['name'],
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'nodes': n_nodes,
                'leaves': n_leaves,
                'marker': method['marker']
            })

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('M5P Pruning Methods Comparison: Weka vs AIC', fontsize=16, fontweight='bold')

for dataset_idx, dataset_info in enumerate(datasets):
    dataset_name = dataset_info['name']
    color = dataset_info['color']
    
    for depth_idx, tree_config in enumerate(tree_configs):
        label = tree_config['label']
        data = results[dataset_name][label]
        
        methods = [d['method'] for d in data]
        nodes = [d['nodes'] for d in data]
        mae_values = [d['mae'] for d in data]
        r2_values = [d['r2'] for d in data]
        
        ax_nodes = axes[dataset_idx][0]
        ax_mae = axes[dataset_idx][1]
        ax_r2 = axes[dataset_idx][2]
        
        x_pos = np.arange(len(methods))
        bar_width = 0.35
        offset = bar_width * depth_idx
        
        ax_nodes.bar(x_pos + offset, nodes, bar_width, label=label, alpha=0.8)
        ax_mae.bar(x_pos + offset, mae_values, bar_width, label=label, alpha=0.8)
        ax_r2.bar(x_pos + offset, r2_values, bar_width, label=label, alpha=0.8)
    
    axes[dataset_idx][0].set_title(f'{dataset_name} - Tree Size (Nodes)', fontsize=12, fontweight='bold')
    axes[dataset_idx][0].set_ylabel('Number of Nodes', fontsize=10)
    axes[dataset_idx][0].set_xticks(np.arange(len(methods)) + bar_width/2)
    axes[dataset_idx][0].set_xticklabels(methods, rotation=45, ha='right')
    axes[dataset_idx][0].legend()
    axes[dataset_idx][0].grid(axis='y', alpha=0.3)
    
    axes[dataset_idx][1].set_title(f'{dataset_name} - Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[dataset_idx][1].set_ylabel('MAE', fontsize=10)
    axes[dataset_idx][1].set_xticks(np.arange(len(methods)) + bar_width/2)
    axes[dataset_idx][1].set_xticklabels(methods, rotation=45, ha='right')
    axes[dataset_idx][1].legend()
    axes[dataset_idx][1].grid(axis='y', alpha=0.3)
    
    axes[dataset_idx][2].set_title(f'{dataset_name} - R² Score', fontsize=12, fontweight='bold')
    axes[dataset_idx][2].set_ylabel('R² Score', fontsize=10)
    axes[dataset_idx][2].set_xticks(np.arange(len(methods)) + bar_width/2)
    axes[dataset_idx][2].set_xticklabels(methods, rotation=45, ha='right')
    axes[dataset_idx][2].legend()
    axes[dataset_idx][2].grid(axis='y', alpha=0.3)
    axes[dataset_idx][2].axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pruning_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: pruning_comparison.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Nodes vs MAE: Weka Formula vs AIC Formula', fontsize=16, fontweight='bold')

for dataset_idx, dataset_info in enumerate(datasets):
    dataset_name = dataset_info['name']
    
    for depth_idx, tree_config in enumerate(tree_configs):
        label = tree_config['label']
        data = results[dataset_name][label]
        
        ax = axes2[dataset_idx][depth_idx]
        
        for d in data:
            ax.scatter(d['nodes'], d['mae'], s=150, marker=d['marker'], 
                      label=d['method'], alpha=0.7)
        
        ax.set_xlabel('Number of Nodes', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.set_title(f'{dataset_name} - {label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nodes_vs_mae.png', dpi=300, bbox_inches='tight')
print("Saved: nodes_vs_mae.png")

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

n_values = np.arange(10, 101, 5)
p_value = 11

weka_penalty = [(n + 2*p_value) / (n - p_value) if n > p_value else np.inf for n in n_values]
aic_penalty = [1 + 2*p_value/n for n in n_values]

ax3.plot(n_values, weka_penalty, 'r-', linewidth=2, label='Weka: (n + 2v)/(n - v)', marker='o')
ax3.plot(n_values, aic_penalty, 'b-', linewidth=2, label='AIC: 1 + 2v/n', marker='s')
ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Threshold=3')
ax3.set_xlabel('Number of Samples (n)', fontsize=12)
ax3.set_ylabel('Penalty Factor', fontsize=12)
ax3.set_title('Penalty Growth: Weka vs AIC (v=11 parameters)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 10])

plt.tight_layout()
plt.savefig('penalty_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: penalty_comparison.png")

print("\nAll plots generated successfully!")
