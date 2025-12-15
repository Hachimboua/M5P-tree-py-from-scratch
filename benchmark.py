import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from model import M5P
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
        'R2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }


def print_results_table(results, title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


def plot_metrics_comparison(results, title, filename):
    models = list(results.keys())
    mae_values = [results[m]['MAE'] for m in models]
    rmse_values = [results[m]['RMSE'] for m in models]
    r2_values = [results[m]['R2'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    bars1 = axes[0].bar(models, mae_values, color=colors[:len(models)], edgecolor='black')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error (lower=better)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mae_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    bars2 = axes[1].bar(models, rmse_values, color=colors[:len(models)], edgecolor='black')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error (lower=better)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    bars3 = axes[2].bar(models, r2_values, color=colors[:len(models)], edgecolor='black')
    axes[2].set_ylabel('R2 Score', fontsize=12)
    axes[2].set_title('R2 Score (higher=better)', fontsize=11)
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, r2_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_predictions_scatter(y_test, results, title, filename):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        
        axes[i].scatter(y_test, y_pred, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolor='black', linewidth=0.3)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        axes[i].set_xlabel('Actual', fontsize=11)
        axes[i].set_ylabel('Predicted', fontsize=11)
        axes[i].set_title(f'{model_name}\nR2={metrics["R2"]:.3f}', fontsize=10)
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_residuals(y_test, results, title, filename):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        residuals = y_test - y_pred
        
        axes[i].scatter(y_pred, residuals, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolor='black', linewidth=0.3)
        axes[i].axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Residuals', fontsize=11)
        axes[i].set_title(f'{model_name}\nMAE={metrics["MAE"]:.2f}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_ablation_heatmap(results, filename):
    configs = list(results.keys())
    metrics_names = ['MAE', 'RMSE', 'R2']
    
    data = np.array([[results[c][m] for m in metrics_names] for c in configs])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_yticklabels(configs, fontsize=10)
    
    for i in range(len(configs)):
        for j in range(len(metrics_names)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    
    ax.set_title('Ablation Study: Pruning & Smoothing Effects', fontsize=14, fontweight='bold')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_final_summary(results_diabetes, results_synthetic, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('M5P Model - Performance Summary', fontsize=16, fontweight='bold')
    
    models_d = list(results_diabetes.keys())
    r2_d = [results_diabetes[m]['R2'] for m in models_d]
    colors_d = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars1 = axes[0].bar(models_d, r2_d, color=colors_d[:len(models_d)], edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R2 Score', fontsize=12)
    axes[0].set_title('Diabetes Dataset', fontsize=13)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, r2_d):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        axes[0].text(bar.get_x() + bar.get_width()/2, ypos, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    models_s = list(results_synthetic.keys())
    r2_s = [results_synthetic[m]['R2'] for m in models_s]
    colors_s = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
    
    bars2 = axes[1].bar(models_s, r2_s, color=colors_s[:len(models_s)], edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('R2 Score', fontsize=12)
    axes[1].set_title('Synthetic Piecewise-Linear Dataset', fontsize=13)
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars2, r2_s):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


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
        'M5P (ours)': M5P(min_samples_split=20, max_depth=4, prune=True, smoothing=True, penalty_factor=2.0)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Diabetes Dataset Results")
    
    plot_metrics_comparison(results, "Diabetes Dataset - Model Comparison", "diabetes_metrics.png")
    plot_predictions_scatter(y_test, results, "Diabetes - Predictions vs Actual", "diabetes_scatter.png")
    plot_residuals(y_test, results, "Diabetes - Residuals Analysis", "diabetes_residuals.png")
    
    return results, y_test


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
        'M5P (ours)': M5P(min_samples_split=15, max_depth=5, prune=True, smoothing=True, penalty_factor=2.0)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Synthetic Piecewise-Linear Results")
    
    plot_metrics_comparison(results, "Synthetic Data - Model Comparison", "synthetic_metrics.png")
    plot_predictions_scatter(y_test, results, "Synthetic - Predictions vs Actual", "synthetic_scatter.png")
    plot_residuals(y_test, results, "Synthetic - Residuals Analysis", "synthetic_residuals.png")
    
    return results, y_test


def benchmark_pruning_smoothing():
    print("\n" + "#"*60)
    print(" BENCHMARK 3: Pruning & Smoothing Analysis")
    print("#"*60)
    
    X, y = generate_piecewise_linear_data(n_samples=500, noise=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    
    configs = {
        'No Prune, No Smooth': M5P(min_samples_split=15, max_depth=5, prune=False, smoothing=False),
        'Prune, No Smooth': M5P(min_samples_split=15, max_depth=5, prune=True, smoothing=False, penalty_factor=2.0),
        'No Prune, Smooth': M5P(min_samples_split=15, max_depth=5, prune=False, smoothing=True),
        'Prune + Smooth': M5P(min_samples_split=15, max_depth=5, prune=True, smoothing=True, penalty_factor=2.0)
    }
    
    results = {}
    for name, model in configs.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Pruning & Smoothing Comparison")
    
    plot_metrics_comparison(results, "Ablation Study - Pruning & Smoothing", "ablation_metrics.png")
    plot_ablation_heatmap(results, "ablation_heatmap.png")
    plot_predictions_scatter(y_test, results, "Ablation - Predictions vs Actual", "ablation_scatter.png")
    
    return results, y_test


def main():
    print("\n" + "="*60)
    print(" M5P MODEL TREE - COMPREHENSIVE BENCHMARK")
    print(" Member 3 - ENSAM Project")
    print("="*60)
    
    results_diabetes, y_test_d = benchmark_diabetes()
    results_synthetic, y_test_s = benchmark_synthetic()
    results_ablation, y_test_a = benchmark_pruning_smoothing()
    
    plot_final_summary(results_diabetes, results_synthetic, "final_summary.png")
    
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    print("\n[Diabetes Dataset]")
    best_d = max(results_diabetes.keys(), key=lambda x: results_diabetes[x]['R2'])
    print(f"  Best model: {best_d} (R2 = {results_diabetes[best_d]['R2']:.4f})")
    
    print("\n[Synthetic Piecewise-Linear]")
    best_s = max(results_synthetic.keys(), key=lambda x: results_synthetic[x]['R2'])
    print(f"  Best model: {best_s} (R2 = {results_synthetic[best_s]['R2']:.4f})")
    
    print("\n[Ablation Study]")
    best_a = max(results_ablation.keys(), key=lambda x: results_ablation[x]['R2'])
    print(f"  Best config: {best_a} (R2 = {results_ablation[best_a]['R2']:.4f})")
    
    print("\n" + "="*60)
    print(" PLOTS GENERATED:")
    print("="*60)
    print("  - diabetes_metrics.png")
    print("  - diabetes_scatter.png")
    print("  - diabetes_residuals.png")
    print("  - synthetic_metrics.png")
    print("  - synthetic_scatter.png")
    print("  - synthetic_residuals.png")
    print("  - ablation_metrics.png")
    print("  - ablation_heatmap.png")
    print("  - ablation_scatter.png")
    print("  - final_summary.png")
    
    print("\n" + "="*60)
    print(" BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
