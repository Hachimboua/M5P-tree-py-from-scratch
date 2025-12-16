"""
M5P Model Tree - Comprehensive Benchmark Script
ENSAM Project - Member 3

Compares M5P performance against baseline models on real-world datasets.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from model import M5P
from utils import train_test_split


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error - average prediction deviation."""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """Root Mean Squared Error - penalizes large errors more heavily."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """R² coefficient - proportion of variance explained (1=perfect, 0=baseline)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# =============================================================================
# EVALUATION AND DISPLAY FUNCTIONS
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train model and compute test metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'y_pred': y_pred  # Stored for visualization
    }


def print_results_table(results, title):
    """Display formatted results table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_metrics_comparison(results, title, filename):
    """Bar chart comparing MAE, RMSE, and R² across models."""
    models = list(results.keys())
    mae_values = [results[m]['MAE'] for m in models]
    rmse_values = [results[m]['RMSE'] for m in models]
    r2_values = [results[m]['R2'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    # MAE (lower is better)
    bars1 = axes[0].bar(models, mae_values, color=colors[:len(models)], edgecolor='black')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error (lower=better)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mae_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # RMSE (lower is better)
    bars2 = axes[1].bar(models, rmse_values, color=colors[:len(models)], edgecolor='black')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error (lower=better)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # R² (higher is better)
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
    """Scatter plot: predicted vs actual values (diagonal = perfect prediction)."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        
        axes[i].scatter(y_test, y_pred, alpha=0.5, color=colors[i % len(colors)], 
                       s=30, edgecolor='black', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect')
        
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
    """Residual analysis - good models have residuals centered at 0."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        residuals = y_test - y_pred  # Error = actual - predicted
        
        axes[i].scatter(y_pred, residuals, alpha=0.5, color=colors[i % len(colors)], 
                       s=30, edgecolor='black', linewidth=0.3)
        axes[i].axhline(y=0, color='r', linestyle='--', linewidth=2)  # Zero line
        
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Residuals', fontsize=11)
        axes[i].set_title(f'{model_name}\nMAE={metrics["MAE"]:.2f}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_ablation_heatmap(results, filename):
    """Heatmap for ablation study (effect of pruning and smoothing)."""
    configs = list(results.keys())
    metrics_names = ['MAE', 'RMSE', 'R2']
    
    data = np.array([[results[c][m] for m in metrics_names] for c in configs])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_yticklabels(configs, fontsize=10)
    
    # Display values in each cell
    for i in range(len(configs)):
        for j in range(len(metrics_names)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', 
                   color='black', fontsize=11, fontweight='bold')
    
    ax.set_title('Ablation Study: Pruning & Smoothing Effects', 
                fontsize=14, fontweight='bold')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_final_summary(results_california, results_friedman, filename):
    """Final summary comparing R² on both main datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('M5P Model - Performance Summary', fontsize=16, fontweight='bold')
    
    # California Housing
    models_c = list(results_california.keys())
    r2_c = [results_california[m]['R2'] for m in models_c]
    colors_c = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars1 = axes[0].bar(models_c, r2_c, color=colors_c[:len(models_c)], 
                       edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R2 Score', fontsize=12)
    axes[0].set_title('California Housing Dataset', fontsize=13)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, r2_c):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        axes[0].text(bar.get_x() + bar.get_width()/2, ypos, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Friedman #1
    models_f = list(results_friedman.keys())
    r2_f = [results_friedman[m]['R2'] for m in models_f]
    colors_f = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
    
    bars2 = axes[1].bar(models_f, r2_f, color=colors_f[:len(models_f)], 
                       edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('R2 Score', fontsize=12)
    axes[1].set_title('Friedman #1 Dataset', fontsize=13)
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars2, r2_f):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_california_housing():
    """
    Benchmark 1: California Housing Dataset
    
    Real-world dataset from sklearn (California house prices).
    - 8 features: median income, house age, rooms, etc.
    - Target: median house price (in $100,000)
    
    Tests model performance on real regression task.
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 1: California Housing Dataset (sklearn)")
    print("#"*60)
    
    # Load dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    # Subsample for faster execution
    np.random.seed(42)
    idx = np.random.choice(len(y), size=2000, replace=False)
    X, y = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: California Housing (House Prices)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")
    print(f"Target: Median house value (in $100,000)")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'M5P (ours)': M5P(min_samples_split=20, max_depth=5, prune=True, 
                         smoothing=True, penalty_factor=2.0)
    }
    
    # Evaluation
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "California Housing Results")
    
    # Generate plots
    plot_metrics_comparison(results, "California Housing - Model Comparison", 
                           "california_metrics.png")
    plot_predictions_scatter(y_test, results, 
                            "California Housing - Predictions vs Actual", 
                            "california_scatter.png")
    plot_residuals(y_test, results, "California Housing - Residuals Analysis", 
                  "california_residuals.png")
    
    return results, y_test


def benchmark_friedman():
    """
    Benchmark 2: Friedman #1 Dataset
    
    Synthetic benchmark from sklearn (standard for regression).
    Formula: y = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5 + noise
    
    Tests ability to capture non-linear relationships.
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 2: Friedman #1 Dataset (sklearn)")
    print("#"*60)
    
    # Generate Friedman #1 dataset
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: Friedman #1 (Regression Benchmark)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Formula: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Models to compare (including multiple Decision Tree depths)
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree (d=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Decision Tree (d=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (d=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        'M5P (ours)': M5P(min_samples_split=15, max_depth=5, prune=True, 
                         smoothing=True, penalty_factor=2.0)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Friedman #1 Results")
    
    plot_metrics_comparison(results, "Friedman #1 - Model Comparison", 
                           "friedman_metrics.png")
    plot_predictions_scatter(y_test, results, 
                            "Friedman #1 - Predictions vs Actual", 
                            "friedman_scatter.png")
    plot_residuals(y_test, results, "Friedman #1 - Residuals Analysis", 
                  "friedman_residuals.png")
    
    return results, y_test


def benchmark_pruning_smoothing():
    """
    Benchmark 3: Ablation Study - Effect of Pruning and Smoothing
    
    Compares 4 configurations:
    - No pruning, no smoothing (baseline tree)
    - Pruning only
    - Smoothing only
    - Both pruning and smoothing (full M5P)
    
    Measures contribution of each technique to final performance.
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 3: Pruning & Smoothing Analysis")
    print("#"*60)
    
    # Dataset with more noise to better show pruning effect
    X, y = make_friedman1(n_samples=800, n_features=10, noise=1.5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    
    print(f"\nDataset: Friedman #1 (with more noise for ablation)")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # 4 configurations to test
    configs = {
        'No Prune, No Smooth': M5P(min_samples_split=15, max_depth=5, 
                                   prune=False, smoothing=False),
        'Prune, No Smooth': M5P(min_samples_split=15, max_depth=5, 
                               prune=True, smoothing=False, penalty_factor=2.0),
        'No Prune, Smooth': M5P(min_samples_split=15, max_depth=5, 
                               prune=False, smoothing=True),
        'Prune + Smooth': M5P(min_samples_split=15, max_depth=5, 
                             prune=True, smoothing=True, penalty_factor=2.0)
    }
    
    results = {}
    for name, model in configs.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "Pruning & Smoothing Comparison")
    
    plot_metrics_comparison(results, "Ablation Study - Pruning & Smoothing", 
                           "ablation_metrics.png")
    plot_ablation_heatmap(results, "ablation_heatmap.png")
    plot_predictions_scatter(y_test, results, "Ablation - Predictions vs Actual", 
                            "ablation_scatter.png")
    
    return results, y_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point - runs all benchmarks."""
    print("\n" + "="*60)
    print(" M5P MODEL TREE - COMPREHENSIVE BENCHMARK")
    print(" Member 3 - ENSAM Project")
    print("="*60)
    
    # Run all 3 benchmarks
    results_california, y_test_c = benchmark_california_housing()
    results_friedman, y_test_f = benchmark_friedman()
    results_ablation, y_test_a = benchmark_pruning_smoothing()
    
    # Summary plot
    plot_final_summary(results_california, results_friedman, "final_summary.png")
    
    # Final summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    print("\n[California Housing Dataset]")
    best_c = max(results_california.keys(), key=lambda x: results_california[x]['R2'])
    print(f"  Best model: {best_c} (R2 = {results_california[best_c]['R2']:.4f})")
    
    print("\n[Friedman #1 Dataset]")
    best_f = max(results_friedman.keys(), key=lambda x: results_friedman[x]['R2'])
    print(f"  Best model: {best_f} (R2 = {results_friedman[best_f]['R2']:.4f})")
    
    print("\n[Ablation Study]")
    best_a = max(results_ablation.keys(), key=lambda x: results_ablation[x]['R2'])
    print(f"  Best config: {best_a} (R2 = {results_ablation[best_a]['R2']:.4f})")
    
    # List of generated plots
    print("\n" + "="*60)
    print(" PLOTS GENERATED:")
    print("="*60)
    print("  - california_metrics.png")
    print("  - california_scatter.png")
    print("  - california_residuals.png")
    print("  - friedman_metrics.png")
    print("  - friedman_scatter.png")
    print("  - friedman_residuals.png")
    print("  - ablation_metrics.png")
    print("  - ablation_heatmap.png")
    print("  - ablation_scatter.png")
    print("  - final_summary.png")
    
    print("\n" + "="*60)
    print(" BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
