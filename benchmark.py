"""
M5P Model Tree - Benchmark Script
     ENSAM Project

Ce script compare les performances de M5P avec d'autres modèles de régression
sur des datasets réels de sklearn.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder sans afficher
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from model import M5P
from utils import train_test_split


# =============================================================================
# MÉTRIQUES D'ÉVALUATION
# =============================================================================

def mean_absolute_error(y_true, y_pred):
    """Erreur absolue moyenne - mesure l'écart moyen entre prédictions et réalité."""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """Racine de l'erreur quadratique moyenne - pénalise davantage les grandes erreurs."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """Coefficient de détermination R² - proportion de variance expliquée (1 = parfait, 0 = nul)."""
    ss_res = np.sum((y_true - y_pred) ** 2)  # Somme des carrés résiduels
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Somme totale des carrés
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# =============================================================================
# FONCTIONS D'ÉVALUATION ET D'AFFICHAGE
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Entraîne un modèle et calcule les métriques sur le jeu de test."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'y_pred': y_pred  # Stocké pour les plots
    }


def print_results_table(results, title):
    """Affiche un tableau formaté des résultats."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print(f"{'-'*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['R2']:>10.4f}")
    print(f"{'='*60}")


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def plot_metrics_comparison(results, title, filename):
    """Graphique en barres comparant MAE, RMSE et R² pour tous les modèles."""
    models = list(results.keys())
    mae_values = [results[m]['MAE'] for m in models]
    rmse_values = [results[m]['RMSE'] for m in models]
    r2_values = [results[m]['R2'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    # MAE (plus bas = meilleur)
    bars1 = axes[0].bar(models, mae_values, color=colors[:len(models)], edgecolor='black')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error (lower=better)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mae_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # RMSE (plus bas = meilleur)
    bars2 = axes[1].bar(models, rmse_values, color=colors[:len(models)], edgecolor='black')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error (lower=better)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # R² (plus haut = meilleur)
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
    """Scatter plot: valeurs prédites vs valeurs réelles (diagonale = prédiction parfaite)."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        
        axes[i].scatter(y_test, y_pred, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolor='black', linewidth=0.3)
        
        # Ligne de prédiction parfaite (y_pred = y_test)
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
    """Analyse des résidus (erreurs) - un bon modèle a des résidus centrés sur 0."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        residuals = y_test - y_pred  # Erreur = réalité - prédiction
        
        axes[i].scatter(y_pred, residuals, alpha=0.5, color=colors[i % len(colors)], s=30, edgecolor='black', linewidth=0.3)
        axes[i].axhline(y=0, color='r', linestyle='--', linewidth=2)  # Ligne zéro
        
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Residuals', fontsize=11)
        axes[i].set_title(f'{model_name}\nMAE={metrics["MAE"]:.2f}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Plot saved: {filename}")


def plot_ablation_heatmap(results, filename):
    """Heatmap pour l'étude d'ablation (effet du pruning et smoothing)."""
    configs = list(results.keys())
    metrics_names = ['MAE', 'RMSE', 'R2']
    
    data = np.array([[results[c][m] for m in metrics_names] for c in configs])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_yticklabels(configs, fontsize=10)
    
    # Afficher les valeurs dans chaque cellule
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


def plot_final_summary(results_california, results_friedman, filename):
    """Résumé final comparant R² sur les deux datasets principaux."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('M5P Model - Performance Summary', fontsize=16, fontweight='bold')
    
    # California Housing
    models_c = list(results_california.keys())
    r2_c = [results_california[m]['R2'] for m in models_c]
    colors_c = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars1 = axes[0].bar(models_c, r2_c, color=colors_c[:len(models_c)], edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R2 Score', fontsize=12)
    axes[0].set_title('California Housing Dataset', fontsize=13)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, r2_c):
        ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        axes[0].text(bar.get_x() + bar.get_width()/2, ypos, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Friedman #1
    models_f = list(results_friedman.keys())
    r2_f = [results_friedman[m]['R2'] for m in models_f]
    colors_f = ['#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71']
    
    bars2 = axes[1].bar(models_f, r2_f, color=colors_f[:len(models_f)], edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('R2 Score', fontsize=12)
    axes[1].set_title('Friedman #1 Dataset', fontsize=13)
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars2, r2_f):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
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
    - Dataset réel de sklearn (prix immobilier en Californie)
    - 8 features: revenu médian, âge maison, nb pièces, etc.
    - Target: prix médian des maisons (en $100,000)
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 1: California Housing Dataset (sklearn)")
    print("#"*60)
    
    # Chargement du dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    
    # Sous-échantillonnage pour accélérer l'exécution
    np.random.seed(42)
    idx = np.random.choice(len(y), size=2000, replace=False)
    X, y = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: California Housing (House Prices)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")
    print(f"Target: Median house value (in $100,000)")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Modèles à comparer
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'M5P (ours)': M5P(min_samples_split=20, max_depth=5, prune=True, smoothing=True, penalty_factor=2.0)
    }
    
    # Évaluation
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print_results_table(results, "California Housing Results")
    
    # Génération des plots
    plot_metrics_comparison(results, "California Housing - Model Comparison", "california_metrics.png")
    plot_predictions_scatter(y_test, results, "California Housing - Predictions vs Actual", "california_scatter.png")
    plot_residuals(y_test, results, "California Housing - Residuals Analysis", "california_residuals.png")
    
    return results, y_test


def benchmark_friedman():
    """
    Benchmark 2: Friedman #1 Dataset
    - Dataset synthétique de sklearn (benchmark standard pour régression)
    - Formule: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise
    - Teste la capacité du modèle à capturer des relations non-linéaires
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 2: Friedman #1 Dataset (sklearn)")
    print("#"*60)
    
    # Génération du dataset Friedman #1
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: Friedman #1 (Regression Benchmark)")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Formula: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Modèles à comparer (incluant plusieurs profondeurs de Decision Tree)
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
    
    print_results_table(results, "Friedman #1 Results")
    
    plot_metrics_comparison(results, "Friedman #1 - Model Comparison", "friedman_metrics.png")
    plot_predictions_scatter(y_test, results, "Friedman #1 - Predictions vs Actual", "friedman_scatter.png")
    plot_residuals(y_test, results, "Friedman #1 - Residuals Analysis", "friedman_residuals.png")
    
    return results, y_test


def benchmark_pruning_smoothing():
    """
    Benchmark 3: Étude d'ablation - Effet du Pruning et Smoothing
    - Compare 4 configurations: avec/sans pruning × avec/sans smoothing
    - Permet de mesurer l'apport de chaque technique
    """
    print("\n" + "#"*60)
    print(" BENCHMARK 3: Pruning & Smoothing Analysis")
    print("#"*60)
    
    # Dataset avec plus de bruit pour mieux voir l'effet du pruning
    X, y = make_friedman1(n_samples=800, n_features=10, noise=1.5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    
    print(f"\nDataset: Friedman #1 (with more noise for ablation)")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # 4 configurations à tester
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal - exécute tous les benchmarks."""
    print("\n" + "="*60)
    print(" M5P MODEL TREE - COMPREHENSIVE BENCHMARK")
    print(" Member 3 - ENSAM Project")
    print("="*60)
    
    # Exécution des 3 benchmarks
    results_california, y_test_c = benchmark_california_housing()
    results_friedman, y_test_f = benchmark_friedman()
    results_ablation, y_test_a = benchmark_pruning_smoothing()
    
    # Graphique récapitulatif
    plot_final_summary(results_california, results_friedman, "final_summary.png")
    
    # Résumé final
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
    
    # Liste des plots générés
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
