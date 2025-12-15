import numpy as np
from regression import fit_linear_model, predict_linear


def compute_mse(node, model):
    """
    Calcule l'erreur quadratique moyenne (MSE) du modèle sur les données du nœud.
    
    MSE = (1/n) * Σ(y_i - ŷ_i)²
    
    Args:
        node: Nœud contenant les données (node.X, node.y)
        model: Modèle linéaire avec 'intercept' et 'coefficients'
    
    Returns:
        float: Erreur quadratique moyenne
    """
    predictions = predict_linear(model, node.X)
    return np.mean((node.y - predictions) ** 2)


def count_subtree_params(node):
    """
    Compte le nombre total de paramètres dans un sous-arbre.
    
    Pour chaque modèle linéaire: p = n_features + 1 (coefficients + intercept)
    Pour un sous-arbre: somme des paramètres de toutes les feuilles
    
    Args:
        node: Racine du sous-arbre
    
    Returns:
        int: Nombre total de paramètres
    """
    if node.is_leaf:
        return len(node.linear_model['coefficients']) + 1
    
    return count_subtree_params(node.left) + count_subtree_params(node.right)


def adjusted_error(raw_error, n_samples, n_params, penalty_factor=1.0, use_weka_formula=False):
    """
    Calcule l'erreur ajustée avec pénalité de complexité.
    
    Deux formules disponibles:
    
    1. AIC (défaut): E_adjusted = E_raw * (1 + penalty_factor * p / n)
       • Pénalité linéaire
       • Robuste pour arbres profonds
       • Pas d'explosion quand n → p
    
    2. Weka (originale M5P): E_adjusted = E_raw * (n + PF * p) / (n - p)
       • Formule standard de Quinlan/Witten
       • Fonctionne bien pour arbres peu profonds
       • Peut exploser si p/n élevé
    
    Args:
        raw_error: Erreur brute (MSE)
        n_samples: Nombre d'échantillons
        n_params: Nombre de paramètres
        penalty_factor: Facteur multiplicatif (défaut=1.0)
                       AIC: 1.0 = standard, 2.0 = recommandé
                       Weka: 2.0 = standard (PF dans la formule)
        use_weka_formula: Si True, utilise formule Weka (n+PF*p)/(n-p)
                         Si False, utilise formule AIC (défaut)
    
    Returns:
        float: Erreur ajustée avec pénalité
    """
    if n_samples == 0:
        return np.inf
    
    if use_weka_formula:
        if n_samples <= n_params:
            return np.inf
        penalty = (n_samples + penalty_factor * n_params) / (n_samples - n_params)
    else:
        penalty = 1.0 + penalty_factor * n_params / n_samples
    
    return raw_error * penalty


def subtree_adjusted_error(node, penalty_factor=1.0, use_weka_formula=False):
    """
    Calcule l'erreur ajustée totale d'un sous-arbre.
    
    Pour un sous-arbre avec plusieurs feuilles:
        - Agrège les erreurs pondérées par le nombre d'échantillons
        - Compte le total de paramètres de toutes les feuilles
        - Applique la pénalité sur l'erreur agrégée
    
    Args:
        node: Racine du sous-arbre
        penalty_factor: Facteur de pénalité de complexité
        use_weka_formula: Si True, utilise formule Weka originale
    
    Returns:
        float: Erreur ajustée du sous-arbre
    """
    if node.is_leaf:
        raw_err = compute_mse(node, node.linear_model)
        n_params = len(node.linear_model['coefficients']) + 1
        return adjusted_error(raw_err, len(node.y), n_params, penalty_factor, use_weka_formula)
    
    left_samples = len(node.left.y)
    right_samples = len(node.right.y)
    total_samples = left_samples + right_samples
    
    left_err = subtree_adjusted_error(node.left, penalty_factor, use_weka_formula) * left_samples
    right_err = subtree_adjusted_error(node.right, penalty_factor, use_weka_formula) * right_samples
    
    return (left_err + right_err) / total_samples


def prune_tree(node, penalty_factor=2.0, use_weka_formula=False):
    """
    Élagage bottom-up basé sur la comparaison des erreurs ajustées.
    
    Algorithme:
        1. Élaguer récursivement les enfants (post-order)
        2. Calculer l'erreur ajustée du sous-arbre complet
        3. Calculer l'erreur ajustée d'un modèle linéaire unique au nœud
        4. Remplacer le sous-arbre si: E_linear_adjusted ≤ E_subtree_adjusted
    
    La pénalité de complexité favorise les modèles plus simples:
        - Sous-arbre: plus de paramètres → pénalité plus forte
        - Modèle unique: moins de paramètres → pénalité plus faible
    
    Args:
        node: Nœud racine du (sous-)arbre à élaguer
        penalty_factor: Facteur de pénalité (défaut=2.0)
                       AIC: 1.0-3.0 (2.0 recommandé)
                       Weka: 2.0 standard (PF)
        use_weka_formula: Si True, utilise formule Weka (n+PF*v)/(n-v)
                         Si False, utilise formule AIC (défaut)
    """
    if node.is_leaf:
        return
    
    prune_tree(node.left, penalty_factor, use_weka_formula)
    prune_tree(node.right, penalty_factor, use_weka_formula)
    
    subtree_err_adj = subtree_adjusted_error(node, penalty_factor, use_weka_formula)
    
    linear_raw_err = compute_mse(node, node.linear_model)
    n_params = len(node.linear_model['coefficients']) + 1
    linear_err_adj = adjusted_error(linear_raw_err, len(node.y), n_params, penalty_factor, use_weka_formula)
    
    if linear_err_adj <= subtree_err_adj:
        node.is_leaf = True
        node.left = None
        node.right = None


def smooth_predictions(node, parent_model=None, k=15):
    """
    Lisse les prédictions en combinant le modèle du nœud avec celui du parent.
    
    Formule de lissage M5:
        θ_smoothed = (n * θ_node + k * θ_parent) / (n + k)
    
    où:
        θ: Paramètres du modèle (intercept et coefficients)
        n: Nombre d'échantillons au nœud
        k: Constante de lissage (typiquement 15)
    
    Args:
        node: Nœud courant
        parent_model: Modèle du parent (None pour la racine)
        k: Paramètre de lissage
    """
    if parent_model is None:
        node.smoothed_model = node.linear_model
    else:
        n = len(node.y)
        
        intercept = (n * node.linear_model['intercept'] + k * parent_model['intercept']) / (n + k)
        coeffs = (n * node.linear_model['coefficients'] + k * parent_model['coefficients']) / (n + k)
        
        node.smoothed_model = {
            'intercept': intercept,
            'coefficients': coeffs
        }
    
    if not node.is_leaf:
        smooth_predictions(node.left, node.smoothed_model, k)
        smooth_predictions(node.right, node.smoothed_model, k)

