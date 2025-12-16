import numpy as np

def fit_linear_model(X, y):
    """
    Fit linear regression model using Ordinary Least Squares (OLS).
    
    Solves: min ||Xβ + β₀ - y||²
    
    Falls back to Ridge regression if matrix is singular (rank-deficient).
    This handles cases with:
    - Multicollinearity (correlated features)
    - More features than samples
    - Numerical instability
    
    Returns:
    --------
    dict with 'intercept' (β₀) and 'coefficients' (β)
    """
    n_samples, n_features = X.shape
    
    # Add bias column for intercept
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # Attempt OLS using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)
    
    if rank < X_with_bias.shape[1]:
        ridge_lambda = 1e-6
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        coeffs = np.linalg.solve(XtX + ridge_lambda * np.eye(n_features + 1), Xty)
    
    return {
        'intercept': coeffs[0],
        'coefficients': coeffs[1:]
    }

def predict_linear(model, X):
    """
    Make predictions using fitted linear model.
    
    Formula: ŷ = Xβ + β₀
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X @ model['coefficients'] + model['intercept']
