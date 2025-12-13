import numpy as np


def fit_linear_model(X, y):
    n_samples, n_features = X.shape
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)
    
    # Fallback to ridge if matrix is singular
    if rank < X_with_bias.shape[1]:
        lambda_ridge = 1e-6
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        coeffs = np.linalg.solve(XtX + lambda_ridge * np.eye(n_features + 1), Xty)
    
    return {
        'intercept': coeffs[0],
        'coefficients': coeffs[1:]
    }


def predict_linear(model, X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X @ model['coefficients'] + model['intercept']
