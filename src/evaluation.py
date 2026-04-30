import numpy as np

# Compute the Root Mean Squared Error (RMSE) between true ratings and predicted ratings
def rmse_function(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Formats the results into a consistent structure for easy comparison and reporting
def format_result(name, preds, rmse, model=None, params=None):
    return {
        "name": name,
        "preds": preds,
        "rmse": rmse,
        "model": model,
        "params": params
    }