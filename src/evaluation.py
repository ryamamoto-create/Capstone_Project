import numpy as np

# Compute the Root Mean Squared Error (RMSE) between true ratings and predicted ratings
def rmse_function(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))