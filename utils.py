import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    r2_score
)
from statsmodels.stats.diagnostic import acorr_ljungbox

# SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Full metrics bundle
def get_all_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    # Only compute MSLE if all values are positive
    msle = None
    if np.all(y_true > 0) and np.all(y_pred > 0):
        msle = mean_squared_log_error(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "SMAPE": smape_val,
        "MSLE": msle
    }

# Residuals
def get_residuals(y_true, y_pred):
    return np.array(y_true) - np.array(y_pred)

# Ljung-Box test
def check_residual_white_noise(residuals, lags=10):
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return result["lb_pvalue"].values[0]
