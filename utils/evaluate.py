from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def test_regression(y, y_hat):
    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    return mae, mse, rmse, r2