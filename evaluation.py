import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero by filtering out zeros
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

def evaluation(actual_prices, predicted_prices):
    
    mse = np.mean((np.array(predicted_prices) - np.array(actual_prices)) ** 2)
    mae = np.mean(abs(np.array(predicted_prices) - np.array(actual_prices)))
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices)

    return mse, mae, mape