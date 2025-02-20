from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scaling(train_data, val_data , test_data):
    """
    Scales the data using MinMaxScaler and returns scaled data and the scaler.
    
    Args:
        train_data (np.ndarray): Training data array.
        val_data (np.ndarray): Validation data array.
        test_data (np.ndarray): Test data array.

    Returns:
        tuple: Scaled train, validation, and test datasets along with the scaler instance.
    """
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()

    # Fit on training data and transform
    train_data_scaled = train_data.copy()
    train_data_scaled[['Close', 'Volume']] = scaler.fit_transform(train_data[['Close', 'Volume']])

    # Transform validation and test data without refitting
    val_data_scaled = val_data.copy()
    val_data_scaled[['Close', 'Volume']] = scaler.fit_transform(val_data[['Close', 'Volume']])

    test_data_scaled = test_data.copy()
    test_data_scaled[['Close', 'Volume']] = scaler.fit_transform(test_data[['Close', 'Volume']])

    return train_data_scaled, val_data_scaled, test_data_scaled, scaler


def inverse_scaling(scaler, predicted_prices, test_labels):
    """
    Inverse transforms the predicted prices and test labels using the given scaler.

    Args:
        scaler (MinMaxScaler): Fitted MinMaxScaler instance.
        predicted_prices (list): List of predicted prices from the model.
        test_labels (torch.Tensor): Ground truth labels from the test set.

    Returns:
        tuple: Inverse-transformed predicted and actual prices.
    """

    # Reshape predicted prices to match scaler's input format
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    # Create a dummy array with both price and volume
    dummy_input = np.zeros((predicted_prices.shape[0], 2))
    dummy_input[:, 0] = predicted_prices[:, 0]  # Populate the price column
    # Inverse transform
    predicted_prices = scaler.inverse_transform(dummy_input)[:, 0]  # Extract only the price

    # Inverse transform the actual test labels
    test_labels_reshaped = test_labels.to_numpy().reshape(-1, 1)
    dummy_actual = np.zeros((test_labels_reshaped.shape[0], 2))
    dummy_actual[:, 0] = test_labels_reshaped[:, 0]

    actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]

    return predicted_prices, actual_prices