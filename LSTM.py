import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from load_data import load_stock_price
from evaluation import evaluation

def set_random_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initialize hidden state: (num_layers * num_directions, batch_size, hidden_size)
        # initialize cell state: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # input x : (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # (batch_size, seq_length, hidden_size)
        out = self.fc(lstm_out[:, -1, :])  # using the last state in seq_length
        return out

def create_sequences(data: np.ndarray, seq_length: int):
    """
    Create sequences with multiple features and labels for one feature.
    
    Parameters:
    - data: 2D numpy array of shape (num_samples, num_features)
    - seq_length: Length of each sequence
    
    Returns:
    - sequences: Numpy array of shape (num_sequences, seq_length, num_features)
    - labels: Numpy array of shape (num_sequences,)
    """
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, 0])  # Only target price (first feature)
    return np.array(sequences), np.array(labels)

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_val_data = scaler.transform(val_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_val_data, scaled_test_data, scaler

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
    test_labels_reshaped = test_labels.cpu().numpy().reshape(-1, 1)
    dummy_actual = np.zeros((test_labels_reshaped.shape[0], 2))
    dummy_actual[:, 0] = test_labels_reshaped[:, 0]

    actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]

    return predicted_prices, actual_prices


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Early stopping utility to stop training when validation loss stops improving.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
    


def main(): 
     
    ticker = "^GSPC"
    train_data, val_data, test_data = load_stock_price(ticker)
    scaled_train_data, scaled_val_data, scaled_test_data, scaler = scaling(train_data, val_data, test_data)


    seq_length = 50  # Example sequence length

    train_sequences, train_labels = create_sequences(scaled_train_data, seq_length)
    val_sequences, val_labels = create_sequences(scaled_val_data, seq_length)
    test_sequences, test_labels = create_sequences(scaled_test_data, seq_length)

    # Convert numpy arrays to PyTorch tensors
    train_sequences = torch.tensor(train_sequences, dtype = torch.float32)
    train_labels = torch.tensor(train_labels, dtype = torch.float32)
    val_sequences = torch.tensor(val_sequences, dtype = torch.float32)
    val_labels = torch.tensor(val_labels, dtype = torch.float32)
    test_sequences = torch.tensor(test_sequences, dtype = torch.float32)
    test_labels = torch.tensor(test_labels, dtype = torch.float32)


    # DataLoader for batching
    batch_size = 64
    train_dataset = TensorDataset(train_sequences, train_labels)
    val_dataset = TensorDataset(val_sequences, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    input_size = train_sequences.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = MyLSTM(input_size, hidden_size, output_size, num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)

    # Train the model
    epochs = 100

    early_stopper = EarlyStopper(patience = 10)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}    Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)

                outputs = model(X_val)
                loss = criterion(outputs.squeeze(), y_val)

                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}    Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Update the best!")
        if early_stopper.should_stop(avg_val_loss):
            print(f"Early stopped!")
            break
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(DEVICE)
        print("Loaded the best model state.")

    # Evaluate the model on test data
    model.eval()

    # Collect predictions on the test set
    predicted_prices = []


    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            X_test = test_sequences[i:i + batch_size].to(DEVICE)
            batch_outputs = model(X_test).cpu().numpy()
            predicted_prices.extend(batch_outputs.flatten())

    predicted_prices, actual_prices = inverse_scaling(scaler, predicted_prices, test_labels)

    ## Evaluation
    mse, mae, mape = evaluation(actual_prices, predicted_prices)
    print(f'LSTM predction of {ticker}: MSE = {mse:.2f}, MAE = {mae:.2f}, MAPE = {mape:.2f}')


    # Plot the test results
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index[seq_length:], actual_prices, label='Actual Prices')
    plt.plot(test_data.index[seq_length:], predicted_prices, label='Predicted Prices')
    plt.title(f'{ticker} (S&P 500) Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

