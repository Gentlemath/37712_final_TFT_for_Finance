import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
        h_0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(lstm_out[:, -1, :])
        return out

# Set input size, hidden size, and output size according to your data
input_size = 10    
hidden_size = 20   
output_size = 5    

# Initialize the model
model = MyLSTM(input_size, hidden_size, output_size)

# Set the stock ticker symbol (example: Apple Inc.)
ticker_symbol = "APPL" 

# Download historical stock data for the last 5 years
data = yf.download(ticker_symbol, start="2019-01-01", end="2025-01-01")

# Use the 'Close' price for prediction
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create training dataset (90% train, 10% test split)
train_size = int(len(scaled_prices) * 0.8)
train_data, test_data = scaled_prices[:train_size], scaled_prices[train_size:]

print("Training Data Size:", len(train_data))
print("Testing Data Size:", len(test_data))


def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return torch.tensor(sequences), torch.tensor(labels)

seq_length = 50  # Example sequence length

train_sequences, train_labels = create_sequences(train_data, seq_length)
test_sequences, test_labels = create_sequences(test_data, seq_length)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Reshape for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1

model = MyLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    
    outputs = model(train_sequences.unsqueeze(-1).float())
    loss = criterion(outputs, train_labels.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data
model.eval()
test_outputs = model(test_sequences.unsqueeze(-1).float()).detach().numpy()
test_labels = test_labels.numpy()

# Inverse transform the results to get the actual stock prices
test_outputs = scaler.inverse_transform(test_outputs)
test_labels = scaler.inverse_transform(test_labels.reshape(-1, 1))

# Plot the test results
plt.figure(figsize=(14, 7))
plt.plot(data.index[len(train_data) + seq_length:], test_labels, label='Actual Prices')
plt.plot(data.index[len(train_data) + seq_length:], test_outputs, label='Predicted Prices')
plt.title(f'{ticker_symbol} Stock Price Prediction (Test Set)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
