import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_api import load_stock_price_and_known
from scaling_and_inverse import scaling, inverse_scaling
from TFT_simple import TFT
import copy

class StockDataset(Dataset):
    def __init__(self, df, whole_seq_length=30, decode_length = 1):
        '''
        seq_length: the whole encoder-decoder length
        return: (in our case of one step prediction)
            data: len(data) = seq_length
            labels: len(labels) = 1
             
        '''
        ## 
        self.seq_len = whole_seq_length
        self.decode_len = decode_length
        self.data = df[['Close', 'Volume', 'Day_of_Week', 'Day_of_Month', 'Week_of_Year', 'Month']].values
        self.labels = df['Close'].values  
    
    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.labels[(index + self.seq_len - self.decode_len):(index + self.seq_len)]

        x_tensor = torch.empty((self.seq_len, x.shape[1]), dtype=torch.float32)
        x_tensor[:, 0:2] = torch.tensor(x[:, 0:2], dtype=torch.float32)  # 'Close' and 'Volume'
        x_tensor[:, 2:] = torch.tensor(x[:, 2:], dtype=torch.int32)  # 'Day_of_Week', 'Day_of_Month', 'Week_of_Year', 'Month'
        
        # Convert y to a tensor (float64)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


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

categorical_cols = ['Day_of_Week', 'Day_of_Month', 'Week_of_Year','Month']
real_cols = ['Close','Volume']

# Hyper pparams
SEQ_LEN = 253 
ENCODE_LEN = 252
DECODE_LEN = SEQ_LEN - ENCODE_LEN
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####### params for TFT
config = {}
# params for variables
config['num_categoical_variables'] = 4  # 'Day_of_Week', 'Day_of_Month', 'Week_of_Year','Month'
config['num_real_encoder'] = 2    # 'Close','Volume'
config['num_real_decoder'] = 0
config['num_masked_vars'] = 2    # 'Close','Volume'
config['cat_embedding_vocab_sizes'] = [8,32,54,13]   # 'Day_of_Week + 1', 'Day_of_Month + 1', 'Week_of_Year + 1','Month + 1'
config['embedding_dim'] = 4
config['ouput_len'] = 1    # 'Close'

#params for sequence
config['seq_length'] = SEQ_LEN  
config['num_encoder_steps'] = ENCODE_LEN

#params for models
config['lstm_hidden_dimension'] = 32
config['lstm_layers'] = 2
config['dropout'] = 0.3
config['device'] = DEVICE
config['batch_size'] = BATCH_SIZE
config['attn_heads'] = 1



def main(): 
     
    ticker = "^GSPC"
    train_data, val_data, test_data = load_stock_price_and_known(ticker)
    scaled_train_data, scaled_val_data, scaled_test_data, scaler = scaling(train_data, val_data, test_data)
    
    train_dataset = StockDataset(scaled_train_data, SEQ_LEN, DECODE_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = StockDataset(scaled_test_data, SEQ_LEN, DECODE_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset = StockDataset(scaled_val_data, SEQ_LEN, DECODE_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TFT(config).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Train the model
    epochs = 50

    early_stopper = EarlyStopper(patience = 20)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
            outputs = model(X_train)
            loss = criterion(outputs, y_train.squeeze())

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
                loss = criterion(outputs, y_val.squeeze())

                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if epoch % 5 == 0:
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
        for X_test, _ in test_loader:
            X_test = X_test.to(DEVICE)
            batch_outputs = model(X_test).cpu().numpy()
            predicted_prices.extend(batch_outputs.flatten())


    '''
    predicted_prices, actual_prices = inverse_scaling(scaler, predicted_prices, test_labels)

    ## Evaluation
    mse, mae, mape = evaluation(actual_prices, predicted_prices)
    print(f'LSTM prediction of {ticker}: MSE = {mse:.2f}, MAE = {mae:.2f}, MAPE = {mape:.2f}')


    # Plot the test results
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index[seq_length:], actual_prices, label='Actual Prices')
    plt.plot(test_data.index[seq_length:], predicted_prices, label='Predicted Prices')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f"{ticker}_price_prediction.png", dpi=300, bbox_inches='tight')
    plt.show()

    '''


if __name__ == "__main__":
    main()

