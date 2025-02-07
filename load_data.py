import yfinance as yf
from typing import List, Dict

def load_stock_price(ticker: str, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
    '''
    input:
        tickers, single string of a ticker, like "AAPL"
        start_date: data start
        end_date: data end
    output:

    '''

    # Load Close price and Volume together
    data = yf.download(ticker, start_date, end_date)[['Close', 'Volume']]

    # Fill any missing values if needed
    data = data.ffill()

    # Split into train, validation, and test sets
    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.10)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data



