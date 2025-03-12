import yfinance as yf
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def load_stock_price(ticker: str, start_date: str = '1995-01-01', end_date: str = '2025-01-01'):
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
    val_size = 242
    test_size = 121
    train_size = len(data) - val_size - test_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def load_stock_price_and_known(ticker: str, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
    '''
    input:
        tickers, single string of a ticker, like "AAPL"
        start_date: data start
        end_date: data end
    output:

    '''

    # Load Close price and Volume together
    df = yf.download(ticker, start=start_date, end=end_date)

    df[['Close', 'Volume']].to_csv("data.csv")
    df = pd.read_csv("data.csv", index_col=0, names=["Close", "Volume"], skiprows=3)
    df.head()
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    
    # Extract required columns and create new time-based features
    df_processed = df[['Close', 'Volume']].copy()
    #df_processed['Day_of_Week'] = df.index.dayofweek  # Monday = 0, Sunday = 6
    # #df_processed['Day_of_Month'] = df.index.day
    # df_processed['Week_of_Year'] = df.index.isocalendar().week
    # df_processed['Week_of_Year'] = df_processed['Week_of_Year'].astype('int32')
    df_processed['Month'] = df.index.month
    

    # Fill any missing values if needed
    data = df_processed.ffill()

    # Split into train, validation, and test sets
    val_size = 242
    test_size = 121
    train_size = len(data) - val_size - test_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data





