import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set the stock ticker symbol (example: Apple Inc.)
ticker_symbol = "AMZN" 

# Download historical stock data for the last 5 years
data = yf.download(ticker_symbol, start="2019-01-01", end="2025-01-01")

# Display the first few rows of the data
print("Historical Stock Data for", ticker_symbol)
print(data.head())

correlation_matrix = data.corr()
print("Correlation Matrix of Features:")
print(correlation_matrix)

# Plot the stock's closing prices
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label=f'{ticker_symbol} Closing Prices')
plt.title(f'{ticker_symbol} Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()



