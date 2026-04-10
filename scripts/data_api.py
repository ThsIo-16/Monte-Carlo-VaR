"""
Data Engineering Pipeline for Market Risk (Magnificent 7)
Fetches live historical data from Yahoo Finance API.
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Tickers
MAG_7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
#Data Download Function
def fetch_portfolio_data(tickers, years=5):
    print(f"Loading data for {len(tickers)} tickers...")
    
    end_date = datetime.now()
    start_date = end_date.replace(year=end_date.year - years)
    
    #Downloading Closing Prices for last 5 years
    df = yf.download(tickers, start=start_date, end=end_date,auto_adjust=True)['Close']
    df.dropna(inplace=True)
    
    print("Data fetched successfully!")
    print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total Trading Days: {len(df)}")
    return df
#Returns Function
def calculate_daily_returns(df):
    print("Calculating daily percentage returns...")
    returns_df = df.pct_change().dropna()
    return returns_df

if __name__ == "__main__":
    price_data = fetch_portfolio_data(MAG_7_TICKERS, years=5)
    returns_data = calculate_daily_returns(price_data)
    #Path creation And Data Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    price_data.to_csv(os.path.join(data_dir, 'mag7_prices.csv'))
    returns_data.to_csv(os.path.join(data_dir, 'mag7_returns.csv'))
    
    print(f"\nData successfully saved to: {data_dir}")
    print("\n--- Preview of Daily Returns ---")
    print(returns_data.tail()) # Last 5 days