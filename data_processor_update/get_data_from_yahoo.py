import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import time

def get_stock_data(ticker_file, start_date, end_date):
    """
    Get price data from Yahoo Finance and change to map WRDS fields
    
    Parameters:
    ticker_file (str): sp500_tickers
    start_date (str):  'YYYY-MM-DD'
    end_date (str):  'YYYY-MM-DD'
    
    Returns:
    pandas.DataFrame: price data in replace of original WRDS file
    """
    
    with open(ticker_file, 'r') as f:
        tickers = [line.strip() for line in f]
    
    all_data = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            time.sleep(0.5)
            
            df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
            
            if df.empty:
                print(f"Warning: {ticker} No data")
                failed_tickers.append((ticker, "No data"))
                continue
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: {ticker} Missing required colunms")
                failed_tickers.append((ticker, "Missing required colunms"))
                continue
                
            wrds_data = pd.DataFrame({
                'tic': ticker,
                'cusip': np.nan,
                'permno': np.nan,
                'permco': np.nan,
                'issuno': 0,
                'hexcd': np.nan,
                'hsiccd': np.nan,
                'date': df.index,
                'bidlo': df['Low'],
                'askhi': df['High'],
                'prc': df['Close'],
                'adj_close_q': df['Close'],  # auto_adjust=True, Close = adj_close
                'vol': df['Volume'],
                'ret': df['Close'].pct_change(),
                'bid': df['Low'],
                'ask': df['High'],
                'shrout': np.nan,  
                'cfacpr': 1,  # auto_adjust=True, cfacpr = 1
                'cfacshr': 1,
                'openprc': df['Open'],
                'numtrd': np.nan,
                'retx': df['Close'].pct_change()
            })
            
            wrds_data['date'] = pd.to_datetime(wrds_data['date'])
            
            all_data.append(wrds_data)
            print(f"Successfully retrieved data for {ticker}")

        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            failed_tickers.append((ticker, str(e)))
            continue
    
    if all_data:
        final_data = pd.concat(all_data, ignore_index=True)
        
        if failed_tickers:
            with open('failed_tickers.txt', 'w') as f:
                for ticker, reason in failed_tickers:
                    f.write(f"{ticker}: {reason}\n")
            
        return final_data
    else:
        print("No data")
        return pd.DataFrame()


stock_data = get_stock_data(
    'sp500_tickers.txt',
    start_date='1996-01-01',
    end_date='2025-02-10'
)

if not stock_data.empty:
    stock_data.to_csv('sp500_price_199601_202502.csv', index=False)
    print(f"Successfully saved data, {len(stock_data.tic.unique())} stocks in total")
