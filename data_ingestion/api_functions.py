# data_ingestion/api_functions.py
import requests
import time
import pandas as pd
import yfinance as yf

from config import (ALPHAVANTAGE_BASE_URL,BACKOFF_SECONDS)
from data_ingestion.data_utilities import get_alpha_vantage_api_key

def fetch_daily_adjusted_prices(stock: str, outputsize = "full", max_attempts=5):
    API_KEY = get_alpha_vantage_api_key()
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }
    for attempt in range(max_attempts):
        try:
            r = requests.get(ALPHAVANTAGE_BASE_URL, params=params,timeout=(5, 30))
            r.raise_for_status()
            data = r.json()
            # handle AlphaVantage rate limit / bad payload
            # retryable AlphaVantage responses
            if isinstance(data, dict) and ("Note" in data or "Information" in data):
                raise RuntimeError(f"AV throttled: {data.get('Note') or data.get('Information')}")

            # non-retryable AV response (bad symbol, etc.)
            if isinstance(data, dict) and "Error Message" in data:
                return data

            # success or unexpected -> retry
            if not isinstance(data, dict) or "Time Series (Daily)" not in data:
                raise RuntimeError(f"Bad payload keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

            return data
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            sleep_time = min(2 ** attempt, 60)  # backoff
            print(f"Exception {e} raised.\nRetry {attempt+1}/{max_attempts} for {stock} in {sleep_time}s")
            time.sleep(sleep_time)

def get_earnings_data_from_api(stock, max_attempts=5):
    # get_earnings_data_from Alpha Vantage
    api_key = get_alpha_vantage_api_key()
    params = {
        "function": "EARNINGS", 
        "symbol": stock, 
        "apikey": api_key}  
    for attempt in range(max_attempts):
        try:
            r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json() 
            if isinstance(data, dict) and ("Note" in data or "Information" in data):
                raise RuntimeError(f"AV throttled: {data.get('Note') or data.get('Information')}")

            # non-retryable AV response (bad symbol, etc.)
            if isinstance(data, dict) and "Error Message" in data:
                return data     
            # success or unexpected -> retry
            if not isinstance(data, dict):
                raise RuntimeError(f"Bad payload keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
 
            return data 
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            sleep_time = min(2 ** attempt, 60)  # backoff
            print(f"Exception {e} raised.\nRetry {attempt+1}/{max_attempts} for {stock} in {sleep_time}s")
            time.sleep(sleep_time)

def get_earnings_dates_yf(ticker: str, limit : int = 12):
    t = yf.Ticker(ticker)
    df = t.get_earnings_dates(limit=limit)
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "earnings_date",
            "estimated_eps",
            "reported_eps",
            "surprise_percentage"
        ])

    df = df.reset_index()

    df = df.rename(columns={
        "Earnings Date": "earnings_date",
        "EPS Estimate": "estimated_eps",
        "Reported EPS": "reported_eps",
        "Surprise(%)": "surprise_percentage"
    })

    #Keep only relevant columns
    df = df[[
        "earnings_date",
        "estimated_eps",
        "reported_eps",
        "surprise_percentage"
    ]]

    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date