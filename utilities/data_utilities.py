import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from config import (DEFAULT_REACTION_WINDOW,STOCK_LIST_PATH)
# ------------------------------------------------------------
# Formatting utilities 
# ------------------------------------------------------------
def parse_date(column_name):
    """
        Parse 'YYYY-MM-DD' into datetime.date. 
        Returns None on failure.
    """
    try:
        return pd.to_datetime(column_name, errors="coerce")
    except Exception:
        raise ValueError (f"parse_date got an invalid input at column {column_name}")
    
def parse_numeric(column_name: pd.Series) -> pd.Series:
    """
        Coerce a Series to numeric.
        Non-parsable values become NaN.
    """
    try:
        return pd.to_numeric(column_name, errors="coerce")
    except Exception:
        raise ValueError ("parse_numeric got an invalid input")
    
def change_column_name(df, list_of_col_names, correct_col_name):
    for col_name in df.columns:
        if col_name in list_of_col_names:
            df = df.rename(columns = {col_name: correct_col_name})
            return df
    return df

def to_float_or_none(x):
    if x in (None, "None", ""):
        return None
    return float(x)

# ------------------------------------------------------------
# Merging utilities 
# ------------------------------------------------------------

def dedup_earnings(earnings_df, window_days=30):
    df = earnings_df.sort_values(["stock", "earnings_date"]).reset_index(drop=True).copy()
    df["_prev_date"] = df.groupby("stock")["earnings_date"].shift(1)
    df["_gap"] = (df["earnings_date"] - df["_prev_date"]).dt.days
    df["_has_eps"] = df["reported_eps"].notna()

    drop_idx = set()
    for i in df.index[df["_gap"] <= window_days]:
        prev_i = i - 1
        if prev_i in drop_idx:
            continue
        if df.at[i, "_has_eps"] and not df.at[prev_i, "_has_eps"]:
            drop_idx.add(prev_i)
        elif not df.at[i, "_has_eps"] and df.at[prev_i, "_has_eps"]:
            drop_idx.add(i)
        else:
            drop_idx.add(prev_i)  # equal quality — keep later date

    if drop_idx:
        print(f"  dedup_earnings: removed {len(drop_idx)} duplicate rows (window={window_days}d) — re-run stage1 with update=True to clean DB")

    return df.drop(index=drop_idx).drop(columns=["_prev_date", "_gap", "_has_eps"]).reset_index(drop=True)


def merge_prices_earnings_dates(stock_prices, earnings_dates):
    merged_df = pd.merge_asof(
        stock_prices, earnings_dates, left_on='date',
        right_on = 'earnings_date', by = "stock", direction="forward") 
    # Merge the result with EPS data
    return merged_df

def map_sector_data_to_main_df(main_df: pd.DataFrame,sector_df:pd.DataFrame):
    sector_df = sector_df.drop(columns=["company_name","ingested_at"])
    merged_df = main_df.merge(sector_df, on="stock", validate="m:1")
    return merged_df

# ------------------------------------------------------------
# Misc utilities 
# ------------------------------------------------------------
def directory_checks():
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/reports/").mkdir(exist_ok=True)

def get_alpha_vantage_api_key() -> str:
    load_dotenv() 
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY environment variable"
        )
    return api_key

def build_earnings_df(df):
    """ Separate earnings rows """
    # Boolean mask, gives True for rows with earnings dates
    earnings_mask =  df[DEFAULT_REACTION_WINDOW].notna()
    earnings_df = df.loc[earnings_mask]
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])
    return earnings_df

def read_stocks_to_fetch() -> list[str]:
    """
        Reads stocks from a file. Supports:
        - .txt: one stock per line
        - .csv: column named symbol/ticker/stock 
        Returns a list of all unique stocks (uppercase, no spaces)
    """
    path = Path(STOCK_LIST_PATH)
    if path.suffix.lower() == ".csv":
        print(f"Reading stocks from {STOCK_LIST_PATH} file")
        stock_prices_df = pd.read_csv(path)
        col = None
        for c in ("stock", "symbol", "ticker","Stock", "Symbol", "Ticker"):
            if c in stock_prices_df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"CSV must contain a symbol/ticker/stock column. Found: {list(stock_prices_df.columns)}")
        stocks = stock_prices_df[col].astype(str).str.strip().tolist()
        print(f"{len(stocks)} imported from .csv file")
    else:
        print("Reading stocks from .txt file")
        stocks = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        print(f"{len(stocks)} imported from .txt file")

    # Basic cleanup
    stocks = [t.replace(" ", "").upper() for t in stocks if t]
    # dedupe preserve order
    seen = set()
    out = []
    for stock in stocks:
        if stock and stock != "NAN" and stock not in seen:
            out.append(stock)
            seen.add(stock)
    return out