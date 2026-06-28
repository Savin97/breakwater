# ingestion/fetch_sp500_sectors.py
from utilities.data_utilities import read_stocks_to_fetch
import pandas as pd, requests
from datetime import datetime
from io import StringIO

def get_sp500_sectors():
    """
        Fetches sector,sub-sector data for all current S&P 500 stocks from Wikipedia (GCIS data).
        Returns a df:
        stock | name | sector | sub_sector
    """
    # importing stock list and sector data
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    stock_list = read_stocks_to_fetch()
    # headers = {
    # "User-Agent": (
    #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #         "AppleWebKit/537.36 (KHTML, like Gecko) "
    #         "Chrome/120.0.0.0 Safari/537.36"
    #     )
    # }
    # r = requests.get(URL, headers=headers, timeout=30)
    # r.raise_for_status()  # will show you 403/429 clearly if it still happens
    # tables = pd.read_html(response.text)
    # sp500_df = tables[0]
    # sp500_changes = tables[1] # TODO: might be useful later for organizing changes in the stock list
    
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()

    sp500_df = (
        pd.read_html(StringIO(response.text))[0]
    )
    sp500_df = sp500_df.rename(columns={
        "Symbol": "stock",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_sector"
    })
    sp500_df["ingested_at"] = datetime.now()
    sp500_df = sp500_df[["stock","name","sector","sub_sector","ingested_at"]]
    sp500_df.sort_values("stock")
    sp500_df["stock"] = sp500_df["stock"].str.replace(".", "-", regex=False)
    # Fetch symbols only
    for stock in stock_list:
        if stock not in sp500_df["stock"].values:
            print(f"PROBLEM! {stock} not in my stock_list file")
    return sp500_df

def ingest_all_sector_data(con):
    FAILED_SECTOR_LOG_PATH = "output/debug_failed_sector_data_ingestion.txt"
    stocks = read_stocks_to_fetch()
    if not stocks:
        raise ValueError("No stocks found.")
    # reset failure log each run (simple)
    with open(FAILED_SECTOR_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("error\n")
    print(f"Fetching GICS Sector Data...")
    try:
        sp500_sector_df = get_sp500_sectors()
        con.register("temp_sectors", sp500_sector_df)
        con.execute("INSERT OR IGNORE INTO stock_data SELECT * FROM temp_sectors")
        con.unregister("temp_sectors")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"FAILED fetching sector data: {err}")
        try:
            con.unregister("tmp_prices")
        except Exception:
            pass
        with open(FAILED_SECTOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{err}\n")
