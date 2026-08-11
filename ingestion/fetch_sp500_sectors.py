# ingestion/fetch_sp500_sectors.py
from utilities.data_utilities import read_stocks_to_fetch
from utilities.output_utilities import get_run_logs_dir
import os
import pandas as pd, requests
from datetime import datetime
from io import StringIO
from pathlib import Path

RENAME_MAP_PATH = "data/ticker_renames.csv"

def get_sp500_sectors():
    """
        Fetches company/sector/sub-sector data for all current S&P 500 stocks
        from Wikipedia (GICS data).
        Returns a df: stock | company_name | sector | sub_sector | ingested_at
    """
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()

    sp500_df = pd.read_html(StringIO(response.text))[0]
    sp500_df = sp500_df.rename(columns={
        "Symbol": "stock",
        "Security": "company_name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_sector"
    })
    sp500_df["stock"] = sp500_df["stock"].str.replace(".", "-", regex=False)
    sp500_df["ingested_at"] = datetime.now()
    return sp500_df[["stock", "company_name", "sector", "sub_sector", "ingested_at"]]

def _load_rename_map() -> dict[str, str]:
    """old_ticker -> new_ticker, from the manually-maintained data/ticker_renames.csv"""
    path = Path(RENAME_MAP_PATH)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return dict(zip(
        df["old_ticker"].astype(str).str.upper(),
        df["new_ticker"].astype(str).str.upper(),
    ))

def reconcile_stock_status(universe: set[str], sp500_df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """
        Classifies every ticker in universe against current S&P 500 membership:
        - in sp500_df -> status='active', fresh company/sector data
        - in rename_map (old_ticker) -> status='inactive', reason='renamed → new_ticker'
        - otherwise -> status='inactive', reason='removed from index'
        Returns a df: stock | company_name | sector | sub_sector | status | reason
        (company/sector/sub_sector are None for inactive tickers — the caller
        preserves their last-known values rather than nulling them out).
    """
    wiki_lookup = sp500_df.set_index("stock")[["company_name", "sector", "sub_sector"]].to_dict("index")

    rows = []
    for stock in sorted(universe):
        if stock in wiki_lookup:
            info = wiki_lookup[stock]
            rows.append((stock, info["company_name"], info["sector"], info["sub_sector"], "active", None))
        elif stock in rename_map:
            rows.append((stock, None, None, None, "inactive", f"renamed → {rename_map[stock]}"))
        else:
            rows.append((stock, None, None, None, "inactive", "removed from index — delisted/merged/unconfirmed"))

    return pd.DataFrame(rows, columns=["stock", "company_name", "sector", "sub_sector", "status", "reason"])

def ingest_all_sector_data(con):
    FAILED_SECTOR_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_sector_data_ingestion.txt")
    stocks = read_stocks_to_fetch()
    if not stocks:
        raise ValueError("No stocks found.")
    with open(FAILED_SECTOR_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("error\n")
    print("Fetching GICS Sector Data...")
    try:
        sp500_df = get_sp500_sectors()
        existing_tickers = {
            row[0] for row in con.execute("SELECT stock FROM stock_data").fetchall()
        }
        universe = set(stocks) | existing_tickers
        rename_map = _load_rename_map()

        upsert_df = reconcile_stock_status(universe, sp500_df, rename_map)
        upsert_df["ingested_at"] = datetime.now()

        con.register("temp_sectors", upsert_df)
        con.execute("""
            INSERT INTO stock_data (stock, company_name, sector, sub_sector, status, reason, ingested_at)
            SELECT stock, company_name, sector, sub_sector, status, reason, ingested_at FROM temp_sectors
            ON CONFLICT (stock) DO UPDATE SET
                company_name = COALESCE(EXCLUDED.company_name, stock_data.company_name),
                sector       = COALESCE(EXCLUDED.sector, stock_data.sector),
                sub_sector   = COALESCE(EXCLUDED.sub_sector, stock_data.sub_sector),
                status       = EXCLUDED.status,
                reason       = EXCLUDED.reason,
                ingested_at  = EXCLUDED.ingested_at
        """)
        con.unregister("temp_sectors")

        counts = upsert_df["status"].value_counts()
        n_renamed = upsert_df["reason"].str.startswith("renamed", na=False).sum()
        n_removed = counts.get("inactive", 0) - n_renamed
        print(f"Sector reconciliation: {counts.get('active', 0)} active, {n_renamed} renamed (inactive), {n_removed} removed/unconfirmed (inactive)")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"FAILED fetching sector data: {err}")
        try:
            con.unregister("temp_sectors")
        except Exception:
            pass
        with open(FAILED_SECTOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{err}\n")
