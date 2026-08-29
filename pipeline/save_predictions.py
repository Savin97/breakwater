# pipeline/save_predictions.py
import subprocess
import duckdb
import pandas as pd
from datetime import date

from config import DB_PATH, MODEL_VERSION
from utilities.db_utilities import create_predictions_table_if_not_exists


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def save_predictions_snapshot(df: pd.DataFrame) -> None:
    """Persist a snapshot of this run's upcoming predictions to the DB so backtesting
    can later compare predicted tier/score against realized reactions. Called once per
    pipeline run (stage5); each run's rows are keyed by today's date, so predictions
    made further out from an earnings event stack up alongside later, closer-in ones
    for the same (stock, earnings_date) instead of overwriting them.
    """
    today = pd.Timestamp(date.today())
    latest = df.sort_values("date").groupby("stock").last().reset_index()
    upcoming = latest[latest["earnings_date"] >= today].copy()

    if upcoming.empty:
        print("No upcoming earnings events — nothing to save to predictions table.")
        return

    snap = pd.DataFrame({
        "prediction_asof_date":    today.date(),
        "week_start":              (upcoming["earnings_date"]
                                     - pd.to_timedelta(upcoming["earnings_date"].dt.weekday, unit="D")).dt.date,
        "stock":                   upcoming["stock"],
        "earnings_date":           upcoming["earnings_date"].dt.date,
        "tier":                    upcoming["earnings_explosiveness_bucket"].astype(str),
        "risk_score":              upcoming["risk_score"],
        "is_high_conviction":      upcoming["is_high_conviction"],
        "pre_earnings_drift_flag": upcoming["pre_earnings_drift_flag"],
        "surprise_momentum_flag":  upcoming["surprise_momentum_flag"],
        "model_version":           MODEL_VERSION,
        "git_commit":              _get_git_commit(),
        "ingested_at":             pd.Timestamp.now(),
    })

    con = duckdb.connect(DB_PATH)
    create_predictions_table_if_not_exists(con)
    col_list = ", ".join(snap.columns)
    con.register("tmp_predictions", snap)
    con.execute(f"INSERT INTO predictions ({col_list}) SELECT {col_list} FROM tmp_predictions ON CONFLICT DO NOTHING")
    con.unregister("tmp_predictions")
    con.close()
    print(f"Saved {len(snap)} prediction rows → predictions table (asof {today.date()})")
