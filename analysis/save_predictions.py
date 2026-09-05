# analysis/save_predictions.py
import subprocess
import duckdb
import pandas as pd
from datetime import date

from config import PREDICTIONS_DB_PATH, MODEL_VERSION
from utilities.db_utilities import create_predictions_table_if_not_exists
from utilities.data_utilities import work_week_window


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def save_predictions_snapshot(df: pd.DataFrame, weeks: int = 1,
                              current_week: bool = False) -> None:
    """Persist the calls we published so backtesting can compare the tier/score we sent
    against the realized reaction. Called once per pipeline run (stage5).

    Selects on the SAME window as the weekly digest — a whole Mon-Fri work week, see
    work_week_window — so every call the email makes is in the table. The table is the
    wider record: it keeps all tiers, while the email shows only High Alert / Elevated. These were
    computed separately before and drifted: a Friday run mailed four calls (ORCL, ADBE,
    COO, CPRT) that this table never recorded, because its window was today..Sunday and
    nothing reports at a weekend.

    The lower bound is clamped to today so a mid-week or --current-week run cannot write
    a "prediction" for an event that already reported, which would leak hindsight into
    any backtest of this table. Rows written earlier for those events stay untouched, and
    the predictions_first_call view picks the earliest call per event regardless.
    """
    today = pd.Timestamp(date.today())
    window_start, window_end = work_week_window(weeks=weeks, current_week=current_week)
    start = max(window_start, today)          # never record an event that already reported
    run_week_start = today - pd.Timedelta(days=today.weekday())      # Monday of this week

    latest = df.sort_values("date").groupby("stock").last().reset_index()
    upcoming = latest[
        (latest["earnings_date"] >= start) & (latest["earnings_date"] <= window_end)
    ].copy()

    if upcoming.empty:
        print(f"No earnings events between {start.date()} and {window_end.date()} — "
              "nothing to save to predictions table.")
        return

    snap = pd.DataFrame({
        "prediction_asof_date":    today.date(),
        # Monday of the week we made the call, vs week_start = Monday of the week the
        # company reports. run_week records WHEN the call was made; it is no longer a
        # grouping key — predictions_first_call dedups on the event itself.
        "run_week":                run_week_start.date(),
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

    con = duckdb.connect(PREDICTIONS_DB_PATH)
    create_predictions_table_if_not_exists(con)
    col_list = ", ".join(snap.columns)
    key = ("stock", "earnings_date", "prediction_asof_date")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in snap.columns if c not in key)
    con.register("tmp_predictions", snap)
    # Upsert, not DO NOTHING: re-scoring the same day after a fix must correct the row.
    # With DO NOTHING the first run of the day wins, so a broken snapshot would be the
    # one that survived. A run on a later date has a new asof date and inserts normally.
    con.execute(
        f"INSERT INTO predictions ({col_list}) SELECT {col_list} FROM tmp_predictions "
        f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}"
    )
    con.unregister("tmp_predictions")
    con.close()
    print(f"Saved {len(snap)} prediction rows → predictions table "
          f"(asof {today.date()}, week ending {week_end.date()})")
