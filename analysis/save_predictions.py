# pipeline/save_predictions.py
import subprocess
import duckdb
import pandas as pd
from datetime import date

from config import PREDICTIONS_DB_PATH, MODEL_VERSION
from utilities.db_utilities import create_predictions_table_if_not_exists


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def save_predictions_snapshot(df: pd.DataFrame) -> None:
    """Persist THIS WEEK's predictions so backtesting can compare the tier/score we
    published against the realized reaction. Called once per pipeline run (stage5).

    Scope is deliberately the run week only — the product is "here is the week's
    earnings risk", so the table holds exactly the calls we made, nothing else. Events
    further out are still scored in full_df/upcoming_df; they land here in the week
    they actually fall in.

    The window is today..Sunday of the run week, not Monday..Sunday: starting at today
    means a mid-week re-run cannot write a "prediction" for an event that already
    reported, which would otherwise leak hindsight into any backtest of this table.
    Monday's rows for those earlier events are already stored and stay untouched.
    """
    today = pd.Timestamp(date.today())
    run_week_start = today - pd.Timedelta(days=today.weekday())      # Monday of this week
    week_end = run_week_start + pd.Timedelta(days=6)                 # through Sunday: a
    # handful of earnings_dates land on a weekend (bad source data), and cutting at
    # Friday would drop them from the record entirely rather than flagging them.

    latest = df.sort_values("date").groupby("stock").last().reset_index()
    upcoming = latest[
        (latest["earnings_date"] >= today) & (latest["earnings_date"] <= week_end)
    ].copy()

    if upcoming.empty:
        print(f"No earnings events between {today.date()} and {week_end.date()} — "
              "nothing to save to predictions table.")
        return

    snap = pd.DataFrame({
        "prediction_asof_date":    today.date(),
        # Monday of the week we made the call, vs week_start = Monday of the week the
        # company reports. run_week is what predictions_week_open groups on.
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
