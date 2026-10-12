"""
Pipeline unit tests — stage 3 daily preparation.

Run with:  pytest testing/test_pipeline.py -v

Each test catches a class of silent breakage:
  - Column deleted or renamed
  - Shift removed (leakage)
  - Reactions bleed onto non-earnings rows

Model B scoring (stage 4) is tested in testing/test_model_b.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.stage3 import stage3


# ── Fixture ───────────────────────────────────────────────────────────────────

def _build_stage2_df(n_days=250, seed=42, e_indices=(60, 120, 180, 220), trend_tail=None):
    """
    Minimal stage-2-like DataFrame: 2 stocks × n_days rows, one earnings event per
    index in e_indices. Earnings are spaced so every event has >= 5 price rows after
    it (needed for reaction_5d), and enough history before it for rolling features.

    trend_tail: optional (n, drift) — force a sustained drift over the final n rows of
    the FIRST stock.
    """
    rng  = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-02")
    # padded past n_days: e_indices may name a FUTURE event that has no price row yet,
    # which is what gives the last rows a non-null days_to_earnings, as in real data.
    dates = [base + pd.Timedelta(days=i) for i in range(n_days + 90)]

    rows = []
    for stock, sector in [("AAA", "Technology"), ("BBB", "Financials")]:
        rets = rng.normal(0.0003, 0.013, n_days)
        if trend_tail is not None and stock == "AAA":
            n_tail, drift = trend_tail
            rets[-n_tail:] = drift
        prices = 100.0 * np.cumprod(1 + rets)

        def _next_earnings_date(i):
            for ei in e_indices:
                if ei >= i:
                    return dates[ei]
            return pd.NaT

        for i in range(n_days):
            is_e = i in e_indices
            rows.append({
                "stock":               stock,
                "date":                dates[i],
                "price":               prices[i],
                "earnings_date":       _next_earnings_date(i),
                "sector":              sector,
                "sub_sector":          "Software",
                "surprise_percentage": float(rng.choice([4.0, -2.0, 7.0, -0.5])) if is_e else np.nan,
                "reported_eps":        1.5  if is_e else np.nan,
                "estimated_eps":       1.4  if is_e else np.nan,
                # IV / EPS-estimate columns that stage2 attaches — NaN is fine for tests
                "expected_move_pct":   np.nan,
                "atm_iv":              np.nan,
                "iv_snapshot_date":    pd.NaT,
                "eps_avg":             np.nan,
                "eps_high":            np.nan,
                "eps_low":             np.nan,
                "eps_num_analysts":    np.nan,
                "eps_dispersion":      np.nan,
                "eps_revision_momentum": np.nan,
                "eps_trend_7d":        np.nan,
                "eps_trend_30d":       np.nan,
                "eps_trend_60d":       np.nan,
                "eps_trend_90d":       np.nan,
                "eps_revisions_up_7d":   np.nan,
                "eps_revisions_down_7d": np.nan,
                "eps_revisions_up_30d":  np.nan,
                "eps_revisions_down_30d":np.nan,
                "revenue_avg":         np.nan,
                "revenue_high":        np.nan,
                "revenue_low":         np.nan,
                "eps_snapshot_date":   pd.NaT,
            })

    df = pd.DataFrame(rows)
    df["date"]          = pd.to_datetime(df["date"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df.sort_values(["stock", "date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def stage2_df():
    return _build_stage2_df()


@pytest.fixture(scope="session")
def stage3_df(stage2_df):
    return stage3(stage2_df)


# ── Stage 3: column existence ─────────────────────────────────────────────────

STAGE3_COLS = [
    "daily_ret",
    "vol_10d", "vol_30d", "vol_ratio_10_to_30",
    "days_to_earnings", "is_earnings_day", "is_earnings_week", "is_earnings_window",
    "reaction_1d", "reaction_3d", "reaction_5d",
    "abs_reaction_3d",
]

@pytest.mark.parametrize("col", STAGE3_COLS)
def test_stage3_column_exists(stage3_df, col):
    assert col in stage3_df.columns, f"Missing stage3 column: {col}"


RETIRED_COLS = ["drift_30d", "mom_5d", "reaction_entropy", "abs_reaction_p75",
                "sector_vol_30d", "pre_earnings_drift_z", "surprise_streak"]

def test_stage3_computes_no_retired_legacy_scoring_inputs(stage3_df):
    """The 0.3.1 scoring inputs were removed with that scorer; none may come back."""
    assert not [c for c in RETIRED_COLS if c in stage3_df.columns]


# ── No-leakage: shift(1) on rolling features ─────────────────────────────────

def test_daily_ret_first_row_nan(stage3_df):
    """pct_change() on first row per stock must be NaN."""
    for stock, grp in stage3_df.groupby("stock"):
        assert pd.isna(grp["daily_ret"].iloc[0]), f"{stock}: first daily_ret should be NaN"


def test_vol_uses_shift(stage3_df):
    """vol_10d uses rolling(10).shift(1) — first 10 rows per stock must be NaN."""
    for stock, grp in stage3_df.groupby("stock"):
        first_valid = grp["vol_10d"].first_valid_index()
        pos = grp.index.get_loc(first_valid)
        assert pos >= 10, f"{stock}: vol_10d went non-NaN too early (row {pos})"


# ── Earnings windows ──────────────────────────────────────────────────────────

def test_is_earnings_day_matches_date(stage3_df):
    """is_earnings_day == 1 iff date == earnings_date."""
    flagged   = stage3_df[stage3_df["is_earnings_day"] == 1]
    unflagged = stage3_df[stage3_df["is_earnings_day"] == 0]
    assert (flagged["date"] == flagged["earnings_date"]).all()
    assert (unflagged["date"] != unflagged["earnings_date"]).all() or unflagged["earnings_date"].isna().any()


def test_is_earnings_week_subset(stage3_df):
    """Every is_earnings_day row must also be is_earnings_week."""
    day_rows = stage3_df[stage3_df["is_earnings_day"] == 1]
    assert (day_rows["is_earnings_week"] == 1).all()


def test_days_to_earnings_zero_on_earnings_day(stage3_df):
    earnings_rows = stage3_df[stage3_df["is_earnings_day"] == 1]
    assert (earnings_rows["days_to_earnings"] == 0).all()


# ── Reaction features ─────────────────────────────────────────────────────────

def test_reactions_nan_on_non_earnings_days(stage3_df):
    """reaction_3d must be NaN on every non-earnings row."""
    non_earnings = stage3_df[stage3_df["is_earnings_day"] == 0]
    assert non_earnings["reaction_3d"].isna().all()


def test_reactions_present_on_earnings_days(stage3_df):
    """Earnings days with enough future data must have reaction_3d populated."""
    earnings = stage3_df[stage3_df["is_earnings_day"] == 1]
    assert earnings["reaction_3d"].notna().any(), "No earnings row has a reaction_3d value"


def test_reaction_3d_correct_value(stage3_df):
    """Spot-check: reaction_3d ≈ price[t+3] / price[t] - 1."""
    earnings = stage3_df[stage3_df["is_earnings_day"] == 1]
    for _, row in earnings.iterrows():
        stock    = row["stock"]
        t        = row["date"]
        stock_df = stage3_df[stage3_df["stock"] == stock].sort_values("date").reset_index(drop=True)
        pos      = stock_df.index[stock_df["date"] == t].tolist()
        if not pos or pos[0] + 3 >= len(stock_df):
            continue
        p0       = stock_df.loc[pos[0], "price"]
        p3       = stock_df.loc[pos[0] + 3, "price"]
        expected = p3 / p0 - 1
        assert abs(row["reaction_3d"] - expected) < 1e-9
        break   # one spot-check is enough


def test_abs_reaction_3d_nonnegative(stage3_df):
    vals = stage3_df["abs_reaction_3d"].dropna()
    assert (vals >= 0).all()


# ── Volatility ────────────────────────────────────────────────────────────────

def test_vol_nonnegative(stage3_df):
    for col in ("vol_10d", "vol_30d"):
        assert (stage3_df[col].dropna() >= 0).all(), f"{col} has negative values"
