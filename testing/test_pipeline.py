"""
Pipeline unit tests — stage3 feature engineering and stage4 scoring.

Run with:  pytest testing/test_pipeline.py -v

Each test catches a class of silent breakage:
  - Column deleted or renamed
  - Shift removed (leakage)
  - Score drifts outside [0, 100]
  - Reactions bleed onto non-earnings rows
  - Bucket labels changed
  - High-conviction logic inverted
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import INCREMENTAL_LOOKBACK_DAYS
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from scoring.scoring_features import engineer_high_conviction


# ── Fixture ───────────────────────────────────────────────────────────────────

def _build_stage2_df(n_days=250, seed=42, e_indices=(60, 120, 180, 220), trend_tail=None):
    """
    Minimal stage-2-like DataFrame: 2 stocks × n_days rows, one earnings event per
    index in e_indices. Earnings are spaced so every event has >= 5 price rows after
    it (needed for reaction_5d), and enough history before it for rolling features.

    trend_tail: optional (n, drift) — force a sustained drift over the final n rows of
    the FIRST stock. Used by the incremental-parity test to guarantee that stock ends
    with a pre-earnings drift flag, so the comparison cannot pass vacuously.
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
    return stage3(stage2_df, incremental=False)


@pytest.fixture(scope="session")
def stage4_df(stage3_df):
    return stage4(stage3_df, incremental=False)


# ── Stage 3: column existence ─────────────────────────────────────────────────

STAGE3_COLS = [
    "daily_ret",
    "drift_30d", "drift_60d",
    "vol_10d", "vol_30d", "vol_ratio_10_to_30",
    "mom_5d", "mom_20d",
    "days_to_earnings", "is_earnings_day", "is_earnings_week", "is_earnings_window",
    "reaction_1d", "reaction_3d", "reaction_5d",
    "abs_reaction_3d",
    "is_up", "is_down", "is_nochange",
    "reaction_std", "reaction_entropy", "directional_bias",
    "abs_reaction_median", "abs_reaction_p75",
    "abs_reaction_p75_rolling", "abs_reaction_p90_rolling",
    "sector_drift_60d", "sector_vol_10d", "sector_vol_30d",
    "stock_vs_sector_vol", "sector_earnings_density",
]

@pytest.mark.parametrize("col", STAGE3_COLS)
def test_stage3_column_exists(stage3_df, col):
    assert col in stage3_df.columns, f"Missing stage3 column: {col}"


# ── Stage 4: column existence ─────────────────────────────────────────────────

STAGE4_COLS = [
    "is_large_reaction", "is_extreme_reaction",
    "vol_stress_elevated", "vol_stress_extreme",
    "sector_vol_stress_high",
    "earnings_explosiveness_z", "earnings_tail_z",
    "proximity_score", "vol_expansion_score", "momentum_fragility_score",
    "earnings_explosiveness_score", "earnings_explosiveness_bucket",
    "earnings_move_bucket",
    "surprise_momentum_flag", "pre_earnings_drift_flag",
    "risk_score",
    "is_high_conviction",
]

@pytest.mark.parametrize("col", STAGE4_COLS)
def test_stage4_column_exists(stage4_df, col):
    assert col in stage4_df.columns, f"Missing stage4 column: {col}"


# ── No-leakage: shift(1) on rolling features ─────────────────────────────────

def test_daily_ret_first_row_nan(stage3_df):
    """pct_change() on first row per stock must be NaN."""
    for stock, grp in stage3_df.groupby("stock"):
        assert pd.isna(grp["daily_ret"].iloc[0]), f"{stock}: first daily_ret should be NaN"


def test_drift_uses_shift(stage3_df):
    """drift_30d uses rolling(30).shift(1) — first 30 rows per stock must be NaN."""
    for stock, grp in stage3_df.groupby("stock"):
        first_valid = grp["drift_30d"].first_valid_index()
        pos = grp.index.get_loc(first_valid)
        assert pos >= 30, f"{stock}: drift_30d went non-NaN too early (row {pos})"


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


# ── Scoring range & bucket labels ────────────────────────────────────────────

def test_risk_score_range(stage4_df):
    vals = stage4_df["risk_score"].dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_explosiveness_score_range(stage4_df):
    vals = stage4_df["earnings_explosiveness_score"].dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_proximity_score_range(stage4_df):
    vals = stage4_df["proximity_score"].dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_bucket_valid_labels(stage4_df):
    valid  = {"Normal", "Elevated", "High Alert"}
    actual = set(stage4_df["earnings_explosiveness_bucket"].dropna().astype(str).unique())
    assert actual <= valid, f"Unexpected bucket labels: {actual - valid}"


def test_large_reaction_binary(stage4_df):
    vals = stage4_df["is_large_reaction"].dropna().unique()
    assert set(vals) <= {0, 1}


def test_extreme_subset_of_large(stage4_df):
    """Every extreme reaction must also be a large reaction."""
    extreme = stage4_df[stage4_df["is_extreme_reaction"] == 1]
    assert (extreme["is_large_reaction"] == 1).all()


# ── High conviction logic ─────────────────────────────────────────────────────

def _carried_bucket(df):
    """The bucket HC is actually defined against. earnings_explosiveness_bucket is only
    materialised on earnings-day rows, so on the pre-earnings rows — where HC matters most,
    since that is what the upcoming-events views read — the stored value is NaN and the
    stock's tier is the last completed event's. Mirrors engineer_high_conviction."""
    return df.groupby("stock")["earnings_explosiveness_bucket"].ffill()


def test_high_conviction_implies_high_alert(stage4_df):
    hc = stage4_df["is_high_conviction"] == True
    assert (_carried_bucket(stage4_df)[hc] == "High Alert").all()


def test_high_conviction_implies_drift_flag(stage4_df):
    hc = stage4_df[stage4_df["is_high_conviction"] == True]
    assert (hc["pre_earnings_drift_flag"] != "").all()


def test_non_high_alert_never_high_conviction(stage4_df):
    not_ha = _carried_bucket(stage4_df) != "High Alert"
    assert not stage4_df.loc[not_ha, "is_high_conviction"].any()


# The stage4_df fixture is synthetic and currently yields no High Alert events, so the
# three tests above are vacuously true on it. These exercise engineer_high_conviction
# directly against a hand-built frame, so the carry-forward behaviour is actually asserted.

def test_high_conviction_carries_across_non_earnings_rows():
    """A High Alert event leaves NaN buckets on the days that follow it. HC must still
    fire on those rows when a drift flag is present — this is the upcoming-event case."""
    df = pd.DataFrame({
        "stock": ["AAA"] * 4,
        "date": pd.date_range("2024-01-01", periods=4, freq="D"),
        "earnings_explosiveness_bucket": ["High Alert", np.nan, np.nan, np.nan],
        "pre_earnings_drift_flag": ["", "", "Extended", ""],
    })
    out = engineer_high_conviction(df)
    assert list(out["is_high_conviction"]) == [False, False, True, False]


def test_high_conviction_not_carried_from_lower_tier():
    """The carry must not promote a stock whose last event was below High Alert."""
    df = pd.DataFrame({
        "stock": ["AAA"] * 3,
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "earnings_explosiveness_bucket": ["Elevated", np.nan, np.nan],
        "pre_earnings_drift_flag": ["", "Extended", "Compressed"],
    })
    assert not engineer_high_conviction(df)["is_high_conviction"].any()


def test_high_conviction_does_not_leak_across_stocks():
    """ffill runs per stock — BBB must not inherit AAA's High Alert."""
    df = pd.DataFrame({
        "stock": ["AAA", "AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
        "earnings_explosiveness_bucket": ["High Alert", np.nan, np.nan, np.nan],
        "pre_earnings_drift_flag": ["", "Extended", "Extended", "Extended"],
    })
    out = engineer_high_conviction(df)
    assert list(out["is_high_conviction"]) == [False, True, False, False]


# ── Incremental vs full parity ────────────────────────────────────────────────
# The droplet scores through the incremental path while we publish from the full one.
# If they disagree, the dashboard and the PDFs describe the same stock differently and
# nothing anywhere reports it — that divergence went unnoticed from Jun to Aug 2026.

INCREMENTAL_PARITY_COLS = [
    "earnings_explosiveness_bucket",
    "earnings_explosiveness_score",
    "risk_score",
    "is_high_conviction",
    "pre_earnings_drift_flag",
    "surprise_momentum_flag",
]


def _full_then_incremental(tmp_path, monkeypatch):
    """Score one dataset both ways and return (full, incremental) latest-row-per-stock."""
    n_days = 900
    df = _build_stage2_df(
        n_days=n_days,
        # ~quarterly history, plus an UPCOMING event 21 days past the last price row
        # so the latest row sits in the 1-60 day pre-earnings window the flag needs.
        e_indices=tuple(range(120, n_days - 10, 63)) + (n_days + 20,),
        trend_tail=(45, 0.006),                          # guarantees a drift flag
    )
    full = stage4(stage3(df, incremental=False), incremental=False)

    # stage3(incremental=True) reads output/full_df.parquet relative to cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    full.to_parquet(tmp_path / "output" / "full_df.parquet", index=False)

    window_start = df["date"].max() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    inc = stage4(stage3(df[df["date"] >= window_start].copy(), incremental=True),
                 incremental=True)

    return (full.sort_values("date").groupby("stock").last(),
            inc.sort_values("date").groupby("stock").last())


def test_incremental_parity_fixture_is_not_vacuous(tmp_path, monkeypatch):
    """Guard the test below. If the full path emits no flags there is nothing to
    disagree about, and parity would pass while proving nothing — which is exactly how
    the HC tests stayed green over a real bug."""
    full, _ = _full_then_incremental(tmp_path, monkeypatch)
    assert (full["pre_earnings_drift_flag"] != "").any(), (
        "fixture produces no pre-earnings drift flag; parity test would be vacuous"
    )


def test_incremental_matches_full_on_latest_row(tmp_path, monkeypatch):
    """Both DAILY paths must describe a stock identically on the latest row per stock.

    NOTE: since the Phase 1 event-frame rebuild, no upcoming-events consumer reads this
    row any more — they read the pending rows of pipeline/events.py (see
    testing/test_event_frame.py). This remains a parity test of the incremental daily
    scoring path against the full one, nothing more. groupby().last() is used here
    deliberately, to compare the two paths on the exact mechanism the incremental cache
    was built around; production code must not use it (test_event_frame invariant 9).
    """
    full, inc = _full_then_incremental(tmp_path, monkeypatch)
    assert list(full.index) == list(inc.index)

    mismatched = {}
    for col in INCREMENTAL_PARITY_COLS:
        a, b = full[col].astype(str), inc[col].astype(str)
        if not a.equals(b):
            mismatched[col] = {s: (a[s], b[s]) for s in a.index[a != b]}
    assert not mismatched, f"full vs incremental disagree: {mismatched}"
