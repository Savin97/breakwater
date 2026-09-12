import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO, TARGET_AVAILABLE
from feature_engineering.event_features import event_stock_bucket_lift_values
from research.phase3_target_rebuild import (
    _missing_aware_lift,
    _phase3_proxy_session_dates,
    _target_history_features,
    evaluate,
    identity_hazards,
    match_benzinga_timing,
    reanchor_to_observed_timestamp,
)


def _vendor_row(
    ticker,
    date,
    ts,
    benzinga_id,
    *,
    confirmed=True,
    company="Fake Co",
    updated="2026-09-10T12:00:00Z",
):
    stamp = pd.Timestamp(ts)
    return {
        "ticker_norm": ticker,
        "report_date": pd.Timestamp(date),
        "announce_ts_vendor": stamp,
        "announce_window": "AMC" if stamp.hour >= 16 else "BMO",
        "time_usable": True,
        "is_confirmed": confirmed,
        "last_updated_utc": pd.Timestamp(updated),
        "benzinga_id": benzinga_id,
        "date_status_norm": "confirmed" if confirmed else "projected",
        "company_name_norm": company.casefold(),
    }


def test_matcher_accepts_unique_class_share_spelling_alias():
    keys = pd.DataFrame({
        "stock": ["BF-B"],
        "earnings_date": [pd.Timestamp("2025-05-01")],
    })
    vendor = pd.DataFrame([
        _vendor_row("BF.B", "2025-05-01", "2025-05-01 16:05", "x1"),
    ])
    timing, audit = match_benzinga_timing(vendor, keys, snapshot_id="earnings_20260910T180412Z")
    assert len(timing) == 1
    assert timing.iloc[0]["stock"] == "BF-B"
    assert audit.iloc[0]["match_status"] == "matched"
    assert audit.iloc[0]["vendor_ticker"] == "BF.B"
    assert timing.iloc[0]["announce_ts_observed_at"] == pd.Timestamp("2026-09-10 14:04:12")


def test_matcher_rejects_conflicting_duplicate_timestamps():
    keys = pd.DataFrame({
        "stock": ["AAPL"],
        "earnings_date": [pd.Timestamp("2025-05-01")],
    })
    vendor = pd.DataFrame([
        _vendor_row("AAPL", "2025-05-01", "2025-05-01 16:05", "x1"),
        _vendor_row("AAPL", "2025-05-01", "2025-05-01 17:05", "x2"),
    ])
    timing, audit = match_benzinga_timing(vendor, keys)
    assert timing.empty
    assert audit.iloc[0]["match_status"] == "ambiguous_timestamp"


def test_matcher_keeps_vendor_calendar_date_on_near_date_match():
    keys = pd.DataFrame({
        "stock": ["XYZ"],
        "earnings_date": [pd.Timestamp("2024-01-08")],
    })
    vendor = pd.DataFrame([
        _vendor_row("XYZ", "2024-01-07", "2024-01-07 16:05", "x1"),
    ])
    timing, audit = match_benzinga_timing(vendor, keys, tolerance_days=1)
    assert len(timing) == 1
    assert timing.iloc[0]["earnings_date"] == pd.Timestamp("2024-01-08")
    assert timing.iloc[0]["announce_ts_ny"] == pd.Timestamp("2024-01-07 16:05")
    assert audit.iloc[0]["date_delta_days"] == -1


def test_identity_hazards_flag_reused_or_discontinuous_histories():
    vendor = pd.DataFrame([
        _vendor_row("SAFE", "2023-01-01", "2023-01-01 16:05", "s1", company="Safe Inc"),
        _vendor_row("SAFE", "2023-04-01", "2023-04-01 16:05", "s2", company="Safe Inc"),
        _vendor_row("GAP", "2012-01-01", "2012-01-01 16:05", "g1", company="Gap Co"),
        _vendor_row("GAP", "2023-01-01", "2023-01-01 16:05", "g2", company="Gap Co"),
        _vendor_row("NAME", "2023-01-01", "2023-01-01 16:05", "n1", company="Old Co"),
        _vendor_row("NAME", "2023-04-01", "2023-04-01 16:05", "n2", company="New Co"),
    ])
    hazards = identity_hazards(vendor, pd.Index(["SAFE", "GAP", "NAME"]))
    assert set(hazards["stock"]) == {"GAP", "NAME"}
    assert "history_gap" in hazards.set_index("stock").loc["GAP", "reason"]
    assert "multiple_company_names" in hazards.set_index("stock").loc["NAME", "reason"]


def test_matcher_excludes_identity_hazard_instead_of_guessing():
    keys = pd.DataFrame({
        "stock": ["SNDK"],
        "earnings_date": [pd.Timestamp("2025-05-01")],
    })
    vendor = pd.DataFrame([
        _vendor_row("SNDK", "2025-05-01", "2025-05-01 16:05", "x1"),
    ])
    timing, audit = match_benzinga_timing(vendor, keys, excluded_stocks={"SNDK"})
    assert timing.empty
    assert audit.iloc[0]["match_status"] == "identity_hazard"


def _daily_sessions():
    return pd.DataFrame({
        "stock": ["XYZ"] * 5,
        "date": pd.to_datetime([
            "2024-01-04",  # Thu
            "2024-01-05",  # Fri
            "2024-01-08",  # Mon
            "2024-01-09",  # Tue
            "2024-01-10",  # Wed
        ]),
        "price": [99.0, 100.0, 110.0, 111.0, 112.0],
    })


def test_proxy_session_date_handles_session_and_weekend_clocks():
    daily = _daily_sessions()
    events = pd.DataFrame({
        "announce_ts_ny": pd.to_datetime([
            "2024-01-08 06:00",  # Monday BMO -> proxy Monday -> anchor Friday
            "2024-01-08 16:05",  # Monday AMC -> proxy Monday -> anchor Monday
            "2024-01-06 06:00",  # Saturday BMO -> proxy Monday -> anchor Friday
            "2024-01-06 16:05",  # Saturday AMC -> proxy Friday -> anchor Friday
        ]),
        "announce_window": [BMO, AMC, BMO, AMC],
    })
    got = _phase3_proxy_session_dates(events, daily)
    assert got.tolist() == [
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-05"),
    ]


def test_reanchor_weekend_announcement_uses_last_close_before_news():
    daily = _daily_sessions()
    events = pd.DataFrame({
        "stock": ["XYZ"],
        "earnings_date": [pd.Timestamp("2024-01-08")],
        "announce_ts_ny": [pd.Timestamp("2024-01-06 16:05")],
        "announce_window": [AMC],
        "is_pending": [False],
    })
    got = reanchor_to_observed_timestamp(events, daily)
    assert got.iloc[0]["phase3_announce_date"] == pd.Timestamp("2024-01-06")
    assert got.iloc[0]["anchor_date"] == pd.Timestamp("2024-01-05")
    assert got.iloc[0]["reaction_1d_anchored_status"] == TARGET_AVAILABLE
    assert np.isclose(got.iloc[0]["reaction_1d_anchored"], 0.10)


def test_target_history_counts_resolved_reactions_not_unresolved_rows():
    n = 32
    events = pd.DataFrame({
        "stock": ["AAA"] * n,
        "earnings_date": pd.date_range("2016-01-01", periods=n, freq="90D"),
        "is_pending": [False] * n,
        "reaction_3d_anchored": np.linspace(-0.03, 0.12, n),
        "reaction_1d_anchored": np.linspace(-0.01, 0.04, n),
    })
    events.loc[5, "reaction_3d_anchored"] = np.nan
    events.loc[12, "reaction_3d_anchored"] = np.nan
    hist = _target_history_features(events)
    assert hist.loc[30, "phase3_n_prior_resolved_reactions"] == 28
    assert pd.notna(hist.loc[30, "phase3_abs_reaction_p75_rolling"])


def test_entropy_keeps_the_legacy_1d_fallback_when_corrected_3d_is_missing():
    n = 10
    events = pd.DataFrame({
        "stock": ["AAA"] * n,
        "earnings_date": pd.date_range("2020-01-01", periods=n, freq="90D"),
        "is_pending": [False] * n,
        "reaction_3d_anchored": np.linspace(0.01, 0.10, n),
        "reaction_1d_anchored": np.linspace(0.01, 0.05, n),
    })
    events.loc[2, "reaction_3d_anchored"] = np.nan
    hist = _target_history_features(events)
    assert hist.loc[9, "phase3_n_prior_resolved_reactions"] == 8
    assert hist.loc[9, "phase3_n_prior_entropy_reactions"] == 9
    assert pd.notna(hist.loc[9, "phase3_reaction_entropy"])


def test_missing_aware_lift_is_identical_to_legacy_when_every_outcome_is_observed():
    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    bucket = pd.Categorical(
        ["Normal"] * 20 + ["Elevated"] * 20,
        categories=["Normal", "Elevated", "High Alert"],
        ordered=True,
    )
    y = pd.Series(([0, 0, 1, 0, 0] * 8), dtype=float)
    base = pd.DataFrame({
        "stock": ["AAA"] * n,
        "date": dates,
        "earnings_date": dates,
        "phase3_announce_date": dates,
        "is_pending": [False] * n,
        "is_extreme_reaction": y,
        "bucket": bucket,
    })
    legacy = event_stock_bucket_lift_values(
        base[["stock", "date", "is_pending", "is_extreme_reaction", "bucket"]]
        .rename(columns={"bucket": "earnings_explosiveness_bucket"})
    )
    corrected = _missing_aware_lift(base, "bucket", "is_extreme_reaction")
    assert np.allclose(legacy.to_numpy(), corrected.to_numpy(), equal_nan=True)


def test_evaluation_uses_the_same_rows_for_all_three_variants():
    n = 30
    dates = pd.date_range("2020-01-01", periods=n, freq="30D")
    buckets = pd.Categorical(
        ["Normal"] * 10 + ["Elevated"] * 10 + ["High Alert"] * 10,
        categories=["Normal", "Elevated", "High Alert"],
        ordered=True,
    )
    events = pd.DataFrame({
        "is_pending": [False] * n,
        "earnings_date": dates,
        "phase3_announce_date": dates,
        "abs_reaction_3d": np.linspace(0.01, 0.12, n),
        "abs_reaction_3d_anchored": np.linspace(0.02, 0.13, n),
        "reaction_3d_anchored_status": [TARGET_AVAILABLE] * n,
        "risk_score": np.linspace(20, 95, n),
        "earnings_explosiveness_bucket": buckets,
        "phase3_risk_score": np.linspace(22, 97, n),
        "phase3_bucket": buckets,
        "phase3_n_prior_resolved_reactions": np.arange(n),
        "announce_window": [AMC] * n,
    })
    metrics, shift, _ = evaluate(events, start="2020-01-01", end="2025-12-31")
    all_rows = metrics[metrics["cohort"].eq("all_resolved")]
    assert len(all_rows) == 3
    assert all_rows["n"].nunique() == 1
    assert shift["n_model_comparison_common"] == all_rows.iloc[0]["n"]
