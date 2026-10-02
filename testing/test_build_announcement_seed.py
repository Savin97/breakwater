"""Tests for `backfills/build_announcement_seed.py`.

Synthetic vendor records go through the real `research.massive.normalize`, so the
midnight and identity rules under test are the production ones, not re-implementations.
"""
import pandas as pd

from backfills.build_announcement_seed import (
    SEED_COLS, YF_OBSERVED_AT, YF_SOURCE_LABEL as YF_LABEL, YF_SOURCE_PARQUET, build,
    load_yfinance_seed)
from research.massive import normalize

SNAP = "earnings_20260910T180412Z"


def _record(i, ticker="AAA", date="2024-05-01", time="16:05:00", **kw):
    rec = {"benzinga_id": f"id{i}", "ticker": ticker, "company_name": "Fake Co",
           "date": date, "time": time, "date_status": "confirmed", "currency": "USD",
           "importance": 1, "fiscal_year": 2024, "fiscal_period": "Q1",
           "last_updated": "2024-05-02T10:00:00Z"}
    rec.update(kw)
    return rec


def _keys(*pairs):
    return pd.DataFrame([{"stock": s, "earnings_date": pd.Timestamp(d).date()}
                         for s, d in pairs])


def _yf(*rows):
    return pd.DataFrame([{"stock": s, "earnings_date": pd.Timestamp(d).date(),
                          "announce_ts_ny": pd.Timestamp(ts)} for s, d, ts in rows],
                        columns=["stock", "earnings_date", "announce_ts_ny"])


def _build(records, keys, yf=None):
    vendor = normalize.normalize(records, snapshot_id=SNAP)
    return build(vendor, SNAP, keys, yf if yf is not None else _yf())


def _row(seed, stock, date):
    hit = seed[(seed["stock"] == stock) & (seed["earnings_date"] == pd.Timestamp(date).date())]
    return None if hit.empty else hit.iloc[0]


def test_an_exact_date_match_carries_traceable_provenance():
    seed, stats = _build([_record(1, time="16:30:00")], _keys(("AAA", "2024-05-01")))
    assert list(seed.columns) == SEED_COLS
    row = _row(seed, "AAA", "2024-05-01")
    assert row["announce_ts_ny"] == pd.Timestamp("2024-05-01 16:30")
    assert row["announce_ts_source"] == f"massive_benzinga:{SNAP}:id1"
    # snapshot 18:04:12 UTC on an EDT date is 14:04:12 NY
    assert row["announce_ts_observed_at"] == pd.Timestamp("2026-09-10 14:04:12")
    assert stats["benzinga_rows"] == 1


def test_midnight_filler_is_never_seeded():
    seed, _ = _build([_record(1, time="00:00:00")], _keys(("AAA", "2024-05-01")))
    assert seed.empty


def test_a_one_day_date_disagreement_is_skipped_not_seeded():
    """Production anchors on OUR earnings_date and reads only the clock from the
    timestamp, so a vendor record a day off would anchor on the wrong session."""
    seed, stats = _build([_record(1, date="2026-02-18", time="16:05:00")],
                         _keys(("AAA", "2026-02-19")))
    assert stats["skipped_date_off_by_one"] == 1
    assert seed.empty


def test_an_identity_hazard_ticker_is_skipped_entirely():
    """A reused symbol (history gap >= 400 days) is not joined on today's ticker."""
    seed, stats = _build([_record(1, date="2014-05-01"), _record(2, date="2024-05-01")],
                         _keys(("AAA", "2014-05-01"), ("AAA", "2024-05-01")))
    assert stats["identity_hazard_tickers"] == 1
    assert seed.empty


def test_yfinance_fills_only_what_benzinga_did_not_time():
    keys = _keys(("AAA", "2024-05-01"), ("BBB", "2024-05-01"), ("CCC", "2024-05-01"))
    yf = _yf(("AAA", "2024-05-01", "2024-05-01 16:00"),     # Benzinga has it -> ignored
             ("BBB", "2024-05-01", "2024-05-01 07:00"),     # Benzinga lacks it -> used
             ("ZZZ", "2024-05-01", "2024-05-01 07:00"))     # not our event -> dropped
    seed, stats = _build([_record(1, ticker="AAA", time="16:30:00")], keys, yf)
    assert stats["yfinance_rows"] == 1 and len(seed) == 2
    assert _row(seed, "AAA", "2024-05-01")["announce_ts_ny"] == pd.Timestamp("2024-05-01 16:30")
    bbb = _row(seed, "BBB", "2024-05-01")
    assert bbb["announce_ts_ny"] == pd.Timestamp("2024-05-01 07:00")
    assert bbb["announce_ts_source"] == YF_LABEL
    assert bbb["announce_ts_observed_at"] == YF_OBSERVED_AT
    assert _row(seed, "CCC", "2024-05-01") is None


def test_the_seed_satisfies_the_loader_contract():
    """Naive NY timestamps and one row per event — what load_announcement_seed demands."""
    seed, _ = _build([_record(1), _record(2, ticker="BBB", time="07:00:00")],
                     _keys(("AAA", "2024-05-01"), ("BBB", "2024-05-01")))
    assert seed["announce_ts_ny"].dt.tz is None
    assert seed["announce_ts_observed_at"].dt.tz is None
    assert not seed.duplicated(["stock", "earnings_date"]).any()


def test_yfinance_seed_provenance_is_fixed():
    """These values are already stamped on rows in the DB; changing them would split
    one source into two."""
    assert YF_SOURCE_PARQUET == "audit/provider_timestamps.parquet"
    assert YF_LABEL == "audit_provider_timestamps_2026_09_05"
    assert YF_OBSERVED_AT == pd.Timestamp("2026-09-05")


def test_load_yfinance_seed_keeps_ny_wall_clock_and_drops_nulls_and_duplicates(tmp_path):
    ny = pd.to_datetime(["2024-05-01 06:30", "2024-01-31 16:05", "2024-05-01 16:00",
                         None]).tz_localize("America/New_York")
    raw = pd.DataFrame({
        "stock": ["AAA", "AAA", "AAA", "BBB"],
        "earnings_date": pd.to_datetime(["2024-05-01", "2024-01-31", "2024-05-01",
                                         "2024-05-01"]),
        "announce_ts_ny": ny,
        "extra": [1, 2, 3, 4],
    })
    path = tmp_path / "ts.parquet"
    raw.to_parquet(path)

    out = load_yfinance_seed(path)
    assert list(out.columns) == ["stock", "earnings_date", "announce_ts_ny"]
    assert out["announce_ts_ny"].dt.tz is None
    # wall clock kept across EST/EDT, the duplicate keeps the first row, the null is gone
    assert list(out["announce_ts_ny"]) == [pd.Timestamp("2024-05-01 06:30"),
                                           pd.Timestamp("2024-01-31 16:05")]
    assert list(out["earnings_date"]) == [pd.Timestamp("2024-05-01").date(),
                                          pd.Timestamp("2024-01-31").date()]
