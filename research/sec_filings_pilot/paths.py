"""Where the SEC pilot keeps its data.

    data/vendor/sec/snapshots/<snapshot_id>/   immutable snapshot of SEC's MUTABLE endpoints
                                               (company_tickers.json, submissions/*.json)
    data/vendor/sec/archives/<cik>/<acc>/      filing documents and index pages, which SEC
                                               never changes once disseminated
    output/sec_filings_pilot/                  small derived audit tables

`data/vendor/` and `output/` are gitignored. Raw SEC documents are public, but thousands of
them do not belong in git, and nothing under `data/vendor/` may become a production input
(`testing/test_massive_earnings.py` guards production modules against naming `vendor/`).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SEC_ROOT = REPO_ROOT / "data" / "vendor" / "sec"
SNAPSHOT_ROOT = SEC_ROOT / "snapshots"
ARCHIVE_ROOT = SEC_ROOT / "archives"
ARCHIVE_MANIFEST = SEC_ROOT / "archive_manifest.jsonl"
ARCHIVE_CHANGES = SEC_ROOT / "archive_changes.jsonl"

OUT = REPO_ROOT / "output" / "sec_filings_pilot"

# Breakwater inputs, read-only.
# The event population and its call cutoffs. Originally Phase 3's frame
# (`output/phase3_refit/feature_frame.parquet`, built 2026-09-28 from Phase 3's events of
# 2026-09-12); since the 2026-10-02 refresh, the same frame rebuilt by the same code on the
# date-corrected data (`research/sec_features/frame.py`). RESULTS.md §12 compares the two.
FEATURE_FRAME_PHASE3 = REPO_ROOT / "output" / "phase3_refit" / "feature_frame.parquet"
FEATURE_FRAME = REPO_ROOT / "output" / "sec_features" / "feature_frame_current.parquet"
EVENTS_DF = REPO_ROOT / "output" / "events_df.parquet"
FULL_DF = REPO_ROOT / "output" / "full_df.parquet"
STOCK_INFO = REPO_ROOT / "data" / "sp500_full_info.csv"
TICKER_RENAMES = REPO_ROOT / "data" / "ticker_renames.csv"
IDENTITY_HAZARDS = REPO_ROOT / "output" / "phase3_target_rebuild" / "identity_hazards.csv"
