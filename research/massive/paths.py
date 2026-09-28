"""Where the Massive/Benzinga research data lives — and why it is not in `db/` or `output/`.

Raw vendor payloads are kept COMPLETELY SEPARATE from Breakwater production data:

    data/vendor/massive/earnings/<snapshot_id>/   immutable raw snapshot (pages + manifest)
    data/vendor/massive/normalized/               normalized research parquet
    data/vendor/massive/reports/                  full-size analysis tables (CSV)

`data/vendor/` is gitignored in its entirety. Two reasons, both binding:

1. It is licensed third-party data and this repository is public.
2. A raw vendor file must never become a production input by accident. Production reads
   `db/breakwater.duckdb` and `output/*.parquet`; nothing under `data/vendor/` is on that path,
   and `testing/test_massive_earnings.py` asserts statically that no pipeline module
   mentions it.

Nothing in this package writes to `db/` or `output/`, or anywhere in `data/` outside `data/vendor/`.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VENDOR_ROOT = REPO_ROOT / "data" / "vendor"
MASSIVE_ROOT = VENDOR_ROOT / "massive"

EARNINGS_SNAPSHOT_ROOT = MASSIVE_ROOT / "earnings"
NORMALIZED_ROOT = MASSIVE_ROOT / "normalized"
REPORT_DATA_ROOT = MASSIVE_ROOT / "reports"

# The one committed artifact of this work: the audit write-up itself.
AUDIT_REPORT_PATH = REPO_ROOT / "audit" / "BENZINGA_EARNINGS_AUDIT.md"

# The independent yardstick the vendor history is validated against (yfinance-sourced,
# tz-aware America/New_York). Read-only here.
PROVIDER_TIMESTAMPS_PATH = REPO_ROOT / "audit" / "provider_timestamps.parquet"

STOCK_LIST_PATH = REPO_ROOT / "data" / "stock_list.csv"
