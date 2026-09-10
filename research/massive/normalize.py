"""Raw vendor payloads -> one normalized research frame. No analysis, no judgement calls
that are not written down here.

Two rules govern this module.

**Every vendor field survives.** The normalized frame carries the union of every key the
vendor ever emitted, unrenamed and unconverted apart from dtype. Normalized columns are
*additional* and are suffixed or explicitly named (`ticker_norm`, `report_date`,
`announce_ts_vendor`, ...), so no vendor value is ever silently replaced by a derived one.
If the vendor adds a field tomorrow it appears automatically; if it drops an optional one
the column is created empty rather than the row being discarded.

**The window classification is Breakwater's, not the vendor's.** `announce_window` comes
from `feature_engineering.announcement_timing.classify_announce_window` — the same
already-reviewed function the production event frame uses — applied to the vendor clock.
It is a pure function of that clock. Nothing here reads a price, a return or a reaction.

The midnight question
---------------------
`time == "00:00:00"` is not treated as a real BMO announcement. Under the literal Phase 2
rule 00:00 < 09:30 would classify as BMO, which would manufacture thousands of confident
BMO labels out of the vendor's filler value for "time unknown" — exactly the fabrication
`audit/PHASE0_AUDIT_REV2.md` forbids. It is recorded as `time_quality="midnight_filler"`
and classified UNKNOWN. `normalize(..., midnight_is_real=True)` produces the literal
reading instead, so the sensitivity of every downstream number to this one decision can be
measured rather than argued about.
"""
import re
from datetime import datetime

import pandas as pd

from feature_engineering.announcement_timing import classify_announce_window

# Vendor fields, in the order the audit report shows them. Anything not listed is still
# carried — this only fixes the column order for the known ones.
VENDOR_FIELDS = [
    "benzinga_id", "ticker", "company_name", "currency", "date", "time", "date_status",
    "importance", "fiscal_year", "fiscal_period", "actual_eps", "estimated_eps",
    "previous_eps", "eps_surprise", "eps_surprise_percent", "eps_method",
    "actual_revenue", "estimated_revenue", "previous_revenue", "revenue_surprise",
    "revenue_surprise_percent", "revenue_method", "notes", "last_updated",
]

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")

TIME_OK = "ok"
TIME_MISSING = "missing"
TIME_MALFORMED = "malformed"
TIME_MIDNIGHT = "midnight_filler"


def parse_vendor_time(value) -> tuple[object, str]:
    """`(datetime.time or None, quality)` for one vendor `time` string.

    Deliberately strict: anything that is not HH:MM[:SS] within real clock bounds is
    MALFORMED and becomes UNKNOWN, never a guess. Midnight is separated out because it is
    the vendor's filler, not a time.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, TIME_MISSING
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None, TIME_MISSING
    m = _TIME_RE.match(s)
    if not m:
        return None, TIME_MALFORMED
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if hh > 23 or mm > 59 or ss > 59:
        return None, TIME_MALFORMED
    t = datetime(2000, 1, 1, hh, mm, ss).time()
    if (hh, mm, ss) == (0, 0, 0):
        return t, TIME_MIDNIGHT
    return t, TIME_OK


def normalize(records, snapshot_id: str | None = None,
              midnight_is_real: bool = False) -> pd.DataFrame:
    """Vendor records -> the normalized research frame.

    `records` is any iterable of vendor dicts (e.g. `acquire.iter_records(snapshot)`).
    """
    rows = list(records)
    raw = pd.DataFrame(rows)
    for col in VENDOR_FIELDS:                      # schema variation: never lose a row
        if col not in raw.columns:
            raw[col] = pd.NA
    ordered = VENDOR_FIELDS + [c for c in raw.columns if c not in VENDOR_FIELDS]
    out = raw[ordered].copy()

    # ---- identifiers -------------------------------------------------------------
    out["ticker_norm"] = (out["ticker"].astype("string").str.strip().str.upper()
                          .replace({"": pd.NA}))
    out["company_name_norm"] = (out["company_name"].astype("string").str.strip()
                                .str.casefold().replace({"": pd.NA}))

    # ---- the calendar date -------------------------------------------------------
    out["report_date"] = pd.to_datetime(out["date"], errors="coerce", format="%Y-%m-%d")
    out["report_year"] = out["report_date"].dt.year.astype("Int64")

    # ---- the announcement clock --------------------------------------------------
    parsed = [parse_vendor_time(v) for v in out["time"]]
    out["announce_time_raw"] = out["time"].astype("string")
    out["time_quality"] = pd.Series([q for _, q in parsed], index=out.index, dtype="string")
    usable_quality = {TIME_OK, TIME_MIDNIGHT} if midnight_is_real else {TIME_OK}
    out["time_usable"] = out["time_quality"].isin(usable_quality) & out["report_date"].notna()

    # `announce_ts_vendor` is the vendor clock as published: report date + published time,
    # naive. Whether that clock is fixed EST or New York local wall time is the question
    # `validate.py` answers empirically; nothing here assumes an answer.
    stamps = []
    for (t, _q), d, usable in zip(parsed, out["report_date"], out["time_usable"]):
        stamps.append(pd.Timestamp.combine(d, t) if (usable and t is not None
                                                     and pd.notna(d)) else pd.NaT)
    out["announce_ts_vendor"] = pd.Series(stamps, index=out.index, dtype="datetime64[ns]")
    out["announce_hour"] = (out["announce_ts_vendor"].dt.hour
                            + out["announce_ts_vendor"].dt.minute / 60.0)

    # The already-reviewed Phase 2 rule, applied to the vendor clock and to nothing else.
    out["announce_window"] = classify_announce_window(out["announce_ts_vendor"]).to_numpy()

    # ---- provenance / freshness --------------------------------------------------
    out["last_updated_utc"] = pd.to_datetime(out["last_updated"], errors="coerce", utc=True)
    out["date_status_norm"] = (out["date_status"].astype("string").str.strip().str.lower()
                               .replace({"": pd.NA}))
    out["is_confirmed"] = out["date_status_norm"].eq("confirmed")
    out["has_actual_eps"] = pd.to_numeric(out["actual_eps"], errors="coerce").notna()

    # ---- duplicate accounting (flagged, never dropped) ---------------------------
    out["is_duplicate_benzinga_id"] = out.duplicated(subset=["benzinga_id"], keep=False)
    key = ["ticker_norm", "report_date"]
    out["n_records_ticker_date"] = out.groupby(key, dropna=False)["benzinga_id"].transform("size")
    fkey = ["ticker_norm", "fiscal_year", "fiscal_period"]
    out["n_records_ticker_fiscal"] = out.groupby(fkey, dropna=False)["benzinga_id"].transform("size")

    out["snapshot_id"] = snapshot_id
    return out.reset_index(drop=True)


def normalize_snapshot(snapshot, midnight_is_real: bool = False) -> pd.DataFrame:
    """Convenience: read a finished snapshot off disk and normalize it."""
    from pathlib import Path

    from research.massive.acquire import iter_records
    snapshot = Path(snapshot)
    return normalize(iter_records(snapshot), snapshot_id=snapshot.name,
                     midnight_is_real=midnight_is_real)
