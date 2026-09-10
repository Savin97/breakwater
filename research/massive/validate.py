"""Validate the Benzinga history against an INDEPENDENT timestamp source.

The yardstick is `audit/provider_timestamps.parquet`: 12,269 event timestamps pulled from
yfinance for the 500 current Breakwater tickers, tz-aware America/New_York. It was
collected for Phase 0, long before this vendor existed in the project, so it is a genuinely
independent read of the same events. It is not ground truth — it carries its own rounding,
which is exactly why the comparisons below separate "same minute" from "same window".

The timezone question
---------------------
The vendor documents `time` as "24-hour HH:MM:SS EST". Taken literally that is a FIXED
UTC-5 clock, which for the ~two-thirds of the year New York spends on EDT would be one
hour behind local market time — and one hour is the difference between 16:00 (AMC) and
15:00 (INTRADAY), or 09:00 (BMO) and 10:00 (INTRADAY). We refuse to reinterpret the field
on the strength of a word in the documentation, in either direction. `timezone_evidence()`
instead tests the two readings against the independent overlap:

    A "as published"  — the value is New York LOCAL market wall time (DST-aware)
    B "fixed EST"     — the value is UTC-5 year-round, so during EDT it is local-time minus
                        one hour and must be shifted +1h to compare

If the field really were fixed EST, reading A would agree with the independent source
during EST months and degrade sharply during EDT months, while reading B would be flat.
The observed pattern picks the answer.
"""
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from feature_engineering.announcement_timing import (AMC, BMO, INTRADAY, UNKNOWN,
                                                     MARKET_CLOSE_HOUR, MARKET_OPEN_HOUR,
                                                     classify_announce_window)
from research.massive import paths

NO_RECORD = "NO_RECORD"          # kept distinct from UNKNOWN: "no clock" != "no row"

# Records this close to a cut point are listed individually: a five-minute error in either
# source flips the label, so they are the cases a human should eyeball.
BOUNDARY_MINUTES = 15


def load_reference(path=None) -> pd.DataFrame:
    """The independent yardstick, as naive NY wall clock plus the DST flag."""
    ref = pd.read_parquet(path or paths.PROVIDER_TIMESTAMPS_PATH)
    ref = ref.rename(columns={"stock": "ticker_norm", "earnings_date": "report_date"})
    ref["report_date"] = pd.to_datetime(ref["report_date"]).dt.normalize()
    ref["ticker_norm"] = ref["ticker_norm"].astype("string").str.strip().str.upper()
    ref["report_date"] = ref["report_date"].astype("datetime64[ns]")
    ts = ref["announce_ts_ny"]
    ref["ref_ts_ny"] = ts.dt.tz_localize(None)
    # utcoffset -4h == EDT, -5h == EST. Read off the timestamps themselves, not a calendar.
    ref["ref_is_edt"] = ts.dt.strftime("%z").str[:3].astype(int).eq(-4)
    ref["ref_window"] = classify_announce_window(ref["ref_ts_ny"]).to_numpy()
    return ref[["ticker_norm", "report_date", "ref_ts_ny", "ref_is_edt", "ref_window"]]


def vendor_event_view(bz: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, report date) for joining — the unambiguous records only.

    Where the vendor holds several records for the same ticker and date the pair is kept
    but marked, so a duplicate can never masquerade as agreement or as a disagreement.
    """
    cols = ["ticker_norm", "report_date", "announce_ts_vendor", "announce_time_raw",
            "time_quality", "time_usable", "announce_window", "announce_hour",
            "date_status_norm", "is_confirmed", "company_name", "benzinga_id"]
    v = bz[cols].copy()
    v["ticker_norm"] = v["ticker_norm"].astype("string")
    v["report_date"] = v["report_date"].astype("datetime64[ns]")
    # Carried through the tolerant join: `merge_asof` keeps the LEFT frame's `report_date`,
    # so without this copy a one-day date disagreement would silently look like agreement.
    v["vendor_report_date"] = v["report_date"]
    v["vendor_records_on_date"] = v.groupby(["ticker_norm", "report_date"],
                                            dropna=False)["benzinga_id"].transform("size")
    # Deterministic pick when duplicated: prefer confirmed, then a usable time, then the
    # first benzinga_id. The `vendor_records_on_date` flag keeps the choice visible.
    v = v.sort_values(["ticker_norm", "report_date", "is_confirmed", "time_usable",
                       "benzinga_id"], ascending=[True, True, False, False, True])
    return v.drop_duplicates(subset=["ticker_norm", "report_date"], keep="first")


def match_events(bz: pd.DataFrame, ref: pd.DataFrame, tolerance_days: int = 1) -> pd.DataFrame:
    """Left-join the reference onto the vendor, exact date first, then +/- `tolerance_days`.

    Two different questions live here and are answered separately:
    *date* agreement (does the vendor know this event happened on this day at all) and
    *timing* agreement (given the same event, do the clocks say the same thing). Matching
    only on the exact date would silently score a one-day date disagreement as "vendor has
    no record", which is a different defect with a different remedy.
    """
    v = vendor_event_view(bz)
    ref = ref.copy()
    ref["ticker_norm"] = ref["ticker_norm"].astype("string")
    ref["report_date"] = pd.to_datetime(ref["report_date"]).astype("datetime64[ns]")
    exact = ref.merge(v, on=["ticker_norm", "report_date"], how="left", indicator=True)
    exact["date_match"] = np.where(exact["_merge"].eq("both"), "exact", "none")
    exact = exact.drop(columns="_merge")

    unmatched = exact["date_match"].eq("none")
    if unmatched.any() and tolerance_days:
        near = pd.merge_asof(
            exact.loc[unmatched, ["ticker_norm", "report_date"]]
                 .sort_values("report_date").reset_index(),
            v.sort_values("report_date"),
            on="report_date", by="ticker_norm", direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days))
        near = near.dropna(subset=["benzinga_id"]).set_index("index")
        for col in v.columns:
            if col in ("ticker_norm", "report_date"):
                continue
            exact.loc[near.index, col] = near[col]
        exact.loc[near.index, "date_match"] = "within_1d"
    exact["has_vendor_record"] = exact["benzinga_id"].notna()
    exact["date_delta_days"] = ((exact["vendor_report_date"] - exact["report_date"])
                                .dt.days)
    exact["announce_window"] = exact["announce_window"].fillna(UNKNOWN)
    exact["delta_minutes"] = ((exact["announce_ts_vendor"] - exact["ref_ts_ny"])
                              .dt.total_seconds() / 60.0)
    return exact


def timezone_evidence(matched: pd.DataFrame) -> pd.DataFrame:
    """Reading A (NY local) vs reading B (fixed EST), scored against the reference.

    Reported separately for EST and EDT events, because the two readings are identical
    during EST by construction — the whole signal lives in the EDT rows.
    """
    m = matched[matched["time_usable"].fillna(False) & matched["ref_ts_ny"].notna()].copy()
    shifted = m["announce_ts_vendor"] + pd.to_timedelta(np.where(m["ref_is_edt"], 60, 0), "m")
    m["window_as_published"] = m["announce_window"]
    m["window_fixed_est"] = classify_announce_window(shifted).to_numpy()
    m["delta_fixed_est"] = (shifted - m["ref_ts_ny"]).dt.total_seconds() / 60.0

    rows = []
    for label, sub in (("EST (winter)", m[~m["ref_is_edt"]]),
                       ("EDT (summer)", m[m["ref_is_edt"]]),
                       ("all", m)):
        if sub.empty:
            continue
        rows.append({
            "period": label,
            "events": len(sub),
            "window_agree_as_published": (sub["window_as_published"] == sub["ref_window"]).mean(),
            "window_agree_fixed_est": (sub["window_fixed_est"] == sub["ref_window"]).mean(),
            "median_delta_min_as_published": sub["delta_minutes"].median(),
            "median_delta_min_fixed_est": sub["delta_fixed_est"].median(),
            "within_5min_as_published": sub["delta_minutes"].abs().le(5).mean(),
            "within_5min_fixed_est": sub["delta_fixed_est"].abs().le(5).mean(),
            "exactly_minus_60_as_published": sub["delta_minutes"].eq(-60).mean(),
        })
    return pd.DataFrame(rows)


def agreement_summary(matched: pd.DataFrame) -> dict:
    """The headline validation numbers."""
    comparable = matched[matched["time_usable"].fillna(False)]
    both = comparable[comparable["ref_window"].isin([BMO, AMC])
                      & comparable["announce_window"].isin([BMO, AMC])]
    out = {
        "reference_events": len(matched),
        "matched_any": int(matched["has_vendor_record"].sum()),
        "matched_exact_date": int(matched["date_match"].eq("exact").sum()),
        "matched_within_1d": int(matched["date_match"].eq("within_1d").sum()),
        "unmatched": int(matched["date_match"].eq("none").sum()),
        "matched_with_usable_time": len(comparable),
        "window_agreement_all": float((comparable["announce_window"]
                                       == comparable["ref_window"]).mean()) if len(comparable) else float("nan"),
        "bmo_amc_pairs": len(both),
        "bmo_amc_agreement": float((both["announce_window"] == both["ref_window"]).mean()) if len(both) else float("nan"),
    }
    for w in (BMO, AMC):
        sub = comparable[comparable["ref_window"].eq(w)]
        out[f"{w.lower()}_reference_events"] = len(sub)
        out[f"{w.lower()}_agreement"] = float((sub["announce_window"] == sub["ref_window"]).mean()) if len(sub) else float("nan")
        vsub = comparable[comparable["announce_window"].eq(w)]
        out[f"{w.lower()}_vendor_precision"] = float((vsub["ref_window"] == w).mean()) if len(vsub) else float("nan")
    out["exact_minute_agreement"] = float(comparable["delta_minutes"].eq(0).mean()) if len(comparable) else float("nan")
    out["within_5_minutes"] = float(comparable["delta_minutes"].abs().le(5).mean()) if len(comparable) else float("nan")
    out["within_60_minutes"] = float(comparable["delta_minutes"].abs().le(60).mean()) if len(comparable) else float("nan")
    return out



# ---------------------------------------------------------------------------- anchoring
def nyse_sessions(start="2009-01-01", end="2030-12-31") -> np.ndarray:
    """NYSE trading days as naive dates. An exchange calendar, not Breakwater's price data:
    this module must not need a database or a parquet to run."""
    days = mcal.get_calendar("NYSE").valid_days(start, end)
    return np.sort(pd.DatetimeIndex(days).tz_localize(None).normalize().to_numpy())


def anchor_session(window: pd.Series, date: pd.Series, sessions=None) -> pd.Series:
    """The last trading session STRICTLY BEFORE the announcement — Phase 2's anchor rule.

        AMC on D   -> the last session on or before D   (the announcement follows D's close)
        BMO on D   -> the last session strictly before D (the announcement precedes D's open)
        otherwise  -> NaT, because the anchor is genuinely ambiguous

    This is what makes "AMC on Friday" and "BMO on the following Monday" comparable: they
    are two descriptions of the same instant — after Friday's close, before Monday's open —
    and they anchor to the same session. A window disagreement that is really a date
    convention disagreement therefore shows up as anchor AGREEMENT, which is the property
    Phase 3 would actually consume.
    """
    sessions = nyse_sessions() if sessions is None else sessions
    d = pd.to_datetime(date).to_numpy(dtype="datetime64[ns]")
    on_or_before = np.searchsorted(sessions, d, side="right") - 1
    strictly_before = np.searchsorted(sessions, d, side="left") - 1
    w = pd.Series(window).to_numpy()
    idx = np.where(w == AMC, on_or_before, np.where(w == BMO, strictly_before, -1))
    out = np.full(len(idx), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = idx >= 0
    out[ok] = sessions[idx[ok]]
    return pd.Series(out, index=pd.Series(window).index)


def anchor_agreement(matched: pd.DataFrame) -> dict:
    """Do the two sources imply the same anchor session? The Phase-3-relevant question.

    Restricted to events where BOTH sources give an unambiguous BMO/AMC window; INTRADAY
    and UNKNOWN have no anchor on either side and are counted, not guessed at.
    """
    sessions = nyse_sessions()
    m = matched[matched["time_usable"].fillna(False)].copy()
    vendor_date = m["vendor_report_date"].fillna(m["report_date"]) if "vendor_report_date" in m else m["report_date"]
    m["ref_anchor"] = anchor_session(m["ref_window"], m["report_date"], sessions)
    m["vendor_anchor"] = anchor_session(m["announce_window"], vendor_date, sessions)
    both = m[m["ref_anchor"].notna() & m["vendor_anchor"].notna()]
    agree = both["ref_anchor"].eq(both["vendor_anchor"])
    dis = m[m["announce_window"] != m["ref_window"]]
    dis_both = dis[dis["ref_anchor"].notna() & dis["vendor_anchor"].notna()]
    return {
        "events_with_an_anchor_on_both_sides": len(both),
        "same_anchor_session": int(agree.sum()),
        "anchor_agreement": float(agree.mean()) if len(both) else float("nan"),
        "different_anchor_session": int((~agree).sum()),
        "window_disagreements": len(dis),
        "window_disagreements_that_still_share_an_anchor":
            int(dis_both["ref_anchor"].eq(dis_both["vendor_anchor"]).sum()),
    }


def reference_placeholder_diagnostics(matched: pd.DataFrame) -> pd.DataFrame:
    """How often the INDEPENDENT source's clock is itself a round-number placeholder.

    The yardstick is not ground truth. A timestamp sitting exactly on the hour, quarter
    after quarter, is a scheduling default rather than an observation, and it is the main
    reason a raw window-agreement figure understates the vendor. Reported so the reader can
    discount both sides, not just one.
    """
    m = matched[matched["ref_ts_ny"].notna()].copy()
    m["ref_on_the_hour"] = m["ref_ts_ny"].dt.minute.eq(0) & m["ref_ts_ny"].dt.second.eq(0)
    rows = []
    for w, sub in m.groupby("ref_window"):
        rows.append({"reference_window": w, "events": len(sub),
                     "on_the_hour": int(sub["ref_on_the_hour"].sum()),
                     "share_on_the_hour": float(sub["ref_on_the_hour"].mean())})
    v = matched[matched["announce_ts_vendor"].notna()]
    on_hour_vendor = (v["announce_ts_vendor"].dt.minute.eq(0)
                      & v["announce_ts_vendor"].dt.second.eq(0))
    rows.append({"reference_window": "— vendor, all matched —", "events": len(v),
                 "on_the_hour": int(on_hour_vendor.sum()),
                 "share_on_the_hour": float(on_hour_vendor.mean())})
    return pd.DataFrame(rows)


def confusion(matched: pd.DataFrame) -> pd.DataFrame:
    """Reference window (rows) x vendor window (columns), every matched event."""
    order = [BMO, AMC, INTRADAY, UNKNOWN, NO_RECORD]
    vendor = matched["announce_window"].where(matched["has_vendor_record"], NO_RECORD)
    ref = pd.Categorical(matched["ref_window"], categories=order)
    ven = pd.Categorical(vendor.fillna(UNKNOWN), categories=order)
    return pd.crosstab(pd.Series(ref, name="reference"), pd.Series(ven, name="vendor"),
                       dropna=False)


def disagreements(matched: pd.DataFrame) -> pd.DataFrame:
    """Every matched event where the two sources classify the window differently."""
    m = matched[matched["time_usable"].fillna(False)]
    d = m[m["announce_window"] != m["ref_window"]].copy()
    d["ref_time"] = d["ref_ts_ny"].dt.strftime("%Y-%m-%d %H:%M")
    d["vendor_time"] = d["announce_ts_vendor"].dt.strftime("%Y-%m-%d %H:%M")
    cols = ["ticker_norm", "report_date", "ref_time", "ref_window", "vendor_time",
            "announce_window", "delta_minutes", "date_match", "date_status_norm",
            "vendor_records_on_date", "benzinga_id"]
    return d[cols].sort_values(["report_date", "ticker_norm"])


def disagreement_taxonomy(matched: pd.DataFrame) -> pd.DataFrame:
    """Sort every window disagreement into a kind, and say whether it moves the anchor.

    A raw disagreement count treats three very different things as one number: the
    reference's scheduling placeholder, a one-day date convention difference that lands on
    the same anchor session, and a genuine contradiction about whether the news preceded or
    followed a session. Only the third can corrupt a corrected target.
    """
    sessions = nyse_sessions()
    m = matched[matched["time_usable"].fillna(False)].copy()
    vd = m["vendor_report_date"].fillna(m["report_date"])
    m["ref_anchor"] = anchor_session(m["ref_window"], m["report_date"], sessions)
    m["vendor_anchor"] = anchor_session(m["announce_window"], vd, sessions)
    d = m[m["announce_window"] != m["ref_window"]].copy()
    d["kind"] = d["ref_window"] + " -> " + d["announce_window"].astype(str)
    d["same_anchor"] = d["ref_anchor"].eq(d["vendor_anchor"]) & d["ref_anchor"].notna()
    d["anchor_undefined"] = d["ref_anchor"].isna() | d["vendor_anchor"].isna()
    g = d.groupby("kind").agg(events=("kind", "size"),
                              same_anchor=("same_anchor", "sum"),
                              anchor_undefined=("anchor_undefined", "sum"))
    g["anchor_moved"] = g["events"] - g["same_anchor"] - g["anchor_undefined"]
    return g.reset_index().sort_values("events", ascending=False)


def disagreements_by_year(matched: pd.DataFrame) -> pd.DataFrame:
    m = matched[matched["time_usable"].fillna(False)].copy()
    m["year"] = m["report_date"].dt.year
    m["disagrees"] = m["announce_window"] != m["ref_window"]
    g = m.groupby("year").agg(events=("disagrees", "size"), disagreements=("disagrees", "sum"))
    g["rate"] = g["disagreements"] / g["events"]
    return g.reset_index()


def disagreements_by_ticker(matched: pd.DataFrame, min_count: int = 1) -> pd.DataFrame:
    m = matched[matched["time_usable"].fillna(False)].copy()
    m["disagrees"] = m["announce_window"] != m["ref_window"]
    g = m.groupby("ticker_norm").agg(events=("disagrees", "size"),
                                     disagreements=("disagrees", "sum"))
    g = g[g["disagreements"] >= min_count]
    g["rate"] = g["disagreements"] / g["events"]
    return g.sort_values(["disagreements", "rate"], ascending=False).reset_index()


def boundary_cases(bz: pd.DataFrame, minutes: int = BOUNDARY_MINUTES) -> pd.DataFrame:
    """Vendor records sitting within `minutes` of 09:30 or 16:00, plus every INTRADAY.

    These are the labels a small clock error can flip, so they are enumerated rather than
    summarised.
    """
    v = bz[bz["time_usable"]].copy()
    tol = minutes / 60.0
    near_open = (v["announce_hour"] - MARKET_OPEN_HOUR).abs() <= tol
    near_close = (v["announce_hour"] - MARKET_CLOSE_HOUR).abs() <= tol
    sel = v[near_open | near_close | v["announce_window"].eq(INTRADAY)].copy()
    sel["boundary"] = np.where(near_open[sel.index], "09:30",
                               np.where(near_close[sel.index], "16:00", "intraday"))
    return sel[["ticker_norm", "report_date", "announce_time_raw", "announce_window",
                "boundary", "date_status_norm", "benzinga_id"]].sort_values(
                    ["boundary", "report_date"])


def time_distribution(bz: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    counts = bz["announce_time_raw"].value_counts(dropna=False)
    out = counts.head(top).rename("records").to_frame().reset_index(names="time")
    out["share"] = out["records"] / len(bz)
    return out


def duplicate_report(bz: pd.DataFrame) -> dict:
    dup_date = bz[bz["n_records_ticker_date"] > 1]
    dup_fiscal = bz[(bz["n_records_ticker_fiscal"] > 1) & bz["fiscal_period"].notna()]
    return {
        "records": len(bz),
        "duplicate_benzinga_ids": int(bz["is_duplicate_benzinga_id"].sum()),
        "records_sharing_ticker_date": len(dup_date),
        "ticker_date_pairs_affected": int(dup_date.groupby(["ticker_norm", "report_date"]).ngroups),
        "records_sharing_ticker_fiscal_period": len(dup_fiscal),
        "ticker_fiscal_groups_affected": int(dup_fiscal.groupby(
            ["ticker_norm", "fiscal_year", "fiscal_period"]).ngroups),
        "examples": dup_date.sort_values(["ticker_norm", "report_date"])
                            [["ticker_norm", "report_date", "announce_time_raw",
                              "date_status_norm", "fiscal_year", "fiscal_period",
                              "benzinga_id"]].head(20),
    }


def status_report(bz: pd.DataFrame) -> pd.DataFrame:
    """Projected vs confirmed, and how usable the clock is inside each."""
    g = bz.groupby("date_status_norm", dropna=False)
    out = g.agg(records=("benzinga_id", "size"),
                usable_time=("time_usable", "sum"),
                midnight_filler=("time_quality", lambda s: (s == "midnight_filler").sum()),
                with_actual_eps=("has_actual_eps", "sum"))
    out["usable_share"] = out["usable_time"] / out["records"]
    return out.reset_index()


def time_quality_report(bz: pd.DataFrame) -> pd.DataFrame:
    g = bz.groupby("time_quality", dropna=False).agg(records=("benzinga_id", "size"))
    g["share"] = g["records"] / len(bz)
    return g.reset_index()
