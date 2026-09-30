"""Phase A — match Breakwater events to the latest options snapshot at the call cutoff.

Research-only. Reads the Phase 3 refit feature frame (`output/phase3_refit/
feature_frame.parquet`, call-cutoff clocks and Model C inputs), the session grid from
`output/full_df.parquet`, and the pinned DoltHub source through `source.py`'s cache.
Writes only `output/options_pilot/`. No database, production module or vendor seed is
written.

    PYTHONPATH=. .venv/bin/python -m research.options_pilot.build [--offline]

The one rule that matters
-------------------------
An event may only see a snapshot whose date is at or before its call cutoff (the last
session before the Monday of the report week — Phase 3's `call_cutoff_date`, reused, not
recomputed). `select_snapshot` walks back from the cutoff through the dates the source
actually has; nothing is forward-filled: a snapshot is used only for the symbol it was
fetched for, its age in market sessions is recorded, and staleness limits are applied
later, in the open, by `evaluate.py`.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO
from research.options_pilot import features as F
from research.options_pilot import source

FEATURE_FRAME = Path("output/phase3_refit/feature_frame.parquet")
FULL_DF = Path("output/full_df.parquet")
OUT = Path("output/options_pilot")
CHAIN_DATES = OUT / "chain_dates.csv"
VH_DATES = OUT / "vh_dates.csv"

FIRST_YEAR = 2019           # the source starts 2019-02-09
MIN_PRIOR = 8               # Phase 3 population rule
MAX_WALK_SESSIONS = 10      # walk back at most this far; beyond it the event is "missing"
WORKERS = 6
IV_HIST_DAYS = 365          # IV-relative-history lookback (calendar days, strictly prior)
IV_HIST_MIN_OBS = 20

# ── identity ───────────────────────────────────────────────────────────────────────────
# The source keys contracts on the symbol AS TRADED ON THE SNAPSHOT DATE (FB until
# 2022-06-08, META after) and spells class shares with a dot. For every rename in the
# population, the switch date is the day after the OLD symbol's last row in the source
# (`output/options_pilot/rename_probe.csv`: both symbols queried on every snapshot date
# within ±16 days of the public change date). The source does not carry the new symbol
# from the switch — it appears days to months later — so those events are missing, not
# guessed. Every mapping is then checked by `identity_audit()` (parity spot / adjusted
# close continuous across the switch). Nothing is inferred from outcomes.
# XYZ (was SQ) has no mapping: SQ never appears in the source.
# {breakwater stock: [(old source symbol, switch date = first date of the NEW symbol)]}
_RENAMES = {
    "META": ("FB", "2022-06-09"), "ELV": ("ANTM", "2022-06-28"),
    "WTW": ("WLTW", "2022-01-08"), "BALL": ("BLL", "2022-05-10"),
    "CTRA": ("COG", "2021-10-02"), "RVTY": ("PKI", "2023-05-16"),
    "DAY": ("CDAY", "2024-02-01"), "COR": ("ABC", "2023-08-29"),
    "EG": ("RE", "2023-07-08"), "TT": ("IR", "2020-03-02"),
    "HWM": ("ARNC", "2020-04-01"), "VTRS": ("MYL", "2020-11-17"),
    "RTX": ("UTX", "2020-04-02"), "J": ("JEC", "2019-12-08"),
    "BKR": ("BHGE", "2019-10-13"), "LHX": ("HRS", "2019-07-01"),
    "DD": ("DWDP", "2019-06-02"),
}
SYMBOL_HISTORY: dict[str, list[tuple[str, str]]] = {
    new: [(old, "1900-01-01"), (new, switch)] for new, (old, switch) in _RENAMES.items()
}
# Snapshots strictly before this date belong to a DIFFERENT company trading under the same
# string; events whose snapshot would come from before it are excluded, not joined.
EXCLUDE_BEFORE: dict[str, str] = {}
EXCLUDE_ALL: dict[str, str] = {}


def source_symbol(stock: str, snap_date) -> str | None:
    """The source's symbol for Breakwater `stock` on `snap_date`; None if excluded."""
    d = pd.Timestamp(snap_date)
    if stock in EXCLUDE_ALL:
        return None
    if stock in EXCLUDE_BEFORE and d < pd.Timestamp(EXCLUDE_BEFORE[stock]):
        return None
    sym = stock.replace("-", ".")
    for s, start in SYMBOL_HISTORY.get(stock, []):
        if d >= pd.Timestamp(start):
            sym = s
    return sym


# ── clocks ─────────────────────────────────────────────────────────────────────────────
def effective_session(snap_dates: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Grid index of the last session at or before each snapshot date (-1 if none).

    A weekday snapshot is its own session. A weekend/holiday-dated snapshot (2019 is
    Saturday-dated) carries the previous session's quotes — verified: 2019-05-11 is
    byte-identical to 2019-05-10 for AAPL.
    """
    d = np.asarray(snap_dates, dtype="datetime64[ns]")
    return (np.searchsorted(grid, d, side="right") - 1).astype(np.int64)


def candidate_snapshots(cutoff_date, chain_dates: np.ndarray, grid: np.ndarray,
                        max_age: int = MAX_WALK_SESSIONS) -> list[tuple[pd.Timestamp, int]]:
    """Source snapshot dates at or before `cutoff_date`, newest first, with their age in
    market sessions, up to `max_age`. Never a date after the cutoff."""
    c = np.datetime64(pd.Timestamp(cutoff_date), "ns")
    cut_idx = int(np.searchsorted(grid, c, side="right") - 1)
    eligible = chain_dates[chain_dates <= c][::-1]
    out = []
    for d in eligible:
        age = cut_idx - int(effective_session(np.array([d]), grid)[0])
        if age > max_age:
            break
        out.append((pd.Timestamp(d), age))
    return out


def select_snapshot(cutoff_date, stock: str, chain_dates: np.ndarray, grid: np.ndarray,
                    fetch=source.chain) -> dict:
    """The latest snapshot at or before the cutoff that has rows for this stock."""
    tried = 0
    for d, age in candidate_snapshots(cutoff_date, chain_dates, grid):
        sym = source_symbol(stock, d)
        if sym is None:
            return {"match_status": "identity_excluded", "n_tried": tried}
        tried += 1
        rows = fetch(d.strftime("%Y-%m-%d"), sym)
        if rows:
            assert d <= pd.Timestamp(cutoff_date)
            return {"match_status": "matched", "snapshot_date": d, "snapshot_age": age,
                    "source_symbol": sym, "rows": rows, "n_tried": tried}
    return {"match_status": f"none_within_{MAX_WALK_SESSIONS}_sessions", "n_tried": tried}


# ── population ─────────────────────────────────────────────────────────────────────────
def population() -> pd.DataFrame:
    """Phase 3's population with Model C's inputs, from 2019.

    Completed, BMO/AMC, corrected 3-session target available (its endpoint resolved),
    >= 8 prior corrected outcomes at the call cutoff, both C inputs present.
    """
    from research.phase3_refit.candidates import add_model_inputs
    f = pd.read_parquet(FEATURE_FRAME)
    f = add_model_inputs(f, "call")
    keep = (~f["is_pending"].astype(bool) & f["announce_window"].isin([BMO, AMC])
            & f["y_extreme"].notna() & f["abs_r3"].notna() & f["endpoint3_idx"].ge(0)
            & f["n_prior_call"].ge(MIN_PRIOR) & f["year"].ge(FIRST_YEAR)
            & f["log_hist_mean_abs"].notna() & f["log_vol_30d"].notna())
    pop = f[keep].copy()
    pop["event_id"] = pop["stock"] + "|" + pd.to_datetime(pop["earnings_date"]).dt.strftime("%Y-%m-%d")
    assert pop["event_id"].is_unique
    return pop


def session_grid() -> np.ndarray:
    d = pd.read_parquet(FULL_DF, columns=["date"])
    return np.sort(pd.unique(pd.to_datetime(d["date"]).to_numpy(dtype="datetime64[ns]")))


WINDOW_CHAIN_DATES = OUT / "calendar_window_dates.csv"


def load_calendar() -> np.ndarray:
    """Snapshot calendar = the volatility_history dates UNION every option_chain date that
    `calendar_check.py` found inside any event's matching window.

    The check found option_chain dates volatility_history lacks (2023-11-16..24), so the
    vh list alone overstated 22 events' snapshot age. With the union, every chain date
    that could be selected for any event is on the calendar. A missed date could only ever
    make a snapshot look older, never admit one after the cutoff."""
    d = pd.read_csv(VH_DATES, parse_dates=["date"])["date"]
    if WINDOW_CHAIN_DATES.exists():
        d = pd.concat([d, pd.read_csv(WINDOW_CHAIN_DATES, parse_dates=["chain_date"])["chain_date"]])
    return np.sort(pd.unique(d.to_numpy(dtype="datetime64[ns]")))


def fetch_all(pop: pd.DataFrame, calendar: np.ndarray, grid: np.ndarray,
              offline: bool = False) -> dict[tuple[str, str], list]:
    """Fetch, pass by pass, the k-th candidate snapshot of every still-unmatched event,
    batched by snapshot date. Stops per event at the first snapshot with rows."""
    store: dict[tuple[str, str], list] = {}
    cands = {r.event_id: candidate_snapshots(r.call_cutoff_date, calendar, grid)
             for r in pop.itertuples(index=False)}
    stock_of = dict(zip(pop["event_id"], pop["stock"]))
    pending = set(cands)
    k = 0
    while pending:
        need: dict[str, set] = {}
        for eid in list(pending):
            if k >= len(cands[eid]):
                pending.discard(eid)
                continue
            d = cands[eid][k][0]
            sym = source_symbol(stock_of[eid], d)
            if sym is None:
                pending.discard(eid)
                continue
            ds = d.strftime("%Y-%m-%d")
            if (ds, sym) not in store:
                need.setdefault(ds, set()).add(sym)
        items = sorted(need.items())
        print(f"pass {k}: {len(pending)} pending, {len(items)} snapshot dates", flush=True)

        def one(item):
            ds, syms = item
            return ds, source.chains_batch(ds, sorted(syms), offline=offline)

        with ThreadPoolExecutor(1 if offline else WORKERS) as ex:
            for ds, got in ex.map(one, items):
                for s, rows in got.items():
                    store[(ds, s)] = rows
        for eid in list(pending):
            d = cands[eid][k][0]
            sym = source_symbol(stock_of[eid], d)
            if store.get((d.strftime("%Y-%m-%d"), sym)):
                pending.discard(eid)
        k += 1
    return store


# ── per-event build ────────────────────────────────────────────────────────────────────
def build_event(row, chain_dates, grid, fetch=source.chain) -> dict:
    base = {"event_id": row.event_id, "stock": row.stock,
            "call_cutoff_date": pd.Timestamp(row.call_cutoff_date)}
    sel = select_snapshot(row.call_cutoff_date, row.stock, chain_dates, grid, fetch)
    rows = sel.pop("rows", None)
    base.update(sel)
    if rows is None:
        return base
    chain = F.normalise_chain(rows)
    feats = F.snapshot_features(chain, row.report_date, row.announce_window)
    base.update(feats)
    return base


def build(offline: bool = False) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = population()
    grid = session_grid()
    calendar = load_calendar()
    store = fetch_all(pop, calendar, grid, offline=offline)
    fetch = lambda d, s: store[(d, s)]        # KeyError = a lookup fetch_all never made
    recs = [build_event(r, calendar, grid, fetch) for r in pop.itertuples(index=False)]
    opt = pd.DataFrame(recs)
    # the calendar check: which calendar dates returned rows for at least one symbol
    seen = pd.DataFrame([(d, bool(v)) for (d, _), v in store.items()], columns=["date", "rows"])
    seen.groupby("date")["rows"].any().rename("exists").to_csv(CHAIN_DATES)
    keep = ["event_id", "stock", "earnings_date", "report_date", "year", "announce_window",
            "call_cutoff_date", "y_extreme", "abs_r3", "log_hist_mean_abs", "log_vol_30d",
            "hist_mean_abs_call", "vol_30d_call", "sector", "endpoint3_date"]
    out = pop[keep].merge(opt.drop(columns=["stock", "call_cutoff_date"]), on="event_id",
                          how="left", validate="one_to_one")
    # the invariant, asserted on the whole table
    m = out["snapshot_date"].notna()
    assert (pd.to_datetime(out.loc[m, "snapshot_date"]) <= out.loc[m, "call_cutoff_date"]).all()
    out.to_parquet(OUT / "event_options.parquet")
    return out


# ── volatility_history (IV relative to own history) ─────────────────────────────────────
def weekly_vh_dates() -> list[pd.Timestamp]:
    """The last volatility_history date in each calendar week (Mon-Sun). One row per week
    gives the lookback the same density in 2019 (weekly source) as in 2025 (daily)."""
    d = pd.read_csv(VH_DATES, parse_dates=["date"])["date"]
    return sorted(d.groupby(d.dt.to_period("W-SUN")).max())


def build_vol_history(ev: pd.DataFrame, offline: bool = False) -> pd.DataFrame:
    """iv_current for our symbols on every weekly date plus every selected snapshot date
    (by-date queries: a per-symbol scan times out)."""
    syms = sorted({s for s in ev["source_symbol"].dropna()})
    dates = set(weekly_vh_dates()) | set(pd.to_datetime(ev["snapshot_date"].dropna()))
    dates = sorted(d.strftime("%Y-%m-%d") for d in dates)
    lst = ",".join(f"'{s}'" for s in syms)
    assert len(syms) < source.ROW_LIMIT

    def one(d):
        return source.query("SELECT date, act_symbol, iv_current, hv_current FROM "
                            f"volatility_history {{REF}} WHERE date = '{d}' AND "
                            f"act_symbol IN ({lst})", offline=offline)

    with ThreadPoolExecutor(1 if offline else WORKERS) as ex:
        allrows = [r for rs in ex.map(one, dates) for r in rs]
    vh = pd.DataFrame(allrows)
    vh["date"] = pd.to_datetime(vh["date"])
    for c in ("iv_current", "hv_current"):
        vh[c] = pd.to_numeric(vh[c], errors="coerce")
    vh["is_weekly"] = vh["date"].isin(weekly_vh_dates())
    vh.to_parquet(OUT / "vol_history.parquet")
    return vh


def iv_relative_history(snap_symbol: str, snap_date, vh: pd.DataFrame) -> dict:
    """iv_current on `snap_date` / median weekly iv_current over the STRICTLY PRIOR 365 days.

    Uses only rows dated before the snapshot (itself at or before the call cutoff), one per
    calendar week. No knowledge of which past dates were earnings dates is used.
    """
    d = pd.Timestamp(snap_date)
    s = vh[vh["act_symbol"] == snap_symbol]
    weekly = s[s["is_weekly"]] if "is_weekly" in s else s
    cur = s.loc[s["date"] == d, "iv_current"]
    prior = weekly.loc[(weekly["date"] < d) & (weekly["date"] >= d - pd.Timedelta(days=IV_HIST_DAYS)),
                       "iv_current"]
    prior = prior[prior > 0]
    out = {"vh_iv_current": float(cur.iloc[0]) if len(cur) else np.nan,
           "vh_prior_n": int(len(prior)), "vh_prior_max_date": s.loc[s["date"] < d, "date"].max()}
    if len(cur) and cur.iloc[0] > 0 and len(prior) >= IV_HIST_MIN_OBS:
        out["iv_rel_hist"] = float(cur.iloc[0] / prior.median())
    return out


# ── identity / split audit ─────────────────────────────────────────────────────────────
CLEAN_RATIOS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0])
JUMP_TOL = 0.15      # |log jump| beyond log(1.15) that is not within 5% of a clean ratio


def identity_audit(ev: pd.DataFrame) -> pd.DataFrame:
    """Per matched event: parity spot / Breakwater adjusted close at the quote session.

    For one company the ratio is constant except at splits (clean ratios) and a slow
    dividend drift. A jump to an unclean ratio between consecutive events means the
    source symbol is not the same security (reused string, wrong rename) or a quote
    problem. Flags are for review; nothing here reads an outcome.
    """
    px = pd.read_parquet(FULL_DF, columns=["stock", "date", "price"])
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    m = ev[ev["status"].eq("ok")][["event_id", "stock", "source_symbol", "snapshot_date",
                                   "parity_spot", "earnings_date"]].copy()
    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"]).astype("datetime64[ns]")
    m = pd.merge_asof(m.sort_values("snapshot_date"), px.sort_values("date"),
                      left_on="snapshot_date", right_on="date", by="stock",
                      direction="backward", tolerance=pd.Timedelta(days=4))
    m["ratio"] = m["parity_spot"] / m["price"]
    m = m.sort_values(["stock", "snapshot_date"])
    m["log_jump"] = np.log(m["ratio"]).groupby(m["stock"]).diff()
    j = np.exp(m["log_jump"].abs())
    near_clean = np.min(np.abs(j.to_numpy()[:, None] / CLEAN_RATIOS[None, :] - 1), axis=1) <= 0.05
    m["jump_flag"] = (m["log_jump"].abs() > np.log(1 + JUMP_TOL)) & ~(near_clean & (j > 1.4))
    m["split_like"] = (m["log_jump"].abs() > np.log(1 + JUMP_TOL)) & near_clean & (j > 1.4)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--vol-history", action="store_true")
    a = ap.parse_args(argv)
    if a.vol_history:
        build_vol_history(pd.read_parquet(OUT / "event_options.parquet"), offline=a.offline)
        return 0
    ev = build(offline=a.offline)
    print(ev["match_status"].value_counts(dropna=False).to_string())
    identity_audit(ev).to_parquet(OUT / "identity_audit.parquet")
    (OUT / "build_meta.json").write_text(json.dumps(
        {"commit": source.PINNED_COMMIT, "n_events": int(len(ev))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
