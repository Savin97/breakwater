# streamlit_dash/data.py
"""What the dashboard serves, and how it knows it is serving the current publication.

Audit H3: the parquet readers were cached with no key, so after a successful publication
the dashboard kept serving the previous calls until someone cleared the cache — while the
archive recorded the new ones as public — and the Upcoming tab used a stored
`days_to_earnings`, so day counts went stale and reported events stayed listed.

Now:
  * `current_served()` is read on EVERY page run (never cached). On the server it follows
    the one-line pointer `output/CURRENT_GENERATION` to `output/generations/<id>/`
    (publishing/generation.py); locally, with no pointer, it serves output/*.parquet.
  * The cached readers take the generation key as an argument, so a new generation (or,
    locally, a rewritten file: mtime + size) is a new cache entry — the old one is never
    returned for it. No manual reload is needed.
  * `upcoming_view()` recomputes days to earnings from today's New York date and drops
    events whose report date has passed, at serve time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from publishing.generation import (CALLS_FILE, DASHBOARD_SCHEMA, HISTORY_FILE, MANIFEST,
                                   POINTER)
from utilities.time_utilities import now_ny

ROOT = Path(__file__).resolve().parents[1]


class ServedGenerationError(RuntimeError):
    """The pointer names a generation the dashboard cannot serve."""


@dataclass(frozen=True)
class Served:
    key: str                    # cache key: the generation id, or local file signature
    generation_id: str | None   # None when serving local output/ files
    calls_path: Path
    history_path: Path


def _signature(*paths: Path) -> str:
    parts = []
    for p in paths:
        st_ = p.stat() if p.exists() else None
        parts.append(f"{p.name}:{st_.st_mtime_ns}:{st_.st_size}" if st_ else f"{p.name}:missing")
    return "local|" + "|".join(parts)


def current_served(root: Path = ROOT) -> Served:
    output = Path(root) / "output"
    pointer = output / POINTER
    if pointer.exists():
        gen = pointer.read_text().strip()
        gen_dir = output / "generations" / gen
        if not gen or not (gen_dir / MANIFEST).exists():
            raise ServedGenerationError(f"{POINTER} names {gen!r}, which is not on this server")
        manifest = json.loads((gen_dir / MANIFEST).read_text())
        if manifest.get("dashboard_schema", 0) > DASHBOARD_SCHEMA:
            raise ServedGenerationError(f"generation {gen} needs dashboard schema "
                                        f"{manifest['dashboard_schema']}; this code knows "
                                        f"{DASHBOARD_SCHEMA} — deploy the matching code")
        return Served(gen, gen, gen_dir / CALLS_FILE, gen_dir / HISTORY_FILE)
    calls, history = output / CALLS_FILE, output / HISTORY_FILE
    return Served(_signature(calls, history), None, calls, history)


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading upcoming events…", max_entries=4)
def get_upcoming_df(generation_key: str, path: str) -> pd.DataFrame:
    """Cached per generation: `generation_key` is part of the cache key on purpose."""
    return _read(path)


@st.cache_data(show_spinner="Loading past reactions…", max_entries=4)
def get_history_df(generation_key: str, path: str) -> pd.DataFrame:
    return _read(path)


def today_ny() -> pd.Timestamp:
    return pd.Timestamp(now_ny().date())


def upcoming_view(upcoming: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Upcoming events as of `today` (New York): days recomputed, reported events dropped."""
    if upcoming.empty:
        return upcoming
    today = pd.Timestamp(today if today is not None else today_ny()).normalize()
    out = upcoming[upcoming["earnings_date"] >= today].copy()
    out["days_to_earnings"] = (out["earnings_date"] - today).dt.days
    return out
