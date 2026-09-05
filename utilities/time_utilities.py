"""The project's one time convention, and the helpers that enforce it.

The convention
--------------
Every announcement-timing column in the `earnings` table is **naive America/New_York
wall-clock time**:

    announce_ts_ny            when the announcement happened, NY wall clock
    announce_ts_observed_at   when the provider was observed saying so, NY wall clock

They are stored naive because the market they describe is a New York market: 09:30 and
16:00 are NY wall-clock facts, and `feature_engineering.announcement_timing` classifies
BMO/AMC by comparing the clock against those two numbers. Anything written into either
column must therefore be produced by `now_ny()` (or converted with `to_ny_wall_clock`),
never by `datetime.now()`.

Why this matters
----------------
`datetime.now()` returns the wall clock of whichever machine happens to run the pipeline.
The refresh rule in `ingestion.fetch_earnings_dates` compares `announce_ts_observed_at`
directly against `announce_ts_ny` to decide whether a stored timestamp is still a
SCHEDULE (refreshable) or an after-the-fact OBSERVATION (frozen forever). On a host east
of New York — UTC, or Israel at UTC+2/+3 — a machine-local `datetime.now()` taken hours
BEFORE an announcement reads as a number LARGER than the NY announcement time, so a
schedule is misclassified as an observation and frozen permanently. The classification
must depend on the New York event clock, not on the host's timezone.

`ingested_at` is deliberately NOT covered by this convention. It is a legacy operational
column written by several ingesters in machine-local time, and its historical values
cannot be reinterpreted after the fact; see `MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS`.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# The widest a machine-local wall clock can run AHEAD of New York's: UTC+14 (Kiribati)
# against New York on EST (UTC-5) is 19 hours. Used to interpret legacy `ingested_at`
# values, whose timezone convention is unknown and unrecoverable — subtracting this makes
# `ingested_at - MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS` a guaranteed LOWER BOUND on the same
# instant expressed as NY wall clock, whatever host wrote it. See
# `ingestion.fetch_earnings_dates._REFRESH_ANNOUNCE_TS_SQL` for how the bound is used and
# why erring toward "still a schedule" is the conservative direction.
MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS = 19


def now_ny() -> datetime:
    """Current New York wall-clock time, naive — the only clock this project observes.

    Identical on every host: it reads the absolute instant and renders it in
    America/New_York, so a UTC box, an Israeli laptop and a New York server all produce
    the same value at the same moment. DST is handled by the zone itself, so the returned
    wall clock is what a clock on a Manhattan wall would read.
    """
    return datetime.now(NY_TZ).replace(tzinfo=None)


def to_ny_wall_clock(dt: datetime) -> datetime:
    """Render an instant as naive New York wall clock.

    A tz-aware datetime is converted; a naive one is assumed to ALREADY be NY wall clock
    and returned unchanged, because there is nothing to convert it from. Never pass a
    naive machine-local timestamp here — that assumption is exactly the bug this module
    exists to prevent.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(NY_TZ).replace(tzinfo=None)


def utc_to_ny_wall_clock(dt: datetime) -> datetime:
    """Render a naive UTC instant as naive New York wall clock."""
    return dt.replace(tzinfo=timezone.utc).astimezone(NY_TZ).replace(tzinfo=None)
