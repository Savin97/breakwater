---
name: infra-digitalocean
description: Production infrastructure — DigitalOcean droplet runs daily cron jobs for data ingestion
metadata:
  type: project
---

IV collection (`cron_iv.py`) runs 4x/day during market hours on a DigitalOcean droplet via cron.
EPS estimate collection (`cron_eps_estimates.py`) runs once daily. The main ingestion
(`cron_ingest.py`) also runs there.

**Why:** Local machine is not always on; droplet keeps data collection running continuously.
**How to apply:** Don't assume scripts need to be run locally. Cron jobs on the droplet handle ingestion — pipeline/reports run separately (likely locally or manually triggered).

**2026-08-02 incident:** when the crontab was updated (~2026-06-28/29) to add intraday IV
snapshots and `cron_eps_estimates`, the new entries invoked the scripts by direct file path
(`python /var/www/breakwater/cron/cron_iv.py`, no `cd`, no `-m`). Running a script by path puts
*its own directory* (`cron/`) on `sys.path`, not the repo root — so `from config import DB_PATH`
fails with `ModuleNotFoundError: No module named 'config'` on every single invocation. Confirmed
via `/var/log/breakwater/iv.log` and `eps_estimates.log` — both are just the same traceback
repeated on every run since. This is why `iv_snapshots` stalled at 2026-06-26 (last run before
the crontab change) and `eps_estimates` has never had a row. **Fix:** always invoke cron scripts
the same way `cron_ingest`/`cron_weekly_digest` do — `cd /var/www/breakwater && .../python -m
cron.<module>` — never by direct file path. Schedule below reflects the corrected form.

**2026-08-12 incident (DuckDB write lock):** `cron_iv` and `cron_eps_estimates` were both
scheduled on the same minute. DuckDB allows only one read-write connection to the file, so the
loser of the race died outright — `cron_iv` lost its first slot every weekday (zero rows for that
hour, not a partial batch). **Fix:** never schedule two cron jobs that open the DB on the same
minute; stagger them (EPS now runs 10 min before IV's first slot).

**2026-08-28 incident (timezone):** droplet system TZ is **UTC**. The crontab had been rewritten
with ET-intended times (`0 10,12,14` + `30 15`) but no `CRON_TZ`, so cron read them as UTC —
10:00 and 12:00 UTC are 6am/8am ET, i.e. **pre-market**, when options have no bid/ask and Yahoo
returns placeholder `iv≈0.00001, bid=ask=0`. `fetch_iv.py`'s zero-ask guard correctly skipped
every stock, so those two runs logged `inserted=0 skipped=N` daily and only the 14:00/15:30 UTC
slots produced data.

**`CRON_TZ` DOES NOT WORK HERE — do not try it again.** The droplet runs Ubuntu 24.04 with
`cron 3.0pl1-184ubuntu2` (Debian vixie-cron), which does **not** support `CRON_TZ` as a
scheduling directive — that is a cronie/RHEL feature. Debian-family cron just exports it to the
job as an ordinary env var, so scheduling stays on system TZ (UTC) and the line silently does
nothing. Verified 2026-08-28: with `CRON_TZ=America/New_York` installed, syslog still showed
`cron_iv` firing at 15:30:01 UTC for a `30 15` entry, and EPS at 10:10 UTC for `10 10`.

**Actual fix (applied 2026-08-28):** write cron times in **UTC**, chosen inside
**14:30–20:00 UTC** — the window that falls within US market hours (9:30–16:00 ET) under both
EDT (UTC-4) and EST (UTC-5), so it needs no DST maintenance. Wall-clock ET drifts an hour across
DST; that is fine, market-hours coverage is what matters. Rejected alternative:
`timedatectl set-timezone America/New_York` would make cron times genuinely ET and DST-correct,
but it changes `datetime.now()` process-wide, so `snapshot_hour`/`ingested_at` would flip to ET
and break continuity with the existing UTC-stamped history.

**Note:** `snapshot_hour` is `datetime.now().hour` on a UTC box, so it stores **UTC hours**, not
ET — all historical values are UTC. Under the current schedule expect 15/16/18/19.

## Cron schedule (droplet) — ALL TIMES UTC

Backup of the pre-fix crontab: `/root/crontab.backup.20260828`.

```
# EPS estimates — 15 min before first IV slot (DuckDB single-writer stagger)
45 14 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_eps_estimates >> /var/log/breakwater/eps_estimates.log 2>&1

# IV snapshots — 4x during market hours
# summer (EDT) 11:00/12:30/14:00/15:30 ET · winter (EST) 10:00/11:30/13:00/14:30 ET
0  15 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1
30 16 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1
0  18 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1
30 19 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1

0 6 * * *       cd /var/www/breakwater && .venv/bin/python -m cron.cron_ingest >> /var/log/breakwater_ingest.log 2>&1
# 0 7 * * 1     cd /var/www/breakwater && .venv/bin/python -m cron.cron_weekly_digest >> ... (currently commented out)
```

Breakwater repo lives at `/var/www/breakwater`. Deploy by pushing locally + `git pull` on droplet.

Website repo (harbor-markets.com): local at `/home/Michael/projects/harbor_webpage`, server at `/var/www/harbor_webpage`, GitHub: `Savin97/harbor_webpage` (renamed from `cv_website` on 2026-06-01).

## Stale tickers — RESOLVED 2026-08-10

Was: tickers removed from the S&P 500 lingered in DuckDB and kept appearing in scoring output.
Fixed by the ticker-lifecycle work (`edb84c2`): `ingestion/fetch_sp500_sectors.py` now reconciles
the full ticker universe against the live Wikipedia S&P 500 list each run and marks each
`active`/`inactive` (with a `reason` — renamed per `data/ticker_renames.csv`, or removed from
index) in `stock_data.status`. `read_stocks_to_fetch(con, active_only=True)` filters inactive
tickers out of incremental price/earnings ingestion. Deployed on droplet; currently 501 active,
16 inactive.
