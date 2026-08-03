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

## Cron schedule (droplet)

```
# IV snapshots — 4x daily during market hours (10am, 12pm, 2pm, 3:30pm ET)
0 14,16,18 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1
30 19      * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1

# EPS estimates — once daily at 10am ET
0 14 * * 1-5         cd /var/www/breakwater && .venv/bin/python -m cron.cron_eps_estimates >> /var/log/breakwater/eps_estimates.log 2>&1

0 6 * * *       cd /var/www/breakwater && .venv/bin/python -m cron.cron_ingest >> /var/log/breakwater_ingest.log 2>&1
0 7 * * 1       cd /var/www/breakwater && .venv/bin/python -m cron.cron_weekly_digest >> /var/log/breakwater_digest.log 2>&1
```

Breakwater repo lives at `/var/www/breakwater`. Deploy by pushing locally + `git pull` on droplet.

Website repo (harbor-markets.com): local at `/home/Michael/projects/harbor_webpage`, server at `/var/www/harbor_webpage`, GitHub: `Savin97/harbor_webpage` (renamed from `cv_website` on 2026-06-01).

## Stale tickers

Stocks present in DuckDB but removed from `data/stock_list.csv` (likely S&P 500 removals): **BK, CTRA, DAY, HOLX, LW, MOH, MTCH, PAYC**. They still appear in scoring output. No fix applied yet.
