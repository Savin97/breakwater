---
name: infra-digitalocean
description: Production infrastructure — DigitalOcean droplet runs daily cron jobs for data ingestion
metadata:
  type: project
---

IV collection (`cron_iv.py`) runs once a day post-close on a DigitalOcean droplet via cron.
The main ingestion (`cron_ingest.py`) also runs there.

**Why:** Local machine is not always on; droplet keeps data collection running continuously.
**How to apply:** Don't assume scripts need to be run locally. Cron jobs on the droplet handle ingestion — pipeline/reports run separately (likely locally or manually triggered).

## Cron schedule (droplet)

```
0 6 * * *       cd /var/www/Breakwater && .venv/bin/python -m cron.cron_ingest >> /var/log/breakwater_ingest.log 2>&1
30 20 * * 1-5   cd /var/www/Breakwater && .venv/bin/python -m cron.cron_iv >> /var/log/breakwater_iv.log 2>&1
0 7 * * 1       cd /var/www/Breakwater && .venv/bin/python -m cron.cron_weekly_digest >> /var/log/breakwater_digest.log 2>&1
```

Breakwater repo lives at `/var/www/Breakwater`. Deploy by pushing locally + `git pull` on droplet.

Website repo (harbor-markets.com): local at `/home/Michael/projects/harbor_webpage`, server at `/var/www/harbor_webpage`, GitHub: `Savin97/harbor_webpage` (renamed from `cv_website` on 2026-06-01).

## Stale tickers

Stocks present in DuckDB but removed from `data/stock_list.csv` (likely S&P 500 removals): **BK, CTRA, DAY, HOLX, LW, MOH, MTCH, PAYC**. They still appear in scoring output. No fix applied yet.
