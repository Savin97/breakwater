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
