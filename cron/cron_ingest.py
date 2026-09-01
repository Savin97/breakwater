# cron/cron_ingest.py
# Daily 06:00 UTC on the droplet: refresh prices + earnings dates in DuckDB.
#
# INGEST-ONLY BY DESIGN. Do not add scoring here. The droplet has ~590 MB of
# usable RAM (Streamlit holds the rest), and every scoring path exceeds it:
# reading full_df.parquet alone peaks at 391 MB, the incremental path at
# 1030 MB, a full run at 2000+ MB. Bundling scoring into this job is what
# produced the 75 OOM kills in /var/log/breakwater_ingest.log — including
# ~19 that died before ingestion even started. Ingestion on its own is small
# and has always succeeded; keeping it alone keeps the DB fresh regardless of
# whether prediction generation is running yet.
from pipeline.stage1 import stage1

stage1(incremental=True)
