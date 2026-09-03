# scripts/full_workflow.sh
# Full Monday workflow: sync DB, run pipeline, generate chart, check last week's results.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
HARBOR_WEBPAGE="/var/www/harbor_webpage"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== [1/5] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/db/breakwater.duckdb" "$LOCAL_REPO/db/breakwater.duckdb"

echo "=== [2/5] Running full pipeline ==="
cd "$LOCAL_REPO"
"$VENV" main.py

RUN_DIR="$LOCAL_REPO/output/output_$(date +%Y_%m_%d)"

echo ""
echo "=== [3/5] Pushing output parquets to droplet ==="
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$LOCAL_REPO/output/upcoming_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"

echo ""
echo "=== [4/5] Pushing recent_calls.json to harbor_webpage ==="
rsync -avz "$RUN_DIR/recent_calls.json" "$REMOTE:$HARBOR_WEBPAGE/recent_calls.json"

echo ""
echo "=== [5/5] Sending weekly digest ==="
# Sent from here, not from a droplet cron. The PDF attachments come from the run
# above (stage5 writes them locally, the droplet never runs stage5) and the
# DIGEST_SMTP_* credentials live in the local .env. Running it here also means the
# email can only ever describe the run that just produced it — a Monday cron could
# fire before this script and mail last week's tiers with nothing to show for it.
# The digest refuses to send if upcoming_df.parquet is more than 24h old.
"$VENV" -m cron.cron_weekly_digest

echo ""
echo "=== Done. ==="
