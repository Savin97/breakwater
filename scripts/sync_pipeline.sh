# scripts/sync_pipeline.sh
# Pull DuckDB from droplet, run full pipeline, push output parquets back.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== [1/3] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/data/breakwater.duckdb" "$LOCAL_REPO/data/breakwater.duckdb"

echo "=== [2/3] Running pipeline ==="
cd "$LOCAL_REPO"
"$VENV" main.py

echo "=== [3/3] Pushing output parquets to droplet ==="
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$LOCAL_REPO/output/upcoming_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"

echo "=== Done ==="
