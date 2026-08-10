# scripts/monday_workflow.sh
# Full Monday workflow: sync DB, run pipeline, generate chart, check last week's results.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
HARBOR_WEBPAGE="/var/www/harbor_webpage"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== [1/4] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/data/breakwater.duckdb" "$LOCAL_REPO/data/breakwater.duckdb"

echo "=== [2/4] Running full pipeline ==="
cd "$LOCAL_REPO"
"$VENV" main.py

RUN_DIR="$LOCAL_REPO/output/output_$(date +%Y_%m_%d)"

echo "=== [3/4] Pushing output parquets to droplet ==="
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$LOCAL_REPO/output/upcoming_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"

echo "=== [4/4] Pushing recent_calls.json to harbor_webpage ==="
rsync -avz "$RUN_DIR/recent_calls.json" "$REMOTE:$HARBOR_WEBPAGE/recent_calls.json"


echo ""
echo "=== Charts at $RUN_DIR/weekly_chart.png and $RUN_DIR/results_chart.png ==="

echo ""
echo "=== Done. ==="
