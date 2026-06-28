# scripts/monday_workflow.sh
# Full Monday workflow: sync DB, run pipeline, generate chart, check last week's results.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
HARBOR_WEBPAGE="/var/www/harbor_webpage"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== [1/5] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/data/breakwater.duckdb" "$LOCAL_REPO/data/breakwater.duckdb"

echo "=== [2/5] Running full pipeline ==="
cd "$LOCAL_REPO"
"$VENV" main.py

echo "=== [3/5] Generating weekly chart + recent calls JSON ==="
"$VENV" -c "from analysis.chart_weekly import generate_weekly_earnings_chart; generate_weekly_earnings_chart()"
"$VENV" analysis/gen_recent_calls.py

echo "=== [4/5] Pushing output parquets to droplet ==="
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$LOCAL_REPO/output/upcoming_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"

echo "=== [5/5] Pushing recent_calls.json to harbor_webpage ==="
rsync -avz "$LOCAL_REPO/output/recent_calls.json" "$REMOTE:$HARBOR_WEBPAGE/recent_calls.json"

echo ""
echo "=== [5/5] Last week's results + chart ==="
"$VENV" analysis/last_week_results.py

echo ""
echo "=== Charts at output/weekly_chart.png and output/results_chart.png ==="

echo ""
echo "=== Done. ==="
