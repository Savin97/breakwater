# scripts/monday_workflow.sh
# Full Monday workflow: sync DB, run pipeline, generate chart, check last week's results.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== [1/4] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/data/breakwater.duckdb" "$LOCAL_REPO/data/breakwater.duckdb"

echo "=== [2/4] Running full pipeline ==="
cd "$LOCAL_REPO"
"$VENV" main.py

echo "=== [3/4] Generating weekly chart ==="
"$VENV" -c "from report.chart_weekly import generate_weekly_earnings_chart; generate_weekly_earnings_chart()"

echo "=== [4/4] Pushing output parquets to droplet ==="
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$LOCAL_REPO/output/upcoming_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"

echo ""
echo "=== Last week's results ==="
"$VENV" scripts/results_check.py

echo ""
echo "=== weekly_chart.png is at output/weekly_chart.png ==="

echo ""
echo "=== Done. ==="
