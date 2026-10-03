# scripts/sync_pipeline.sh
# Pull DuckDB from droplet, run full pipeline, push output parquets back.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"

echo "=== Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/db/breakwater.duckdb" "$LOCAL_REPO/db/breakwater.duckdb"

echo "=== Done ==="
