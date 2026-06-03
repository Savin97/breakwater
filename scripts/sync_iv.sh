#!/usr/bin/env bash
# Pull iv_snapshots from the droplet into the local DuckDB.
# Inserts new rows only (unique index on stock, snapshot_date prevents duplicates).
# Run this before python main.py to get current IV data.
#
# Usage: bash scripts/sync_iv.sh

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/Breakwater"
REMOTE_TMP="/tmp/iv_snapshots_sync.parquet"
LOCAL_TMP="/tmp/iv_snapshots_sync.parquet"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "--- Exporting iv_snapshots on droplet ---"
ssh "$REMOTE" "cd $REMOTE_REPO && .venv/bin/python -c \"
import duckdb
con = duckdb.connect('data/breakwater.duckdb')
con.execute(\\\"COPY iv_snapshots TO '$REMOTE_TMP' (FORMAT PARQUET)\\\")
rows = con.execute('SELECT COUNT(*) FROM iv_snapshots').fetchone()[0]
con.close()
print(f'Exported {rows} rows to $REMOTE_TMP')
\""

echo "--- Copying to local ---"
scp "$REMOTE:$REMOTE_TMP" "$LOCAL_TMP"

echo "--- Merging into local DuckDB ---"
cd "$LOCAL_REPO"
.venv/bin/python - <<'PYEOF'
import duckdb, pandas as pd
from config import DB_PATH

df = pd.read_parquet("/tmp/iv_snapshots_sync.parquet")
print(f"  Received {len(df)} rows from droplet")

con = duckdb.connect(DB_PATH)
before = con.execute("SELECT COUNT(*) FROM iv_snapshots").fetchone()[0]
con.execute("INSERT OR IGNORE INTO iv_snapshots SELECT * FROM df")
after = con.execute("SELECT COUNT(*) FROM iv_snapshots").fetchone()[0]
con.close()

print(f"  Inserted {after - before} new rows ({before} → {after} total)")
PYEOF

echo "--- Done. Run python main.py to regenerate output with IV data. ---"
