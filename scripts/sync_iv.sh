#!/usr/bin/env bash
# Sync the full DuckDB from the droplet (prices, earnings, stock_data, iv_snapshots).
# Run this before python main.py to ensure local data is current.
#
# Usage: bash scripts/sync_iv.sh

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "--- Syncing DuckDB from droplet ---"
rsync -avz "$REMOTE:$REMOTE_REPO/data/breakwater.duckdb" "$LOCAL_REPO/data/breakwater.duckdb"

echo "--- Done. Run python main.py to regenerate output. ---"
