# scripts/full_workflow.sh
# Full Monday workflow — the ONLY production publication path.
#   1-2  local on any branch: pull the DB, score, write reports / parquets under output/
#   ---  scripts/deploy_guard.sh: everything below runs on master only (else exit 3)
#   3    push the dashboard parquets            -> calls become public
#   4    archive exactly those calls            -> db/predictions.duckdb (publishing/archive_calls.py)
#   5    push recent_calls.json (paused)
#   6    mail the digest (re-checks the branch itself before a real send)

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
HARBOR_WEBPAGE="/var/www/harbor_webpage"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"
source "$LOCAL_REPO/scripts/deploy_guard.sh"

echo "=== [1/6] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/db/breakwater.duckdb" "$LOCAL_REPO/db/breakwater.duckdb"

echo "=== [2/6] Running full pipeline (scoring + local outputs; publishes nothing) ==="
cd "$LOCAL_REPO"
"$VENV" main.py

RUN_DIR="$LOCAL_REPO/output/output_$(date +%Y_%m_%d)"

# Everything below publishes: the droplet's dashboard, the website, subscribers' inboxes.
# Only master may do that (scripts/deploy_guard.sh). Exit 3 = refused, local run complete.
if ! deploy_branch_ok "$LOCAL_REPO"; then
  exit 3
fi

echo ""
echo "=== [3/6] Pushing output parquets to droplet ==="
# upcoming_df.parquet (the calls) goes last: if the push dies part-way, the calls are the
# file least likely to have reached the dashboard.
rsync -avz \
  "$LOCAL_REPO/output/full_df.parquet" \
  "$LOCAL_REPO/output/streamlit_df.parquet" \
  "$REMOTE:$REMOTE_REPO/output/"
rsync -avz "$LOCAL_REPO/output/upcoming_df.parquet" "$REMOTE:$REMOTE_REPO/output/"

echo ""
echo "=== [4/6] Archiving the published calls ==="
# The calls are public once the dashboard push above has succeeded (set -e stops the run
# before this line if it failed). Archive exactly that file — output/upcoming_df.parquet —
# into db/predictions.duckdb. This is the ONLY writer of the canonical archive.
"$VENV" -m publishing.archive_calls --published

echo ""
echo "=== [5/6] Pushing recent_calls.json to harbor_webpage ==="
# The public track record generator is PAUSED until Model B (0.4.0) has a prospective
# record, so stage 5 writes no recent_calls.json and there is nothing to publish. Skip
# instead of failing: `set -e` would abort the run here and the weekly digest below would
# never be sent, which the pause is not supposed to do. When the generator is un-paused,
# the file reappears and this pushes it again with no further change.
if [ -f "$RUN_DIR/recent_calls.json" ]; then
  rsync -avz "$RUN_DIR/recent_calls.json" "$REMOTE:$HARBOR_WEBPAGE/recent_calls.json"
else
  echo "SKIPPED: no $RUN_DIR/recent_calls.json."
  echo "         The public track record is paused pending prospective 0.4.0 evidence —"
  echo "         see marketing/generate_public_track_record.py."
  echo "         The landing page keeps serving whatever it already has; nothing new is published."
fi

echo ""
echo "=== [6/6] Sending weekly digest ==="
# Sent from here, not from a droplet cron. The PDF attachments come from the run
# above (stage5 writes them locally, the droplet never runs stage5) and the
# DIGEST_SMTP_* credentials live in the local .env. Running it here also means the
# email can only ever describe the run that just produced it — a Monday cron could
# fire before this script and mail last week's tiers with nothing to show for it.
# The digest refuses to send if upcoming_df.parquet is more than 24h old.
"$VENV" -m cron.cron_weekly_digest

echo ""
echo "=== Done. ==="
