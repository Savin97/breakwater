# scripts/full_workflow.sh
# Full Monday workflow — the ONLY production publication path.
#   1-2  local on any branch: pull the DB, score, write reports / parquets under output/
#   ---  scripts/deploy_guard.sh: everything below runs on master only (else exit 3)
#   3    freeze the outputs into an immutable generation  (publishing/generation.py build)
#   4    check the droplet runs this exact commit          (else exit 4: deploy the code first)
#   5    upload the generation, then make it live with one atomic pointer switch
#   6    read back what the droplet serves; write the publication receipt only if it matches
#   7    archive that generation's calls                   -> db/predictions.duckdb
#   8    push recent_calls.json (paused)
#   9    mail the digest (re-checks the branch itself before a real send)
# Any failure stops the run (set -e): nothing is archived or mailed unless the generation
# was verified as the one being served.

set -e

REMOTE="root@harbor-markets.com"
REMOTE_REPO="/var/www/breakwater"
HARBOR_WEBPAGE="/var/www/harbor_webpage"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$LOCAL_REPO/.venv/bin/python"
source "$LOCAL_REPO/scripts/deploy_guard.sh"

echo "=== [1/9] Pulling DuckDB from droplet ==="
rsync -avz "$REMOTE:$REMOTE_REPO/db/breakwater.duckdb" "$LOCAL_REPO/db/breakwater.duckdb"

echo "=== [2/9] Running full pipeline (scoring + local outputs; publishes nothing) ==="
cd "$LOCAL_REPO"
"$VENV" main.py

RUN_DIR="$LOCAL_REPO/output/output_$(date +%Y_%m_%d)"

# Everything below publishes: the droplet's dashboard, the website, subscribers' inboxes.
# Only master may do that (scripts/deploy_guard.sh). Exit 3 = refused, local run complete.
if ! deploy_branch_ok "$LOCAL_REPO"; then
  exit 3
fi

echo ""
echo "=== [3/9] Freezing the publication generation ==="
# An immutable copy of exactly what goes public (calls + history + manifest with hashes,
# model version and reference checksum). Everything below refers to it by id.
GEN_ID="$("$VENV" -m publishing.generation build | tail -n 1)"
GEN_DIR="$LOCAL_REPO/output/publication/$GEN_ID"
if [ -z "$GEN_ID" ] || [ ! -d "$GEN_DIR" ]; then
  echo "REFUSING TO PUBLISH: no generation was built." >&2
  exit 4
fi

echo ""
echo "=== [4/9] Checking the droplet runs this code ==="
# The dashboard must understand the files it is about to serve. Deploy code first.
LOCAL_HEAD="$(git -C "$LOCAL_REPO" rev-parse HEAD)"
REMOTE_HEAD="$(ssh "$REMOTE" "git -C $REMOTE_REPO rev-parse HEAD")"
if [ "$REMOTE_HEAD" != "$LOCAL_HEAD" ]; then
  echo "REFUSING TO PUBLISH: droplet code is $REMOTE_HEAD, this run is $LOCAL_HEAD." >&2
  echo "Pull the same master commit on the droplet, then rerun." >&2
  exit 4
fi

echo ""
echo "=== [5/9] Uploading generation $GEN_ID and making it live ==="
# full_df.parquet is not served by the dashboard; it goes to the droplet as before.
rsync -avz "$LOCAL_REPO/output/full_df.parquet" "$REMOTE:$REMOTE_REPO/output/"
# Uploaded into its own directory first: invisible until the pointer names it.
rsync -avz "$GEN_DIR/" "$REMOTE:$REMOTE_REPO/output/generations/$GEN_ID/"
# Never replace a live generation that is not fully recorded (receipt + archive): its calls
# were public. check-live exits 5 and prints how to finish it.
LIVE_GEN="$(ssh "$REMOTE" "cat $REMOTE_REPO/output/CURRENT_GENERATION 2>/dev/null || true")"
"$VENV" -m publishing.generation check-live "$LIVE_GEN"
# One atomic rename: viewers see the old generation or the new one, never a mix. The
# pre-generation flat files are removed so a dashboard still running old code shows no
# calls rather than stale ones.
ssh "$REMOTE" "cd $REMOTE_REPO/output && printf '%s\\n' '$GEN_ID' > CURRENT_GENERATION.tmp && mv -f CURRENT_GENERATION.tmp CURRENT_GENERATION && rm -f upcoming_df.parquet streamlit_df.parquet"

echo ""
echo "=== [6/9] Verifying what the droplet serves ==="
SERVED_POINTER="$(ssh "$REMOTE" "cat $REMOTE_REPO/output/CURRENT_GENERATION")"
SERVED_SHA="$(ssh "$REMOTE" "sha256sum $REMOTE_REPO/output/generations/$SERVED_POINTER/upcoming_df.parquet" | cut -d' ' -f1)"
# Writes output/publication/$GEN_ID/receipt.json (published_at, UTC) only if the served
# pointer and calls file match the generation; otherwise exits non-zero and the run stops.
"$VENV" -m publishing.generation receipt "$GEN_ID" --served-pointer "$SERVED_POINTER" --served-sha "$SERVED_SHA"

echo ""
echo "=== [7/9] Archiving the published calls ==="
# Exactly the verified generation, stamped with its receipt's publication time. This is
# the ONLY writer of the canonical archive, and it refuses anything without a receipt.
"$VENV" -m publishing.archive_calls --published --generation "$GEN_ID"

echo ""
echo "=== [8/9] Pushing recent_calls.json to harbor_webpage ==="
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
echo "=== [9/9] Sending weekly digest ==="
# Sent from here, not from a droplet cron. The PDF attachments come from the run
# above (stage5 writes them locally, the droplet never runs stage5) and the
# DIGEST_SMTP_* credentials live in the local .env. Running it here also means the
# email can only ever describe the run that just produced it — a Monday cron could
# fire before this script and mail last week's tiers with nothing to show for it.
# It mails the verified generation's calls and reports (never output/, which may have
# changed since) and refuses one that went live more than 24h ago.
"$VENV" -m cron.cron_weekly_digest --generation "$GEN_ID"

echo ""
echo "=== Done. ==="
