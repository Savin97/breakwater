# scripts/deploy_guard.sh — sourced by scripts that publish to production.
#
# Only `master` is production. A run from any other branch (methodology-rebuild, a
# feature branch, a detached HEAD) may score, write reports and validate locally, but must
# never push artifacts to the droplet or the website, or mail subscribers. There is no
# override: to deploy, merge to master and run from master.

# deploy_branch_ok [repo_dir] — returns 0 on master; otherwise prints a loud refusal
# naming the current branch and returns 1.
deploy_branch_ok() {
  local repo="${1:-.}" branch
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null)" || branch="(not a git checkout)"
  if [ "$branch" = "master" ]; then
    return 0
  fi
  {
    echo ""
    echo "=============================================================================="
    echo " REFUSING TO DEPLOY — current branch: ${branch}"
    echo "=============================================================================="
    echo " Production artifacts (dashboard parquets, recent_calls.json, the weekly digest"
    echo " and its PDF reports) are published only from 'master'. Everything local from"
    echo " this run (DB pull, scoring, reports, parquets under output/) is complete."
    echo " To deploy: merge to master, check it out, and run this script again."
    echo "=============================================================================="
  } >&2
  return 1
}
