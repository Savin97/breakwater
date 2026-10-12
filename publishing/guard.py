# publishing/guard.py
"""Who may publish. Scoring and local outputs never ask; anything that makes a call
public or records it as published does.

Mirrors scripts/deploy_guard.sh: only the `master` branch is production. A run from any
other branch, a detached HEAD or a directory that is not a git checkout may score, write
reports and parquets and validate locally, but must not deploy, mail subscribers or write
the canonical prediction archive. There is no override.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_BRANCH = "master"
CANONICAL_PREDICTIONS_DB = REPO_ROOT / "db" / "predictions.duckdb"


class PublicationRefused(RuntimeError):
    """Raised when a publication step is attempted from a non-production checkout."""


def current_branch(repo: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "(not a git checkout)"


def require_production_branch(action: str, repo: Path = REPO_ROOT) -> None:
    branch = current_branch(repo)
    if branch != PRODUCTION_BRANCH:
        raise PublicationRefused(
            f"REFUSING TO {action.upper()} — current branch: {branch}. "
            f"Only '{PRODUCTION_BRANCH}' publishes; merge and run from there.")


def is_canonical_archive(db_path) -> bool:
    """True when `db_path` is the tracked production archive db/predictions.duckdb."""
    return Path(db_path).resolve() == CANONICAL_PREDICTIONS_DB.resolve()
