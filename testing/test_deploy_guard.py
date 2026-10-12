"""Only `master` may publish to production (scripts/deploy_guard.sh).

`scripts/full_workflow.sh` is run end to end in a throwaway git repo, with `rsync` and
the project Python replaced by stubs that only record their arguments. From any branch
but master the local steps run (DB pull, pipeline) and nothing is pushed or mailed.

Run with:  pytest testing/test_deploy_guard.py -v
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _sandbox(tmp_path: Path, branch: str | None) -> Path:
    """A copy of the workflow + guard in a git repo on `branch` (None = detached HEAD),
    with stub rsync / python that log their calls."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for f in ("full_workflow.sh", "deploy_guard.sh"):
        shutil.copy(REPO / "scripts" / f, root / "scripts" / f)
    stub = '#!/bin/sh\necho "$0 $*" >> "{log}"\nexit 0\n'
    (root / ".venv" / "bin").mkdir(parents=True)
    py = root / ".venv" / "bin" / "python"
    py.write_text(stub.format(log=tmp_path / "calls.log"))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rsync").write_text(stub.format(log=tmp_path / "calls.log"))
    for exe in (py, bindir / "rsync"):
        exe.chmod(0o755)

    git = ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git[:3] + ["init", "-q", "-b", "master"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "init"], check=True)
    if branch is None:
        subprocess.run(git + ["checkout", "-q", "--detach"], check=True)
    elif branch != "master":
        subprocess.run(git + ["checkout", "-q", "-b", branch], check=True)
    return root


def _run(tmp_path, branch):
    root = _sandbox(tmp_path, branch)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(root / "scripts" / "full_workflow.sh")],
                       capture_output=True, text=True, env=env)
    log = (tmp_path / "calls.log").read_text() if (tmp_path / "calls.log").exists() else ""
    return r, log


@pytest.mark.parametrize("branch", ["methodology-rebuild", "feature-x", None])
def test_non_master_runs_locally_but_publishes_nothing(tmp_path, branch):
    r, log = _run(tmp_path, branch)
    assert r.returncode == 3, r.stdout + r.stderr
    shown = branch if branch else "HEAD"
    assert f"REFUSING TO DEPLOY — current branch: {shown}" in r.stderr
    rsyncs = [l for l in log.splitlines() if "/rsync " in l]
    assert len(rsyncs) == 1 and "root@harbor-markets.com:/var/www/breakwater/db/" in rsyncs[0]
    assert "main.py" in log                                  # local scoring still ran
    assert "cron.cron_weekly_digest" not in log              # no subscriber email
    assert "publishing.archive_calls" not in log             # no canonical call archived
    assert "/output/" not in rsyncs[0] and "recent_calls" not in log


def test_master_publishes(tmp_path):
    r, log = _run(tmp_path, "master")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSING" not in r.stderr
    assert "root@harbor-markets.com:/var/www/breakwater/output/" in log
    assert "cron.cron_weekly_digest" in log


def test_master_archives_after_the_calls_go_public_and_before_the_digest(tmp_path):
    """Order on master: push dashboard parquets (upcoming_df last) -> archive exactly those
    calls with --published -> digest. The archive is never written before the push."""
    _, log = _run(tmp_path, "master")
    lines = log.splitlines()
    push = max(i for i, l in enumerate(lines) if "upcoming_df.parquet" in l and "/rsync " in l)
    archive = next(i for i, l in enumerate(lines) if "publishing.archive_calls" in l)
    digest = next(i for i, l in enumerate(lines) if "cron.cron_weekly_digest" in l)
    assert push < archive < digest
    assert lines[archive].rstrip().endswith("--published")


def test_guard_outside_a_git_checkout_refuses(tmp_path):
    r = subprocess.run(["bash", "-c", f'source "{REPO}/scripts/deploy_guard.sh"; '
                                      f'deploy_branch_ok "{tmp_path}"'],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSING TO DEPLOY" in r.stderr


def test_every_push_in_the_workflow_sits_behind_the_guard():
    """Static: each rsync TO the remote and the digest come after the guard; the only
    rsync before it pulls FROM the remote."""
    lines = (REPO / "scripts" / "full_workflow.sh").read_text().splitlines()
    guard = next(i for i, l in enumerate(lines) if "deploy_branch_ok" in l and "if" in l)
    for i, l in enumerate(lines):
        if l.strip().startswith("rsync") and i < guard:
            assert l.strip().startswith('rsync -avz "$REMOTE:'), f"push before the guard: {l}"
        if ("cron.cron_weekly_digest" in l or "publishing.archive_calls" in l) \
                and not l.lstrip().startswith("#"):
            assert i > guard, l
    assert "exit 3" in lines[guard + 1]
