"""Only `master` may publish to production (scripts/deploy_guard.sh), and publication is
verified before anything is archived or mailed.

`scripts/full_workflow.sh` is run end to end in a throwaway git repo, with `rsync`, `ssh`
and the project Python replaced by stubs that log their arguments. From any branch but
master the local steps run (DB pull, pipeline) and nothing is pushed or mailed. On master
the order is: build generation -> droplet code check -> upload -> atomic activation ->
read back what is served -> receipt -> archive -> digest; a failed check stops the run.

Run with:  pytest testing/test_deploy_guard.py -v
"""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = "20261019T120000Z-abcdef0123"


def _sandbox(tmp_path: Path, branch: str | None) -> Path:
    """A copy of the workflow + guard in a git repo on `branch` (None = detached HEAD),
    with stub rsync / ssh / python that log their calls."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for f in ("full_workflow.sh", "deploy_guard.sh"):
        shutil.copy(REPO / "scripts" / f, root / "scripts" / f)
    log = tmp_path / "calls.log"
    (root / ".venv" / "bin").mkdir(parents=True)
    py = root / ".venv" / "bin" / "python"
    py.write_text(f"""#!/bin/sh
echo "PY $*" >> "{log}"
case "$*" in
  *"publishing.generation build"*) mkdir -p "$PWD/output/publication/{GEN}"; echo "Built"; echo "{GEN}";;
  *"publishing.generation receipt"*) exit "${{STUB_RECEIPT_EXIT:-0}}";;
  *"publishing.generation check-live"*) exit "${{STUB_CHECK_EXIT:-0}}";;
esac
exit 0
""")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rsync").write_text(f'#!/bin/sh\necho "RSYNC $*" >> "{log}"\nexit 0\n')
    (bindir / "ssh").write_text(f"""#!/bin/sh
echo "SSH $*" >> "{log}"
case "$*" in
  *"rev-parse HEAD"*) echo "${{STUB_REMOTE_HEAD}}";;
  *"cat "*CURRENT_GENERATION*) if grep -q "mv -f CURRENT_GENERATION.tmp" "{log}"; then echo "{GEN}"; else echo "${{STUB_LIVE_GEN:-gen-previous}}"; fi;;
  *sha256sum*) echo "deadbeef  /x/upcoming_df.parquet";;
esac
exit 0
""")
    for exe in (py, bindir / "rsync", bindir / "ssh"):
        exe.chmod(0o755)

    git = ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git[:3] + ["init", "-q", "-b", "master"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "init"], check=True)
    if branch is None:
        subprocess.run(git + ["checkout", "-q", "--detach"], check=True)
    elif branch != "master":
        subprocess.run(git + ["checkout", "-q", "-b", branch], check=True)
    return root


def _run(tmp_path, branch, remote_head=None, receipt_exit=0, check_exit=0):
    root = _sandbox(tmp_path, branch)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
           "STUB_REMOTE_HEAD": remote_head or head, "STUB_RECEIPT_EXIT": str(receipt_exit),
           "STUB_CHECK_EXIT": str(check_exit)}
    r = subprocess.run(["bash", str(root / "scripts" / "full_workflow.sh")],
                       capture_output=True, text=True, env=env)
    log = (tmp_path / "calls.log").read_text() if (tmp_path / "calls.log").exists() else ""
    return r, log.splitlines()


def _rsync_dest(line: str) -> str:
    """The transfer destination of a logged rsync call: its last argument. Checking the
    whole command for '/output/' falsely failed when the temp dir itself contained it."""
    return shlex.split(line)[-1]


def _first(lines, needle):
    return next(i for i, l in enumerate(lines) if needle in l)


@pytest.mark.parametrize("branch", ["methodology-rebuild", "feature-x", None])
def test_non_master_runs_locally_but_publishes_nothing(tmp_path, branch):
    r, lines = _run(tmp_path, branch)
    assert r.returncode == 3, r.stdout + r.stderr
    shown = branch if branch else "HEAD"
    assert f"REFUSING TO DEPLOY — current branch: {shown}" in r.stderr
    rsyncs = [l for l in lines if l.startswith("RSYNC ")]
    assert len(rsyncs) == 1
    assert _rsync_dest(rsyncs[0]).endswith("/db/breakwater.duckdb")         # a pull, local
    assert not any(l.startswith("SSH ") for l in lines)                      # nothing remote
    assert any("main.py" in l for l in lines)                                # scoring ran
    assert not any("cron.cron_weekly_digest" in l for l in lines)            # no email
    assert not any("publishing.archive_calls" in l for l in lines)           # no archive
    assert not any("publishing.generation" in l for l in lines)


def test_the_destination_check_survives_a_temp_dir_named_output(tmp_path):
    """Regression for the auditor's false failure: a temp path containing /output/."""
    nested = tmp_path / "output" / "x"
    nested.mkdir(parents=True)
    r, lines = _run(nested, "feature-x")
    assert r.returncode == 3
    rsyncs = [l for l in lines if l.startswith("RSYNC ")]
    assert len(rsyncs) == 1 and _rsync_dest(rsyncs[0]).endswith("/db/breakwater.duckdb")


def test_master_publishes_in_the_verified_order(tmp_path):
    r, lines = _run(tmp_path, "master")
    assert r.returncode == 0, r.stdout + r.stderr
    build = _first(lines, "publishing.generation build")
    code = _first(lines, "rev-parse HEAD")
    upload = next(i for i, l in enumerate(lines) if l.startswith("RSYNC ")
                  and _rsync_dest(l).endswith(f"/output/generations/{GEN}/"))
    check = _first(lines, "publishing.generation check-live gen-previous")
    activate = _first(lines, "mv -f CURRENT_GENERATION.tmp CURRENT_GENERATION")
    readback = max(i for i, l in enumerate(lines) if "cat /var/www/breakwater/output/CURRENT_GENERATION" in l)
    receipt = _first(lines, "publishing.generation receipt")
    archive = _first(lines, "publishing.archive_calls")
    digest = _first(lines, "cron.cron_weekly_digest")
    assert build < code < upload < check < activate < readback < receipt < archive < digest
    assert "rm -f upcoming_df.parquet streamlit_df.parquet" in lines[activate]
    assert f"receipt {GEN} --served-pointer {GEN} --served-sha deadbeef" in lines[receipt]
    assert lines[archive].rstrip().endswith(f"--published --generation {GEN}")
    assert lines[digest].rstrip().endswith(f"cron.cron_weekly_digest --generation {GEN}")


def test_droplet_running_other_code_stops_before_anything_goes_public(tmp_path):
    r, lines = _run(tmp_path, "master", remote_head="0" * 40)
    assert r.returncode == 4 and "droplet code is" in r.stderr
    assert not any(l.startswith("RSYNC ") and "/output/" in _rsync_dest(l) for l in lines)
    assert not any("CURRENT_GENERATION" in l for l in lines)
    assert not any("publishing.archive_calls" in l or "cron.cron_weekly_digest" in l for l in lines)


def test_a_failed_served_check_archives_and_mails_nothing(tmp_path):
    r, lines = _run(tmp_path, "master", receipt_exit=4)
    assert r.returncode != 0
    assert any("publishing.generation receipt" in l for l in lines)
    assert not any("publishing.archive_calls" in l for l in lines)
    assert not any("cron.cron_weekly_digest" in l for l in lines)


def test_an_unrecorded_live_generation_is_never_replaced(tmp_path):
    """The live generation went public but its run died before receipt/archive: the next
    run must not switch the dashboard to a new generation (and then archive that one as the
    first public call)."""
    r, lines = _run(tmp_path, "master", check_exit=5)
    assert r.returncode == 5
    assert any("check-live gen-previous" in l for l in lines)
    assert not any("mv -f CURRENT_GENERATION" in l for l in lines)
    assert not any("publishing.archive_calls" in l or "cron.cron_weekly_digest" in l for l in lines)


def test_guard_outside_a_git_checkout_refuses(tmp_path):
    r = subprocess.run(["bash", "-c", f'source "{REPO}/scripts/deploy_guard.sh"; '
                                      f'deploy_branch_ok "{tmp_path}"'],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSING TO DEPLOY" in r.stderr


def test_every_push_in_the_workflow_sits_behind_the_guard():
    """Static: every rsync TO the remote, every ssh, the generation, archive and digest
    steps come after the guard; the only rsync before it pulls FROM the remote."""
    lines = (REPO / "scripts" / "full_workflow.sh").read_text().splitlines()
    guard = next(i for i, l in enumerate(lines) if "deploy_branch_ok" in l and "if" in l)
    for i, l in enumerate(lines):
        code = l.strip()
        if code.startswith("#"):
            continue
        if code.startswith("rsync") and i < guard:
            assert code.startswith('rsync -avz "$REMOTE:'), f"push before the guard: {l}"
        if any(k in code for k in ("ssh ", "cron.cron_weekly_digest", "publishing.archive_calls",
                                   "publishing.generation")):
            assert i > guard, l
    assert "exit 3" in lines[guard + 1]
