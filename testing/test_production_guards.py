"""Static guards: licensed vendor data must never become a production input.

`data/vendor/` holds raw third-party data (Massive / Benzinga, SEC, options) that is
licensed and kept out of a public repo. The announcement timestamps it produced reached
production once, as a seed loaded into the `earnings` table; nothing at runtime may read
the vendor files or call the vendor. The vendor tooling itself is preserved in the tag
`methodology-audit-archive-october-2026`.

Both tests are carried over unchanged from that tag's `testing/test_massive_earnings.py`.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PRODUCTION_DIRS = ["pipeline", "feature_engineering", "scoring", "ingestion", "utilities",
                   "analysis", "streamlit_dash", "report", "cron"]


def _python_files(*dirs):
    for d in dirs:
        for f in sorted(Path(d).rglob("*.py")):
            if "__pycache__" not in f.parts:
                yield f


def _code_only(path: Path) -> str:
    """The file's CODE, with comments and docstrings removed.

    The guard must fail on code that reads a vendor file, not on a comment explaining
    that the module deliberately never does. Scanning raw source cannot tell those apart;
    this strips every docstring and, via `ast.unparse`, every comment.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def test_no_production_module_depends_on_a_raw_vendor_file():
    """The whole point of keeping `data/vendor/` out of git is that production cannot come
    to depend on it. This is the same guard `test_announcement_timing` puts on
    `audit/provider_timestamps.parquet`."""
    needles = ["vendor/", "research.massive", "research/massive", "massive.com",
               "benzinga", "MASSIVE_API_KEY"]
    # The one sanctioned mention: the `announce_ts_source` value the seed wrote into the
    # earnings table. Comparing a stored provenance label reads the DB, not a vendor file
    # (the duplicate-date cleanup prefers Benzinga-timed rows this way).
    provenance_label = "'massive_benzinga:'"  # ast.unparse quotes strings with '
    for f in _python_files(*(REPO / d for d in PRODUCTION_DIRS if (REPO / d).is_dir())):
        source = _code_only(f).lower().replace(provenance_label, "")
        for needle in needles:
            assert needle.lower() not in source, f"{f} references {needle!r}"


def test_the_vendor_directory_is_gitignored():
    ignored = (REPO / ".gitignore").read_text()
    assert "data/vendor/" in ignored
