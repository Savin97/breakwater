"""A thin, paginating client for the Massive Benzinga REST API.

Scope: read-only HTTP. It knows how to authenticate, how to retry, and how to walk
`next_url` until the vendor stops handing out cursors. It knows nothing about earnings
semantics — that is `normalize.py` — and nothing about Breakwater's database.

The API key
-----------
Read once from the `MASSIVE_API_KEY` environment variable (`.env` is loaded if present)
and held only in the `Authorization` request header. It is never placed in a URL, never
written to a snapshot, never logged and never included in an exception message: the key
lives in `_auth_header()` and nothing else in this package ever sees it. `scrub()` is
applied to every URL before it is stored or logged, so that even a future vendor change
that started echoing credentials back in `next_url` could not leak one into a snapshot.
"""
import logging
import os
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

try:  # optional: the key may already be exported
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements
    load_dotenv = None

log = logging.getLogger(__name__)

BASE_URL = "https://api.massive.com/benzinga/v1"
EARNINGS_PATH = "/earnings"

# The vendor documents `limit` max 50,000. Ask for the largest sensible page so a full
# history costs tens of requests rather than thousands.
MAX_PAGE_LIMIT = 50_000

# Query parameters that could conceivably carry a credential. Stripped from any URL
# before it is stored or logged.
_SECRET_QUERY_KEYS = frozenset({"apikey", "api_key", "apiKey", "token", "key",
                                "access_token", "authorization"})

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class MassiveAPIError(RuntimeError):
    """A vendor response we will not silently accept."""


def scrub(url: str) -> str:
    """`url` with any credential-shaped query parameter removed.

    Applied to every URL this module stores or logs. Today the key travels in a header
    and this is a no-op; it is here so that it stays a no-op if that ever changes.
    """
    parts = urlsplit(url)
    if not parts.query:
        return url
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in {s.lower() for s in _SECRET_QUERY_KEYS}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))


def _auth_header() -> dict:
    """The only place the API key is read, and the only place it is used."""
    if load_dotenv is not None:
        load_dotenv(override=False)
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        raise MassiveAPIError(
            "MASSIVE_API_KEY is not set. Export it or put it in .env; it is never "
            "passed on the command line and never written to disk.")
    return {"Authorization": f"Bearer {key}"}


class MassiveClient:
    """Paginating reader for one Massive collection.

    `session` is injectable so tests can drive the pagination logic against a fake
    transport without a network or a key.
    """

    def __init__(self, base_url: str = BASE_URL, session=None, timeout: float = 120.0,
                 max_retries: int = 6, backoff_base: float = 1.5, sleep=time.sleep,
                 auth_header=None):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._auth_header = auth_header or _auth_header
        self.request_count = 0

    # ---------------------------------------------------------------- one request
    def get(self, url: str, params: dict | None = None) -> dict:
        """One GET, retried on transport errors and retryable statuses.

        Returns the decoded payload. Raises `MassiveAPIError` rather than returning a
        partial or error body — an acquisition that swallowed a 500 would produce a
        snapshot that is silently short, which is the one failure mode a completeness
        audit cannot tolerate.
        """
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.request_count += 1
                resp = self.session.get(url, params=params, headers=self._auth_header(),
                                        timeout=self.timeout)
            except requests.RequestException as exc:  # network-level
                last = f"{type(exc).__name__}"
                log.warning("massive: %s on %s (attempt %d/%d)",
                            last, scrub(url), attempt, self.max_retries)
            else:
                if resp.status_code == 200:
                    payload = resp.json()
                    status = payload.get("status")
                    if status not in (None, "OK"):
                        raise MassiveAPIError(
                            f"vendor returned status={status!r} for {scrub(url)}")
                    return payload
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUSES:
                    raise MassiveAPIError(f"{last} for {scrub(url)}")
                log.warning("massive: %s on %s (attempt %d/%d)",
                            last, scrub(url), attempt, self.max_retries)
            if attempt < self.max_retries:
                self._sleep(self.backoff_base ** attempt)
        raise MassiveAPIError(
            f"gave up after {self.max_retries} attempts ({last}) on {scrub(url)}")

    # ---------------------------------------------------------------- pagination
    def iter_pages(self, path: str = EARNINGS_PATH, params: dict | None = None,
                   start_url: str | None = None, max_pages: int | None = None):
        """Yield `(page_number, url, payload)` following `next_url` until exhausted.

        Two guards, both learned from how cursor APIs fail:

        * a `next_url` identical to the one just fetched ends the walk instead of
          looping forever;
        * a page whose `results` is empty ends the walk even if a cursor came back,
          because an endless tail of empty pages is indistinguishable from progress.

        `start_url` resumes a partial acquisition from a stored cursor.
        """
        url = start_url or f"{self.base_url}{path}"
        query = None if start_url else dict(params or {})
        seen_urls = set()
        page = 0
        while url:
            if url in seen_urls:
                log.warning("massive: next_url repeated, stopping at page %d", page)
                return
            seen_urls.add(url)
            payload = self.get(url, params=query)
            query = None  # params only apply to the first request; the cursor carries them
            page += 1
            yield page, scrub(url), payload
            results = payload.get("results") or []
            nxt = payload.get("next_url")
            if not results or not nxt:
                return
            if max_pages is not None and page >= max_pages:
                return
            url = nxt
