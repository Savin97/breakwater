# PAUSED — 2026-10-02

The options work is on hold until professional, point-in-time historical options data is
affordable or available. Nothing here has been run on such data; nothing was bought.

Where it stands:

- `research/options_pilot/` — done on free DoltHub data. Verdict **WEAK / UNCERTAIN VALUE**
  (`research/options_pilot/RESULTS.md`, bar for buying data in §9).
- `research/options_confirmatory/` (this folder) — the one confirmatory test, C vs C +
  implied earnings move, designed and pre-registered (`PREREGISTRATION.md`,
  `DATA_SOURCES.md`, `core.py`, `testing/test_options_confirmatory.py`) but **not run**.

To pick it up again:

1. Get point-in-time historical options quotes (ORATS was the lead; check that the
   vendor's snapshots are as-of the quote date, not back-filled, before paying).
2. Write the vendor adapter and the population/coverage audit listed as "still to write"
   in `.claude/memory/MEMORY.md` (2026-10-02 entry).
3. Run the test exactly as `PREREGISTRATION.md` says. Do not reuse the pilot's DoltHub
   sample as evidence.
