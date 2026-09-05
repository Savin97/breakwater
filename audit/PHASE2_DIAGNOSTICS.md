# Phase 2 diagnostics — the corrected outcome dataset

Frozen output of `PYTHONPATH=. .venv/bin/python -m audit.phase2_diagnostics`, run on the
Phase 2 implementation commit against `output/events_df.parquet`.

**This describes a dataset, not a model.** No tier hit rate, lift or capture number appears
here on purpose: every tier in the frame was produced by a scorer fitted on the mismeasured
target, so reading performance off it would repeat exactly the mistake
`audit/PHASE0_AUDIT_REV2.md` §Q3 documents. The rebuild and re-fit are Phase 3.

Every figure below is computed on events with an **independently observed announcement
timestamp**. Nothing is inferred from price behavior.

## Cross-check against the audit

The audit's verified-timestamp analysis (§Q2, `audit/verified_timing_analysis.py`, a
standalone script that never touched the pipeline) reported the market baseline
P(|reaction_3d| ≥ 8%) moving 0.128 → 0.204 and the BMO baseline 0.041 → 0.173. The
production implementation reproduces both to three decimals on 11,417 resolved events,
which is the intended agreement: the pipeline now computes what the audit script proved
was needed.

```

==============================================================================
1. ANNOUNCEMENT WINDOWS — observed timestamps only, never price behavior
==============================================================================
  BMO           6575   14.4%
  AMC           4842   10.6%
  INTRADAY        98    0.2%
  UNKNOWN      34186   74.8%
  TOTAL        45701

  timestamped events: 11515 (25.2%)   range 2008-04-16 -> 2026-09-03
  provenance: audit_provider_timestamps_2026_09_05=11515

==============================================================================
2. RESOLVED COVERAGE BY YEAR
==============================================================================
      events  resolved   BMO  AMC  INTRADAY coverage
year                                                
2000    1038         0     0    0         0     0.0%
2001    1320         0     0    0         0     0.0%
2002    1414         0     0    0         0     0.0%
2003    1448         0     0    0         0     0.0%
2004    1469         0     0    0         0     0.0%
2005    1509         0     0    0         0     0.0%
2006    1541         0     0    0         0     0.0%
2007    1576         0     0    0         0     0.0%
2008    1614         3     3    0         0     0.2%
2009    1622         4     4    0         0     0.2%
2010    1650         0     0    0         0     0.0%
2011    1690         0     0    0         0     0.0%
2012    1693         0     0    0         0     0.0%
2013    1756         0     0    0         0     0.0%
2014    1789         3     3    0         0     0.2%
2015    1813         4     4    0         0     0.2%
2016    1834         4     4    0         0     0.2%
2017    1859         4     4    0         0     0.2%
2018    1871         4     4    0         0     0.2%
2019    1910         8     4    4         0     0.4%
2020    1926       493   282  211         2    25.6%
2021    1961      1881  1086  795        23    95.9%
2022    1969      1896  1101  795        20    96.3%
2023    1973      1902  1091  811        22    96.4%
2024    1997      1929  1103  826        20    96.6%
2025    2004      1896  1088  808         6    94.6%
2026    1455      1386   794  592         5    95.3%

  Coverage is thin before ~2020 and zero early on. That is the binding constraint
  on Phase 3: a walk-forward re-fit cannot claim a window the timestamps do not cover.

==============================================================================
3. LEGACY vs ANCHORED  P(|reaction_3d| >= 8%)  — resolved events only
==============================================================================
  n = 11417 resolved events   2008-04-16 -> 2026-09-03

  slice             n      legacy    anchored     delta
  ALL           11411      0.1281      0.2043   +0.0762
  BMO            6573      0.0411      0.1733   +0.1322
  AMC            4838      0.2464      0.2464   +0.0000

  Read this as a measurement correction, not a result. The legacy BMO rate is
  low because the legacy window starts one session AFTER the news.

==============================================================================
4. BMO DISTRIBUTION — legacy vs anchored |reaction_3d|
==============================================================================
     legacy  anchored   ratio
p10  0.0038    0.0063  1.6763
p25  0.0096    0.0169  1.7725
p50  0.0210    0.0364  1.7364
p75  0.0377    0.0658  1.7437
p90  0.0594    0.1022  1.7210
p95  0.0751    0.1353  1.8007
p99  0.1163    0.2100  1.8061

  mean   legacy 0.0276   anchored 0.0482   (1.75x)
  n = 6573 BMO events

==============================================================================
5. AMC EQUALITY CHECK — anchored must be BIT-IDENTICAL to legacy
==============================================================================
  reaction_1d : IDENTICAL   n=4842
  reaction_3d : IDENTICAL   n=4842
  reaction_5d : IDENTICAL   n=4842
  abs_reaction_3d: IDENTICAL

  PASS — AMC was never mismeasured, so the correction must be a no-op there.
  This is the control that proves the anchoring code is not moving things on its own.

==============================================================================
6. UNRESOLVED EVENTS — counts and reasons
==============================================================================
    34186  unresolved_no_timestamp        no independently observed announcement timestamp for this event
       98  unresolved_intraday            announced mid-session; no unambiguous pre-announcement close
    34284  TOTAL UNRESOLVED (75.0% of completed events)

  resolved but with an incomplete forward window: 4 (the 3 sessions after the anchor have not all closed yet)
  None of these carry an anchored target, and resolved_events() is the only gate
  into corrected calibration — invariant 7.

==============================================================================
7. MISSING-PRICE / NON-SESSION CASES — counted, never rolled
==============================================================================
  Events in the `earnings` table with no row in the event frame:
       972  outside this ticker's own price history
       496  outside the loaded price window (future date, or pre-history)
       299  unresolved_no_session
        24  unresolved_price_gap
      1791  TOTAL

  The 24 price-gap cases are an INGESTION bug, not a calendar problem —
  the market traded that day and we have no row. Rolling them forward would hide it.
  HD@2026-05-19, WMT@2026-05-21, DE@2026-05-21, TGT@2026-05-20, ADI@2026-05-20, CPRT@2026-05-21, LOW@2026-05-20, WSM@2026-05-21, JBL@2026-06-16, GIS@2026-06-24, SPGI@2008-01-24, DECK@2026-05-21, TTWO@2026-05-21, WDAY@2026-05-21, HAS@2026-05-20, KEYS@2026-05-19, FDS@2026-06-25, NDSN@2026-05-20, ROST@2026-05-21, NKE@2026-06-25, TJX@2026-05-20, SPGI@2006-04-25, INTU@2026-05-20, RL@2026-05-21

  In-frame report-date session check (anchor_session_status):
     45701  ok

==============================================================================
END — dataset properties only. No model claim is made or implied.
==============================================================================
```
