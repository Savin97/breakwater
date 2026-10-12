# Breakwater

Earnings-move risk for S&P 500 stocks. Before each earnings report, Breakwater ranks how
large the stock's move is likely to be **in size, not direction**, using only information
available before the announcement.

> **No performance figures are published at present.** The production model (Model B,
> version 0.4.0) went live on 2026-10-04 and has no prospective record yet; the public track
> record stays paused until enough archived 0.4.0 calls have resolved. The previously
> published figures (40% vs 6.9%, 5.8x lift, a consistent 2015–2025 OOS record) were derived
> from a target that mismeasured before-open announcements, were overstated by roughly 3x,
> and are retracted. The history is recorded in `audit/PHASE0_AUDIT_REV2.md`.

Live at **harbor-markets.com/breakwater** (Streamlit dashboard). Weekly email digest sent Monday mornings.

---

## The model (Model B, 0.4.0)

* **Signal.** For each event: the mean absolute **corrected 3-session earnings reaction**
  over the stock's prior reports. "Corrected" means measured from the last close before
  the announcement, using a verified announcement time (before-open reports anchor on the
  previous close). Prior reports without a verified time are skipped, never counted as zero.
* **Causal cutoff.** Only prior outcomes complete by the call cutoff count — the last
  market session before the Monday of the report week. A rerun later in the week keeps it.
* **Minimum history.** At least 8 usable prior reactions; otherwise the event is
  **Unscored** (not "low risk").
* **Score.** `risk_score` is the **Breakwater Risk Percentile**: where the signal falls in a
  frozen reference distribution (2014–2025, `scoring/model_b_reference.json`). It is a rank,
  not a probability.
* **Tiers.** High Alert ≥ 90 (top decile of the reference), Elevated 80–90, Normal < 80,
  Unscored. Fixed bands, not fitted on outcomes. Nothing else changes a tier.
* **Validation target.** `abs_reaction_3d_anchored >= 0.08`. Selection and validation:
  `analysis/model_c_refit/` (research; production does not import it).
* **Options data** (implied move, ATM IV) is shown as market context only. It does not
  enter the score or the tier.

Implementation: `scoring/model_b.py`.

---

## Quick Commands

```bash
source .venv/bin/activate

python main.py                                   # Full pipeline (ingest + score + outputs)
python -c "from pipeline.pipeline import run_pipeline; run_pipeline(ingest=False)"   # re-score, no API calls
streamlit run streamlit_dash/app.py              # Dashboard locally
python -m analysis.predictions_range --weeks-back 4   # Model B calls + corrected outcomes, by week
python -m analysis.last_week_results             # Archived 0.4.0 calls vs corrected outcomes (internal)
pytest testing                                   # Test suite
```

Validation runs must never write the canonical prediction archive: set
`BREAKWATER_PREDICTIONS_DB=<scratch copy>` first.

## Monday Workflow

```bash
bash scripts/full_workflow.sh
```

1. Pull `breakwater.duckdb` from the droplet (rsync)
2. Run the full pipeline — scoring and local outputs only (reports, calendar, parquets, chart)
3. Push the dashboard parquets to the droplet — the calls are now public
4. Archive exactly those calls: `python -m publishing.archive_calls --published`
5. Push `recent_calls.json` to the harbor webpage — skipped while the track record is paused
6. Send the weekly digest

Steps 3–6 publish, and run **only from `master`**: on any other branch the script stops after
step 2 with `REFUSING TO DEPLOY — current branch: …` (exit 3). See `scripts/deploy_guard.sh`.
Scoring never writes the prediction archive; only step 4 does.

---

## Pipeline Architecture

`main.py` → `pipeline/pipeline.py`:

| Step | File | What it does |
|---|---|---|
| 1 | `pipeline/stage1.py` | Ingestion into DuckDB (`incremental=True` → yfinance). |
| 2 | `pipeline/stage2.py` | DB → daily frame (prices, earnings dates, sectors, IV and EPS snapshots). |
| 3 | `pipeline/stage3.py` | Daily preparation: earnings-window clock, daily returns/volatility, legacy reactions (anchoring control only). |
| — | `pipeline/events.py` | Event frame: one row per earnings event + one pending row per active stock, with verified timing and corrected anchored outcomes. |
| 4 | `pipeline/stage4.py` | Model B scoring on the event frame (`scoring/model_b.py`). |
| 5 | `pipeline/stage5.py` | Outputs: parquets, PDF reports, weekly calendar, dashboard export, weekly chart, last-week results, prediction snapshot. |

---

## Codebase Map

```
breakwater/
├── main.py / config.py
├── pipeline/            stage1–5, events.py (event frame)
├── scoring/             model_b.py, model_b_reference.json, build_model_b_reference.py
├── feature_engineering/ announcement_timing.py (BMO/AMC, anchored reactions), daily features
├── ingestion/           prices, earnings dates, IV, EPS estimates, sectors (yfinance)
├── report/              PDF report, reaction chart, weekly calendar, shared customer wording
├── streamlit_dash/      dashboard app + parquet export
├── analysis/            weekly chart, prediction snapshot, last-week results, predictions_range,
│                        model_c_refit/ + model_b_validation/ (research evidence, not production)
├── marketing/           public track record generator (paused) and positioning docs
├── cron/                droplet ingestion/IV/EPS jobs, weekly digest
├── backfills/           one-time loaders (already applied)
├── testing/             pytest suite (`pytest testing`)
└── db/predictions.duckdb   archive of every published call (tracked in git)
```

---

## Data Storage

**DuckDB** (`db/breakwater.duckdb`, gitignored, synced from the droplet): `prices`,
`earnings` (incl. verified announcement timestamps), `stock_data`, `iv_snapshots`,
`eps_estimates`.

**Prediction archive** (`db/predictions.duckdb`): one row per call per run day. Pre-0.4 rows
(legacy model) are kept as history and never count toward a track record; the
`predictions_track_record` view holds only first 0.4.0 calls.

**Parquet** (`output/`): `full_df.parquet` (daily frame), `events_df.parquet` (event frame
with Model B fields), `streamlit_df.parquet` / `upcoming_df.parquet` (dashboard).

---

## Infrastructure

**DigitalOcean droplet** (`harbor-markets.com`):
- Repo: `/var/www/breakwater` — deploy by pushing locally + `git pull` on droplet
- Crons run ingestion, IV and EPS snapshots (see `cron/`); scoring runs locally
- Dashboard served at `harbor-markets.com/breakwater` (systemd + Nginx)

**Website** (`harbor-markets.com`): separate repo `Savin97/harbor_webpage`, server at `/var/www/harbor_webpage`.

---

## Model Performance

None published. Model B's research validation (`analysis/model_c_refit/RESULTS.md`) is a
historical backtest, not live performance, and is not presented as a track record. The
audit measurement of the legacy tiers is historical record in `audit/PHASE0_AUDIT_REV2.md`.
