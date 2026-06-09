---
name: social-media-strategy
description: Social media strategy for Harbor Markets / Breakwater — platforms, cadence, content rules, and weekly workflow
metadata:
  type: project
---

## Platforms

**X (Twitter):** Active account, primary posting platform. Post 3x/week minimum.

**Reddit:** Account exists but has ~4 karma as of Jun 9, 2026. Strategy is active thread participation — find and comment in ANY thread discussing a stock reporting that week. Do NOT limit to options subreddits. Post in every relevant community.

## Reddit — subreddits to monitor every week

- r/stocks — weekly earnings megathreads + individual ticker posts
- r/wallstreetbets — huge, earnings plays constantly discussed
- r/investing — slower but serious, earnings risk matters
- r/StockMarket — similar to r/stocks
- r/thetagang — relevant audience even if you don't trade options yourself
- r/options — same
- r/SecurityAnalysis — smaller, high quality, earnings surprises discussed
- r/Daytrading — earnings plays are core content

## Reddit — thread types to find and comment in

1. "[TICKER] earnings this week / tomorrow" — pre-earnings discussion
2. Weekly "what are you watching" / earnings calendar megathreads (r/stocks does these every Monday)
3. Post-earnings "[TICKER] just reported / crushed / missed" threads — drop outcome commentary

Search: `site:reddit.com ORCL earnings`, or use Reddit search for the ticker in relevant subs.

## Posting cadence (X)

| Day | Post type |
|---|---|
| Mon/Tue | Weekly earnings watch — chart image + top stocks |
| Wed/Thu | Outcome commentary after events report |
| Fri (optional) | Evergreen earnings observation |

Bio always links harbor-markets.com. Occasional post footer CTA — don't push it every time.

## Tone — IMPORTANT

You track and surface earnings risk. You are NOT an options trader, NOT giving trading advice. Frame as: "I track earnings tail risk across the S&P 500 — here's what's on the radar this week."

## Weekly workflow

1. Run `python report/chart_weekly.py` → saves `output/weekly_chart.png`
2. Pull key stats for top stocks from `full_df.parquet` (avg move, >5%/>8% frequency)
3. Post on X with image
4. Search Reddit for active threads on this week's stocks, drop comments

## Content rules (STRICT)

- **NO** model lift numbers, calibration stats, methodology, or how the model works
- **NO** mention of ML / algorithms / model
- **NO** framing as options advice or trading advice
- **YES** factual historical data (avg move %, frequency counts)
- **YES** "I track earnings tail risk" framing — analyst/tracker perspective, not trader
- **ALWAYS** link harbor-markets.com (never the raw dashboard URL)

## Post templates

### Weekly watch (X, with image)
```
Earnings risk this week — [day/day] are the ones to watch.

★ $TICKER (High Alert) — avg ~X% move, Y of 8 quarters moved >5%. [1-line context].

$TICKER (Elevated) — avg ~X% move. [1-line context].

Full weekly tracker → harbor-markets.com
```

### Outcome post (X, after earnings)
```
$TICKER moved [X]% on earnings.

Had it flagged as [High Alert / Elevated] this week. [Brief observation vs. historical pattern].

harbor-markets.com
```

### Reddit comment (pre-earnings thread)
```
$TICKER has averaged about X% over its last 8 quarters — [1-line historical context].

I track earnings tail risk across the S&P 500 weekly: harbor-markets.com
```

### Reddit comment (post-earnings thread)
```
Had $TICKER flagged this week — [avg move / historical context]. [Brief observation about the actual result vs. history].

Track it weekly at harbor-markets.com
```

## What's working / notes

- First weekly post: week of Jun 9, 2026 — ORCL (High Alert, ★) + ADBE (Elevated) + SJM (Normal)
- ORCL avg ~12% last 8 quarters, 6/8 moved >5%. Extended momentum, erratic surprise history.
- ADBE avg ~8% last 8 quarters, 6/8 moved >5%. Extended beat streak — bar elevated.
- `chart_weekly.py` at `report/chart_weekly.py` — scatter chart by risk tier + peer percentile, Mon-Fri window
