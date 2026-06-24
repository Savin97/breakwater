---
name: reddit-marketing-playbook
description: Reddit/X comment strategy for building Breakwater karma — tone, angles, data pulls, and real examples from Jun 23 2026 session
metadata:
  type: project
---

# Breakwater Reddit/X Marketing Playbook

## Core approach
Comment with real data from the model. Never hype, never trash a name. Position as someone running actual numbers. Let the data create engagement — don't force the Breakwater mention, let curious people ask.

## Tone rules
- Factual, not preachy
- Respect the thesis even when pushing back
- Correct narrative with data, not opinion
- "I ran it through a risk model I use" is the soft plug — natural opener for DMs
- No directional calls — the model flags tail risk, not direction

## Data pulls to include in comments
For any stock: bucket, peer percentile, recent 3d reactions, beat rate, beat-but-fell rate, expected move (IV), iv_vs_hist_ratio, pre_earnings_drift_flag

## Angles that land well

### "Beat but stock fell" angle (NKE, MU)
Pull beat rate + % of beats where stock still fell. Counterintuitive, gets engagement.
> NKE: 73% beat rate, stock fell in 45% of those beats. Avg 3d return after a beat: +0.6%

### "Options underpricing" angle (FDX)
iv_vs_hist_ratio < 1.0 = market may be underestimating the move.
> FDX iv_vs_hist_ratio = 0.72 — options pricing below historical vol

### "Options overpricing" angle (MU)
iv_vs_hist_ratio > 1.5 = fear is priced in.
> MU iv_vs_hist_ratio = 1.63, expected move 14.8%

### "Compressed drift" angle (FDX)
pre_earnings_drift_flag = Compressed → unusually quiet pre-earnings, historically precedes larger moves.

### "Coiled spring" angle
Combine: calm recent quarters + Compressed drift flag + High Alert bucket = setup the model flags as high risk despite apparent calm.

### World Cup / macro catalyst pushback
Pull actual price data from the event window. 2018 WC: NKE +5.7%. 2022 WC pop was a broad market rally, not Nike-specific.

### Margin debt warning (NKE post)
When someone mentions margin/leverage into a volatile name — flag the tail risk explicitly using the bucket + recent reaction history.

## Real comment examples (Jun 23 2026)

### MU "life savings" post
Led with: 43% of events move ≥8%, median 3d = 6.3%, tail includes +21.7% and -13.6%. Model flags tail risk not direction. Waiting until after earnings preserves capital either way.

### FDX diversification post
Led with: FDX reports tonight, High Alert 90th percentile, Compressed drift, options underpricing at 0.72x hist vol. Suggested waiting for the print to settle.

### MU analyst commentary post
Led with: MU beats 64% of the time, avg 3d return after beat = -0.1%. Beat by 33% in Mar 2026, fell 12.4%. The guide vs expectations is what moves it, not the thesis.

### NKE YOLO post
Led with: beat 7/8 recent quarters, fell after 5 of them including -16.6% and -12.6%. World Cup data: 2018 = +5.7%, 2022 rally was market-wide. 20k on margin into a downtrend = risk even if eventually right.

### MU X post format
Short, data-only, no directional call:
> $MU reports tomorrow. Last 8 quarters: [list moves]. 43% of events move ≥8%. Options pricing 14.8% move, IV 63% above historical. Extended beat streak going in. No edge on direction. Pure volatility event.

## Stocks not in universe (noted Jun 23 2026)
SOFI, ELF, CELH, RVLV — smaller caps outside coverage
