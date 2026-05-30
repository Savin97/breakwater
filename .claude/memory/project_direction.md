---
name: project-direction
description: Breakwater product direction, target market, and business strategy as of 2026-05-30
metadata:
  type: project
---

Breakwater is being built as a paid product for retail/prosumer options traders who trade around earnings.

**Target market:** Retail and prosumer options traders (straddle/strangle buyers, hedgers). NOT institutional quants — they build their own and need governance, SHAP, sector-specific models. The accessible market is people who want a systematic edge in identifying dangerous earnings events.

**Value proposition:** The signal predicts earnings move *magnitude* (not direction), which is directly useful for options traders. High Alert stocks move >8% on earnings ~39% of the time vs. 11.5% base rate (4.5x lift, sustained 15 years OOS). When options-implied move < historical 75th percentile reaction for a High Alert stock, options appear cheap relative to history — a genuine edge.

**Price target:** $50–200/month per user for retail tier.

**Infrastructure already live:**
- harbor-markets.com — landing page (separate repo, same droplet)
- harbor-markets.com/breakwater — live Streamlit dashboard (this repo)
- Daily ingestion + IV collection running on droplet via cron

**Revenue path:** Add payment gate (Stripe) to dashboard. Email digest as push delivery. Landing page copy should lead with the 4.5x lift / 15-year OOS story.

**Why:** Model quality is proven. The gaps are product completeness (IV not shown, manual coverage) and delivery (no email push, no payment gate).
