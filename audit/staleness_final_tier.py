"""Q4 — staleness impact on the FINAL SHIPPED tier.

Recomputes every history-dependent piece of state through the most recently COMPLETED
earnings event, replicating scoring/scoring_features.py exactly, and compares against
what groupby().last() actually shipped. Target definition held fixed (Issue 1 not
applied here) so the two defects stay separable.
"""
import numpy as np, pandas as pd
from config import (BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR,
                    LIFT_PRIOR_STRENGTH, LIFT_TO_ELEVATED, LIFT_TO_HIGH_ALERT,
                    EXTREME_EARNINGS_REACTION_THRESHOLD as X)

COLS = ["stock","date","earnings_date","is_earnings_day","abs_reaction_3d",
        "reaction_3d","reaction_1d","abs_reaction_p75","abs_reaction_p75_rolling",
        "reaction_entropy","earnings_explosiveness_score","earnings_explosiveness_bucket",
        "earnings_explosiveness_bucket_structural","stock_bucket_lift",
        "is_extreme_reaction","pre_earnings_drift_flag"]
df = pd.read_parquet("output/full_df.parquet", columns=COLS)
ev = df[df["is_earnings_day"]==1].sort_values(["stock","date"]).copy()
ev["best"] = ev["reaction_3d"].fillna(ev["reaction_1d"])
ev = ev[ev["abs_reaction_3d"].notna()]

# what actually shipped
up = pd.read_parquet("output/upcoming_df.parquet").set_index("stock")

def entropy(s, bins=8):
    s = pd.Series(s).dropna()
    if len(s) < bins: return np.nan
    h,_ = np.histogram(s, bins); p = h/h.sum(); p = p[p>0]
    return float(-np.sum(p*np.log(p)))

global_prior = ev["is_extreme_reaction"].mean()     # market baseline as of now

rows = []
for stock, g in ev.groupby("stock", sort=False):
    if stock not in up.index: continue
    g = g.sort_values("date")
    a = g["abs_reaction_3d"]
    p75_roll = a.iloc[-28:].quantile(0.75) if len(a) >= 28 else np.nan
    p75_exp  = a.quantile(0.75)
    p75      = p75_roll if pd.notna(p75_roll) else p75_exp
    ent      = entropy(g["best"].abs())
    e3 = np.clip(p75/0.12, 0, 1) if pd.notna(p75) else np.nan
    e4 = np.clip(ent, 0, 1) if pd.notna(ent) else 0.0
    score = 100*np.clip(0.85*e3 + 0.15*e4, 0, 1) if pd.notna(e3) else np.nan
    struct = ("High Alert" if score > BUCKET_HIGH_ALERT_FLOOR else
              "Elevated"   if score > BUCKET_ELEVATED_FLOOR else "Normal") if pd.notna(score) else None
    # lift: this stock's prior record inside the SAME structural bucket
    prior = g[g["earnings_explosiveness_bucket_structural"].astype(str) == str(struct)]
    n_prior, sum_prior = len(prior), prior["is_extreme_reaction"].sum()
    shrunk = (sum_prior + LIFT_PRIOR_STRENGTH*global_prior)/(n_prior + LIFT_PRIOR_STRENGTH)
    lift = shrunk/global_prior
    final = struct
    if struct in ("Normal","Elevated") and lift >= LIFT_TO_HIGH_ALERT:   final = "High Alert"
    elif struct == "Normal" and lift >= LIFT_TO_ELEVATED:                final = "Elevated"
    flag = str(up.at[stock,"pre_earnings_drift_flag"] or "").strip()
    rows.append({"stock":stock,
        "p75_new":p75, "ent_new":ent, "score_new":score, "lift_new":lift,
        "struct_new":struct, "final_new":final, "hc_new": final=="High Alert" and flag!="",
        "p75_old":g["abs_reaction_p75_rolling"].iloc[-1] if pd.notna(g["abs_reaction_p75_rolling"].iloc[-1])
                  else g["abs_reaction_p75"].iloc[-1],
        "ent_old":g["reaction_entropy"].iloc[-1],
        "score_old":g["earnings_explosiveness_score"].iloc[-1],
        "lift_old":g["stock_bucket_lift"].iloc[-1],
        "struct_old":str(g["earnings_explosiveness_bucket_structural"].iloc[-1]),
        "final_old":str(up.at[stock,"earnings_explosiveness_bucket"]),
        "hc_old":bool(up.at[stock,"is_high_conviction"]),
        "last_reaction":a.iloc[-1], "n_events":len(a),
        "earnings_date":up.at[stock,"earnings_date"]})
r = pd.DataFrame(rows).set_index("stock").dropna(subset=["score_new"])
r.to_parquet("audit/staleness_final_tier.parquet")

print(f"upcoming events compared: {len(r)}\n")
print(f"{'quantity':<22}{'changes':>10}{'mean |delta|':>15}{'max |delta|':>14}")
for lbl, o, n, f in [("abs_reaction_p75","p75_old","p75_new","{:.4f}"),
                     ("reaction_entropy","ent_old","ent_new","{:.4f}"),
                     ("explosiveness_score","score_old","score_new","{:.2f}"),
                     ("stock_bucket_lift","lift_old","lift_new","{:.3f}")]:
    d = (r[n]-r[o]).abs()
    print(f"{lbl:<22}{(d>1e-9).sum():>6} ({(d>1e-9).mean():>4.0%}){f.format(d.mean()):>15}{f.format(d.max()):>14}")
for lbl, o, n in [("structural tier","struct_old","struct_new"),
                  ("FINAL shipped tier","final_old","final_new"),
                  ("is_high_conviction","hc_old","hc_new")]:
    ch = (r[o].astype(str) != r[n].astype(str))
    print(f"{lbl:<22}{ch.sum():>6} ({ch.mean():>4.0%})")

print("\n--- FINAL shipped tier: what ships now (rows) vs what is correct (cols) ---")
print(pd.crosstab(r["final_old"], r["final_new"]).to_string())
print("\n--- high conviction ---")
print(pd.crosstab(r["hc_old"], r["hc_new"]).to_string())

moved = r[r["final_old"].astype(str) != r["final_new"].astype(str)].copy()
moved["dir"] = np.where(moved["score_new"] > moved["score_old"], "UNDER-called", "OVER-called")
print(f"\n--- all {len(moved)} events whose FINAL tier is wrong today ---")
print(moved.sort_values("earnings_date")[["earnings_date","score_old","score_new",
      "lift_old","lift_new","final_old","final_new","last_reaction","dir"]].round(3).to_string())
