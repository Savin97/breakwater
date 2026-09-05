"""How much of the model's apparent lift is WITHIN-window discrimination vs
between-window composition (i.e. 'this stock reports AMC')? Verified events only."""
import numpy as np, pandas as pd
from config import EXTREME_EARNINGS_REACTION_THRESHOLD as X

m = pd.read_parquet("audit/verified_scored_events.parquet")
base = m.groupby("window")["a_anch"].apply(lambda s: (s >= X).mean())
overall = (m["a_anch"] >= X).mean()

print(f"anchored baseline by verified window:  " +
      "  ".join(f"{w}={base[w]:.3f} (n={(m['window']==w).sum()})" for w in base.index))
print(f"pooled anchored baseline: {overall:.3f}\n")
print(f"{'tier':<12}{'n':>6}{'obs':>8}{'crude lift':>12}{'expected*':>11}{'stratified lift':>17}")
for t in ["High Alert", "Elevated", "Normal", "HighConv"]:
    s = m[m["hc"]] if t == "HighConv" else m[m["tier"] == t]
    obs = (s["a_anch"] >= X).mean()
    exp = s["window"].map(base).mean()          # rate expected from window mix alone
    print(f"{t:<12}{len(s):>6}{obs:>8.3f}{obs/overall:>11.2f}x{exp:>11.3f}{obs/exp:>16.2f}x")
print("\n* expected = the rate this tier's events would show from their window mix alone,")
print("  with zero within-window skill. crude/stratified gap = composition effect.")

print("\nAMC vs BMO as a standalone 'signal' (no model at all):")
print(f"  anchored P(>=8%) AMC {base['AMC']:.3f} vs BMO {base['BMO']:.3f}"
      f"  ->  {base['AMC']/base['BMO']:.2f}x  from knowing only the reporting window")
