import pandas as pd, numpy as np, warnings
from sklearn.metrics import roc_auc_score
from testing.testing_functions import forward_eval_onefactor
warnings.filterwarnings('ignore')

df = pd.read_parquet("output/full_df.parquet")
earn = df[df['is_earnings_day'] == 1].copy()
earn['date'] = pd.to_datetime(earn['date'])
earn['year'] = earn['date'].dt.year

prior = 5

# ── OOS Calibration: do predicted bucket probabilities match realized rates? ──
# Walk-forward: for each test year, bucket probs are computed from all prior years only.
# ECE (Expected Calibration Error) measures average probability mispricing in percentage points.
def oos_calibration():
    

    bucket_order = ['Normal', 'Elevated', 'High Alert']

    rows = []
    for y in range(2011, 2026):
        train = earn[earn['year'] < y]
        test  = earn[earn['year'] == y]
        if len(train) < 500 or len(test) < 50:
            continue

        p_global = train['is_extreme_reaction'].mean()
        bstats = (
            train.groupby('earnings_explosiveness_bucket')['is_extreme_reaction']
            .agg(extreme_count='sum', event_count='count')
        )
        bstats['predicted'] = (
            (bstats['extreme_count'] + prior * p_global)
            / (bstats['event_count'] + prior)
        )
        for bucket in bucket_order:
            if bucket not in bstats.index:
                continue
            subset = test[test['earnings_explosiveness_bucket'] == bucket]
            if subset.empty:
                continue
            rows.append({
                'year':      y,
                'bucket':    bucket,
                'predicted': bstats.loc[bucket, 'predicted'],
                'actual':    subset['is_extreme_reaction'].mean(),
                'n':         len(subset),
            })

    cal = pd.DataFrame(rows)
    agg = (
        cal.groupby('bucket')
        .apply(lambda g: pd.Series({
            'avg_predicted_pct': g['predicted'].mean() * 100,
            'avg_actual_pct':    g['actual'].mean()    * 100,
            'error_pp':          (g['predicted'] - g['actual']).abs().mean() * 100,
            'total_n':           g['n'].sum(),
        }), include_groups=False)
        .loc[bucket_order]
    )

    ece = (agg['error_pp'] * agg['total_n'] / agg['total_n'].sum()).sum()

    print('OOS Calibration — walk-forward 2011-2025 (prior=20):')
    print(agg.round(2).to_string())
    print(f'\nECE: {ece:.2f}pp   |   Global baseline: {earn["is_extreme_reaction"].mean()*100:.1f}%')

def threshold_grid_search():
    # ── Threshold grid search ──
    # Walk-forward calibration across a grid of (low_thr, high_thr) pairs.
    # For each pair: recompute buckets on-the-fly, compute OOS bucket probs, measure ECE.
    print('\nThreshold grid search (ECE in pp):')
    print(f"{'low_thr':>8} {'high_thr':>9} {'ECE':>7}  Normal_pred  Normal_act  Elev_pred  Elev_act  HA_pred  HA_act")

    best_ece, best_thrs = 999, (78, 87)
    for low_thr in range(55, 82, 3):
        for high_thr in range(low_thr + 6, 95, 3):
            grid_rows = []
            for y in range(2011, 2026):
                train = earn[earn['year'] < y].copy()
                test  = earn[earn['year'] == y].copy()
                if len(train) < 500 or len(test) < 50:
                    continue
                for df_ in (train, test):
                    df_['_bucket'] = pd.cut(df_['earnings_explosiveness_score'],
                                            bins=[-np.inf, low_thr, high_thr, np.inf],
                                            labels=['Normal','Elevated','High Alert'])
                p_global = train['is_extreme_reaction'].mean()
                bstats = train.groupby('_bucket')['is_extreme_reaction'].agg(ec='sum', n='count')
                bstats['pred'] = (bstats['ec'] + prior * p_global) / (bstats['n'] + prior)
                for b in ['Normal','Elevated','High Alert']:
                    if b not in bstats.index: continue
                    sub = test[test['_bucket'] == b]
                    if sub.empty: continue
                    grid_rows.append({'bucket': b, 'predicted': bstats.loc[b,'pred'],
                                    'actual': sub['is_extreme_reaction'].mean(), 'n': len(sub)})
            if not grid_rows: continue
            gc = pd.DataFrame(grid_rows)
            ga = gc.groupby('bucket').apply(lambda g: pd.Series({
                'pred': g['predicted'].mean(), 'act': g['actual'].mean(),
                'err': (g['predicted']-g['actual']).abs().mean(), 'n': g['n'].sum()
            }), include_groups=False)
            ece_g = (ga['err'] * ga['n'] / ga['n'].sum()).sum() * 100
            if ece_g < best_ece:
                best_ece, best_thrs = ece_g, (low_thr, high_thr)
            print(f"{low_thr:>8} {high_thr:>9} {ece_g:>7.2f}  "
                f"{ga.loc['Normal','pred']*100:>10.1f}  {ga.loc['Normal','act']*100:>9.1f}  "
                f"{ga.loc['Elevated','pred']*100:>9.1f}  {ga.loc['Elevated','act']*100:>8.1f}  "
                f"{ga.loc['High Alert','pred']*100:>7.1f}  {ga.loc['High Alert','act']*100:>6.1f}")

    print(f'\nBest: thresholds={best_thrs}, ECE={best_ece:.2f}pp')

oos_calibration()
exit()

# ── Correlation decomposition: is 0.456 real or inflated by between-stock persistence? ──
# The pooled correlation mixes two distinct effects:
#   BETWEEN-stock: explosive stocks (TSLA, NVDA) always score high AND always react big
#   WITHIN-stock:  does THIS quarter look more dangerous than usual for THIS stock?
# The between-stock component is real signal but trivially high (p75 rolling IS past reactions).
# The within-stock component is what actually tells you something new each quarter.
print('\n\n========================================')
print('CORRELATION DECOMPOSITION CHECK')
print('========================================')

oos_corr = earn[earn['year'].between(2011, 2025)].dropna(
    subset=['earnings_explosiveness_score', 'abs_reaction_3d']
).copy()

# Pooled correlation (what yearly_oos_report shows)
pooled = oos_corr['earnings_explosiveness_score'].corr(oos_corr['abs_reaction_3d'])
print(f'\nPooled OOS corr (2011-2025):        {pooled:.3f}  -- mixes between- and within-stock')

# Between-stock: correlate each stock's mean score vs mean reaction
stock_means = oos_corr.groupby('stock')[['earnings_explosiveness_score', 'abs_reaction_3d']].mean()
between = stock_means['earnings_explosiveness_score'].corr(stock_means['abs_reaction_3d'])
print(f'Between-stock corr (mean vs mean):  {between:.3f}  -- how much is "volatile stocks stay volatile"')

# Within-stock: demean each stock, correlate residuals
oos_corr['score_dm']  = oos_corr['earnings_explosiveness_score'] - oos_corr.groupby('stock')['earnings_explosiveness_score'].transform('mean')
oos_corr['react_dm']  = oos_corr['abs_reaction_3d']              - oos_corr.groupby('stock')['abs_reaction_3d'].transform('mean')
within = oos_corr['score_dm'].corr(oos_corr['react_dm'])
print(f'Within-stock corr (demeaned):       {within:.3f}  -- timing signal, same stock different quarters')

# Per-year cross-sectional rank corr (Spearman, removes outlier distortion)
cs_corrs = []
for y in range(2011, 2026):
    yr = oos_corr[oos_corr['year'] == y]
    if len(yr) < 50: continue
    cs_corrs.append(yr['earnings_explosiveness_score'].rank().corr(yr['abs_reaction_3d'].rank()))
print(f'Avg cross-sectional rank corr/year: {np.mean(cs_corrs):.3f}  -- rank ordering within each year')

# Shuffle test: shuffle reactions WITHIN each stock, preserving cross-sectional structure
np.random.seed(42)
oos_corr['react_shuffled'] = oos_corr.groupby('stock')['abs_reaction_3d'].transform(np.random.permutation)
shuffle_corr = oos_corr['earnings_explosiveness_score'].corr(oos_corr['react_shuffled'])
print(f'Shuffle-within-stock corr:          {shuffle_corr:.3f}  -- what remains if timing signal is destroyed')
print('\nInterpretation: pooled - shuffle = timing contribution')
print(f'  Timing contribution: {pooled - shuffle_corr:.3f}  |  Between-stock persistence: {shuffle_corr:.3f}')

# ── Product Report ──────────────────────────────────────────────────────────
# Metrics a subscriber or investor would care about: how often does the model
# flag events, how often are those events extreme, and how much signal is there
# vs just picking randomly.

print('\n\n========================================')
print('BREAKWATER PRODUCT REPORT')
print('========================================')

# --- 1. Bucket distribution and realized rates (OOS 2011-2025) ---
oos = earn[earn['year'].between(2011, 2025)].copy()
bucket_order_full = ['Normal', 'Elevated', 'High Alert']

print('\n--- 1. Risk Bucket Profile (OOS 2011-2025) ---')
p_global_oos = oos['is_extreme_reaction'].mean()
for b in bucket_order_full:
    sub = oos[oos['earnings_explosiveness_bucket'] == b]
    if sub.empty: continue
    rate = sub['is_extreme_reaction'].mean()
    n    = len(sub)
    pct  = n / len(oos)
    lift = rate / p_global_oos
    print(f"  {b:<12}  n={n:>5}  ({pct:>5.1%} of events)  extreme rate={rate:.1%}  lift={lift:.2f}x")
print(f"  {'Baseline':<12}  n={len(oos):>5}  (100%)              extreme rate={p_global_oos:.1%}  lift=1.00x")

# --- 2. High Alert signal efficiency: flag ≈7% of events, capture what % of extremes? ---
print('\n--- 2. Signal Efficiency: High Alert (top tier) ---')
ha = oos[oos['earnings_explosiveness_bucket'] == 'High Alert']
all_extremes     = oos['is_extreme_reaction'].sum()
captured         = ha['is_extreme_reaction'].sum()
pct_flagged      = len(ha) / len(oos)
pct_captured     = captured / all_extremes
precision        = ha['is_extreme_reaction'].mean()
print(f"  Flags {pct_flagged:.1%} of all upcoming earnings events")
print(f"  Of those flagged: {precision:.1%} actually move >=8% post-earnings")
print(f"  Captures {pct_captured:.1%} of ALL extreme moves in the universe")
print(f"  Miss rate (extreme moves NOT flagged as High Alert): {1-pct_captured:.1%}")

# --- 3. Year-by-year High Alert lift (robustness check) ---
print('\n--- 3. Year-by-Year High Alert Lift vs Baseline (2011-2025) ---')
print(f"  {'Year':>6}  {'Baseline':>9}  {'HighAlert':>10}  {'Lift':>6}  {'n_HA':>6}  {'Captured%':>10}")
annual = []
for y in range(2011, 2026):
    yr = oos[oos['year'] == y]
    ha_yr = yr[yr['earnings_explosiveness_bucket'] == 'High Alert']
    if yr.empty or ha_yr.empty: continue
    base   = yr['is_extreme_reaction'].mean()
    ha_rt  = ha_yr['is_extreme_reaction'].mean()
    lift_y = ha_rt / base if base > 0 else np.nan
    cap    = ha_yr['is_extreme_reaction'].sum() / yr['is_extreme_reaction'].sum()
    annual.append({'year': y, 'base': base, 'ha_rate': ha_rt, 'lift': lift_y, 'n_ha': len(ha_yr), 'capture': cap})
    print(f"  {y:>6}  {base:>9.1%}  {ha_rt:>10.1%}  {lift_y:>6.2f}x  {len(ha_yr):>6}  {cap:>10.1%}")
ann_df = pd.DataFrame(annual)
print(f"  {'AVERAGE':>6}  {ann_df['base'].mean():>9.1%}  {ann_df['ha_rate'].mean():>10.1%}  {ann_df['lift'].mean():>6.2f}x  {ann_df['n_ha'].mean():>6.0f}  {ann_df['capture'].mean():>10.1%}")
print(f"  Lift > 2x in {(ann_df['lift'] > 2).sum()}/{len(ann_df)} years  |  Min lift: {ann_df['lift'].min():.2f}x  |  Max lift: {ann_df['lift'].max():.2f}x")

# --- 4. Options relevance: High Alert vs implied vol proxy ---
# Proxy: extreme rate IS the realized tail probability. Compare to a flat 11.6% baseline.
# A trader buying straddles on High Alert stocks is working with a ~38% realized pop rate
# vs paying premiums priced for ~12% — structural edge.
print('\n--- 4. Structural Edge for Options / Volatility Strategies ---')
print(f"  Market baseline (all S&P500 earnings): {p_global_oos:.1%} extreme-move rate")
print(f"  High Alert realized rate:              {ha['is_extreme_reaction'].mean():.1%}")
edge = ha['is_extreme_reaction'].mean() / p_global_oos
print(f"  If implied vol is priced near baseline: {edge:.1f}x realized-to-implied edge on High Alert names")
print(f"  (This is structural/historical — not a live vol comparison)")

# --- 5. Score rank-ordering: Pearson corr by year ---
print('\n--- 5. Score Rank-Ordering Quality (Pearson corr, OOS) ---')
corrs = []
for y in range(2011, 2026):
    sub = oos[oos['year'] == y][['earnings_explosiveness_score', 'abs_reaction_3d']].dropna()
    if len(sub) < 50: continue
    c = sub['earnings_explosiveness_score'].corr(sub['abs_reaction_3d'])
    corrs.append(c)
    print(f"  {y}: corr={c:.3f}")
print(f"  Average: {np.mean(corrs):.3f}  |  Median: {np.median(corrs):.3f}  |  All positive: {all(c > 0 for c in corrs)}")

print('\n========================================\n')

# ── Decile calibration: is the score itself well-ordered? ──
# Freeze decile edges on 2005-2010 train data, evaluate each test year.
train_all = earn[earn['year'] < 2011]
_, edges = pd.qcut(train_all['earnings_explosiveness_score'], q=10, retbins=True, duplicates='drop')

decile_rows = []
for y in range(2011, 2026):
    test = earn[earn['year'] == y].copy()
    if test.empty:
        continue
    test['decile'] = pd.cut(test['earnings_explosiveness_score'], bins=edges, labels=False, include_lowest=True)
    for d, g in test.groupby('decile'):
        decile_rows.append({'year': y, 'decile': int(d)+1, 'avg_score': g['earnings_explosiveness_score'].mean(), 'actual_rate': g['is_extreme_reaction'].mean(), 'n': len(g)})

deciles = pd.DataFrame(decile_rows)
dec_agg = deciles.groupby('decile').agg(avg_score=('avg_score','mean'), actual_pct=('actual_rate', lambda x: x.mean()*100), total_n=('n','sum')).round(2)
print('\nDecile calibration (OOS 2011-2025, edges frozen on 2005-2010):')
print(dec_agg.to_string())

def oos_overlay_validation():
    # ── OOS Validation: Surprise Momentum & Pre-Earnings Drift overlays ───────
    print('\n\n========================================')
    print('OOS VALIDATION: DYNAMIC OVERLAYS')
    print('========================================')

    oos = earn[earn['year'].between(2011, 2025)].copy()
    baseline_rate = oos['is_extreme_reaction'].mean()

    # --- 1. Surprise streak vs abs reaction ---
    print('\n--- 1. Surprise Streak vs Reaction Magnitude (OOS 2011-2025) ---')
    streak_bins   = [-np.inf, -3, -1, 1, 4, 6, np.inf]
    streak_labels = ['<=-3 (Miss Streak)', '-2 to -1', '0 to 1', '2 to 4', '5 to 6', '>=6 (Overdue Miss)']
    oos_streak = oos.dropna(subset=['surprise_streak', 'abs_reaction_3d']).copy()
    oos_streak['streak_bucket'] = pd.cut(oos_streak['surprise_streak'], bins=streak_bins, labels=streak_labels)
    streak_tbl = (
        oos_streak.groupby('streak_bucket', observed=True)
        .agg(n=('abs_reaction_3d', 'count'), avg_abs_reaction=('abs_reaction_3d', 'mean'),
             extreme_rate=('is_extreme_reaction', 'mean'), signed_reaction=('reaction_3d', 'mean'))
        .round(4)
    )
    streak_tbl['lift'] = (streak_tbl['extreme_rate'] / baseline_rate).round(2)
    print(streak_tbl.to_string())

    # --- 2. Surprise momentum flag vs reaction ---
    print('\n--- 2. Surprise Momentum Flag vs Reaction (OOS 2011-2025) ---')
    flag_tbl = (
        oos.groupby('surprise_momentum_flag', observed=True)
        .agg(n=('abs_reaction_3d', 'count'), avg_abs_reaction=('abs_reaction_3d', 'mean'),
             extreme_rate=('is_extreme_reaction', 'mean'), signed_reaction=('reaction_3d', 'mean'))
        .round(4)
    )
    flag_tbl['lift'] = (flag_tbl['extreme_rate'] / baseline_rate).round(2)
    print(flag_tbl.to_string())

    # --- 3. Pre-earnings drift z deciles vs reaction ---
    print('\n--- 3. Pre-Earnings Drift Z vs Reaction (OOS 2011-2025) ---')
    oos_drift = oos.dropna(subset=['pre_earnings_drift_z', 'abs_reaction_3d']).copy()
    oos_drift['drift_decile'] = pd.qcut(oos_drift['pre_earnings_drift_z'], q=10, labels=False, duplicates='drop')
    drift_tbl = (
        oos_drift.groupby('drift_decile', observed=True)
        .agg(n=('abs_reaction_3d', 'count'), avg_drift_z=('pre_earnings_drift_z', 'mean'),
             avg_abs_reaction=('abs_reaction_3d', 'mean'), signed_reaction=('reaction_3d', 'mean'),
             extreme_rate=('is_extreme_reaction', 'mean'))
        .round(4)
    )
    print(drift_tbl.to_string())
    corr_abs    = oos_drift['pre_earnings_drift_z'].corr(oos_drift['abs_reaction_3d'])
    corr_signed = oos_drift['pre_earnings_drift_z'].corr(oos_drift['reaction_3d'])
    print(f'Corr drift_z vs abs_reaction: {corr_abs:.3f}  |  vs signed reaction: {corr_signed:.3f}')

    # --- 4. Drift flag vs reaction ---
    print('\n--- 4. Pre-Earnings Drift Flag vs Reaction (OOS 2011-2025) ---')
    drift_flag_tbl = (
        oos.groupby('pre_earnings_drift_flag', observed=True)
        .agg(n=('abs_reaction_3d', 'count'), avg_abs_reaction=('abs_reaction_3d', 'mean'),
             extreme_rate=('is_extreme_reaction', 'mean'), signed_reaction=('reaction_3d', 'mean'))
        .round(4)
    )
    drift_flag_tbl['lift'] = (drift_flag_tbl['extreme_rate'] / baseline_rate).round(2)
    print(drift_flag_tbl.to_string())

    # --- 5. Coverage on High Alert names ---
    print('\n--- 5. Flag Coverage on High Alert Events ---')
    ha = oos[oos['earnings_explosiveness_bucket'] == 'High Alert']
    ha_surprise = (ha['surprise_momentum_flag'] != '').sum()
    ha_drift    = (ha['pre_earnings_drift_flag'] != '').sum()
    ha_either   = ((ha['surprise_momentum_flag'] != '') | (ha['pre_earnings_drift_flag'] != '')).sum()
    print(f'  High Alert events:   {len(ha)}')
    print(f'  With surprise flag:  {ha_surprise} ({ha_surprise/len(ha):.1%})')
    print(f'  With drift flag:     {ha_drift} ({ha_drift/len(ha):.1%})')
    print(f'  With either flag:    {ha_either} ({ha_either/len(ha):.1%})')
    print('\n========================================\n')

def high_conviction_stacking_test():
    # ── Do the signals stack? High Alert + dynamic flag vs High Alert alone ───
    print('\n\n========================================')
    print('HIGH CONVICTION STACKING TEST')
    print('========================================')

    oos = earn[earn['year'].between(2011, 2025)].copy()
    baseline_rate = oos['is_extreme_reaction'].mean()

    ha   = oos[oos['earnings_explosiveness_bucket'] == 'High Alert']
    norm = oos[oos['earnings_explosiveness_bucket'] == 'Normal']

    def stats(subset, label):
        n     = len(subset)
        rate  = subset['is_extreme_reaction'].mean()
        lift  = rate / baseline_rate
        signed = subset['reaction_3d'].mean()
        pct_universe = n / len(oos)
        print(f"  {label:<45}  n={n:>5} ({pct_universe:>5.1%})  extreme={rate:.1%}  lift={lift:.2f}x  signed={signed:+.3f}")

    print(f'\n  Baseline (all events):                                 extreme={baseline_rate:.1%}  lift=1.00x')
    print()
    stats(norm,                                                       'Normal bucket')
    stats(ha,                                                         'High Alert (any)')
    print()

    # High Alert split by each dynamic flag
    stats(ha[ha['surprise_momentum_flag'] == ''],                     'High Alert + no surprise flag')
    stats(ha[ha['surprise_momentum_flag'] == 'Beat Streak'],          'High Alert + Beat Streak')
    stats(ha[ha['surprise_momentum_flag'] == 'Overdue Miss'],         'High Alert + Overdue Miss')
    stats(ha[ha['surprise_momentum_flag'] == 'Miss Streak'],          'High Alert + Miss Streak')
    stats(ha[ha['surprise_momentum_flag'] == 'Erratic'],              'High Alert + Erratic')
    print()
    stats(ha[ha['pre_earnings_drift_flag'] == ''],                    'High Alert + no drift flag')
    stats(ha[ha['pre_earnings_drift_flag'] == 'Extended'],            'High Alert + Extended drift')
    stats(ha[ha['pre_earnings_drift_flag'] == 'Compressed'],           'High Alert + Compressed drift')
    print()

    # High Conviction tier: High Alert + any flag
    any_flag = (oos['surprise_momentum_flag'] != '') | (oos['pre_earnings_drift_flag'] != '')
    ha_any   = ha[any_flag.loc[ha.index]]
    ha_none  = ha[~any_flag.loc[ha.index]]
    stats(ha_any,  'High Alert + any flag (High Conviction)')
    stats(ha_none, 'High Alert + no flags at all')
    print()

    # Best combinations
    ha_erratic_dep  = ha[(ha['surprise_momentum_flag'] == 'Erratic') |
                         (ha['pre_earnings_drift_flag'] == 'Compressed')]
    ha_dep_ext      = ha[ha['pre_earnings_drift_flag'].isin(['Compressed', 'Extended'])]
    ha_erratic      = ha[ha['surprise_momentum_flag'] == 'Erratic']
    ha_dep          = ha[ha['pre_earnings_drift_flag'] == 'Compressed']
    stats(ha_erratic_dep, 'High Alert + (Erratic OR Compressed)')
    stats(ha_dep_ext,     'High Alert + any drift flag')
    stats(ha_erratic,     'High Alert + Erratic only')
    stats(ha_dep,         'High Alert + Compressed only')
    print('\n========================================\n')

def high_conviction_effectiveness():
    print('\n\n========================================')
    print('HIGH CONVICTION: EFFECTIVENESS DRILL-DOWN')
    print('========================================')

    oos = earn[earn['year'].between(2011, 2025)].copy()
    baseline_rate = oos['is_extreme_reaction'].mean()

    hc = oos[
        (oos['earnings_explosiveness_bucket'] == 'High Alert') &
        (oos['pre_earnings_drift_flag'] != '')
    ].copy()

    ha_only = oos[
        (oos['earnings_explosiveness_bucket'] == 'High Alert') &
        (oos['pre_earnings_drift_flag'] == '')
    ].copy()

    # --- Year-by-year HC performance ---
    print(f'\n--- Year-by-Year: High Conviction vs High Alert (no flag) ---')
    print(f"  {'Year':>5}  {'HC n':>6}  {'HC rate':>8}  {'HC lift':>8}  {'HA n':>6}  {'HA rate':>8}  {'HA lift':>8}")
    yy_rows = []
    for y in range(2011, 2026):
        hc_y  = hc[hc['year'] == y]
        ha_y  = ha_only[ha_only['year'] == y]
        base_y = oos[oos['year'] == y]['is_extreme_reaction'].mean()
        if hc_y.empty and ha_y.empty: continue
        hc_rate = hc_y['is_extreme_reaction'].mean() if len(hc_y) else np.nan
        ha_rate = ha_y['is_extreme_reaction'].mean() if len(ha_y) else np.nan
        hc_lift = hc_rate / base_y if (len(hc_y) and base_y > 0) else np.nan
        ha_lift = ha_rate / base_y if (len(ha_y) and base_y > 0) else np.nan
        yy_rows.append({'year': y, 'hc_n': len(hc_y), 'hc_rate': hc_rate, 'hc_lift': hc_lift,
                        'ha_n': len(ha_y), 'ha_rate': ha_rate, 'ha_lift': ha_lift})
        print(f"  {y:>5}  {len(hc_y):>6}  {hc_rate:>8.1%}  {hc_lift:>8.2f}x  {len(ha_y):>6}  {ha_rate:>8.1%}  {ha_lift:>8.2f}x")
    yy = pd.DataFrame(yy_rows)
    print(f"  {'AVG':>5}  {yy['hc_n'].mean():>6.1f}  {yy['hc_rate'].mean():>8.1%}  {yy['hc_lift'].mean():>8.2f}x  "
          f"{yy['ha_n'].mean():>6.1f}  {yy['ha_rate'].mean():>8.1%}  {yy['ha_lift'].mean():>8.2f}x")
    hc_beat_ha = (yy['hc_lift'] > yy['ha_lift']).sum()
    print(f'\n  HC outperforms HA (no flag) in {hc_beat_ha}/{len(yy)} years')

    # --- Split: Extended vs Compressed ---
    print('\n--- Extended vs Compressed: do both tails compound equally? ---')
    for flag in ['Extended', 'Compressed']:
        sub = hc[hc['pre_earnings_drift_flag'] == flag]
        rate = sub['is_extreme_reaction'].mean()
        avg_move = sub['abs_reaction_3d'].mean()
        signed   = sub['reaction_3d'].mean()
        print(f"  {flag:<12}  n={len(sub):>4}  extreme={rate:.1%}  lift={rate/baseline_rate:.2f}x  "
              f"avg_abs={avg_move:.1%}  signed={signed:+.2%}")

    # --- Most notable individual HC events ---
    print('\n--- Notable High Conviction events (extreme move = YES) ---')
    hits = hc[hc['is_extreme_reaction'] == 1].copy()
    hits['abs_reaction_3d_pct'] = (hits['abs_reaction_3d'] * 100).round(1)
    hits['reaction_3d_pct']     = (hits['reaction_3d']     * 100).round(1)
    cols = ['stock', 'earnings_date', 'pre_earnings_drift_flag', 'pre_earnings_drift_z',
            'surprise_momentum_flag', 'earnings_explosiveness_score', 'abs_reaction_3d_pct', 'reaction_3d_pct']
    print(hits[cols].sort_values('abs_reaction_3d_pct', ascending=False).head(20).to_string(index=False))

    # --- Misses: HC events that did NOT produce extreme move ---
    print('\n--- High Conviction misses (extreme move = NO) ---')
    misses = hc[hc['is_extreme_reaction'] == 0].copy()
    misses['abs_reaction_3d_pct'] = (misses['abs_reaction_3d'] * 100).round(1)
    misses['reaction_3d_pct']     = (misses['reaction_3d']     * 100).round(1)
    print(misses[cols].sort_values('abs_reaction_3d_pct', ascending=False).head(10).to_string(index=False))

    print(f'\n  Total HC events: {len(hc)} | Hits: {len(hits)} ({len(hits)/len(hc):.1%}) | Misses: {len(misses)} ({len(misses)/len(hc):.1%})')
    print('\n========================================\n')

oos_overlay_validation()
high_conviction_stacking_test()
high_conviction_effectiveness()
exit()
# ── Grid search: normalization denominator for earnings_explosiveness_score ──
# Recomputes gated_explosiveness_score inline for each candidate denominator.
# No need to re-run main.py — raw features are already in the parquet.
# Key metric: avg OOS Pearson correlation (walk-forward, year-by-year).

def score_at_denom(df, denom):
    epsilon = 1e-6
    vol = np.maximum(df["vol_30d"], epsilon)
    p75 = df["abs_reaction_p75_rolling"].fillna(df["abs_reaction_p75"])
    e1 = (df["earnings_explosiveness_z"].fillna(0) / denom).clip(0, 1)
    e2 = (p75 / vol / denom).clip(0, 1)
    e3 = (p75 / 0.12).clip(0, 1)
    e4 = np.clip(df["reaction_entropy"], 0, 1)
    exp = 100 * np.clip(0.35 * e2 + 0.30 * e1 + 0.25 * e3 + 0.10 * e4, 0, 1)
    vol_gate = (
        1.0
        + 0.4 * (df["stock_vs_sector_vol"].fillna(1) > 1).astype(float)
        + 0.3 * df["vol_stress_extreme"].fillna(0).astype(float)
    )
    return (exp * vol_gate).clip(0, 100)


def avg_oos_corr(df, score_series, date_col="date", target_col="abs_reaction_3d"):
    d = df[df["is_earnings_day"] == 1][[date_col, target_col]].copy()
    d["score"] = score_series.loc[d.index]
    d = d.dropna()
    d[date_col] = pd.to_datetime(d[date_col])
    corrs = []
    for y in sorted(d[date_col].dt.year.unique())[1:]:
        train = d[d[date_col] < pd.Timestamp(f"{y}-01-01")]
        test  = d[(d[date_col] >= pd.Timestamp(f"{y}-01-01")) & (d[date_col] < pd.Timestamp(f"{y+1}-01-01"))]
        if len(train) < 500 or len(test) < 100:
            continue
        lo, hi = train["score"].min(), train["score"].max()
        if hi == lo:
            continue
        test_score = (test["score"] - lo) / (hi - lo)
        corrs.append(test_score.corr(test[target_col]))
    return np.mean(corrs) if corrs else np.nan


spot_stocks = ["AAPL", "TSLA", "NVDA", "MSFT"]
results = []

for denom in [4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,25,30,35,40]:
    df["_gated"] = score_at_denom(df, denom)
    corr = avg_oos_corr(df, df["_gated"])
    latest = {}
    for s in spot_stocks:
        sdf = df[(df["stock"] == s) & (df["is_earnings_day"] == 1)]
        latest[s] = f"{sdf['_gated'].iloc[-1]:.0f}" if not sdf.empty else "n/a"
    results.append({"denom": denom, "avg_oos_corr": round(corr, 4), **latest})

df.drop(columns=["_gated"], inplace=True)

print(pd.DataFrame(results).to_string(index=False))

# ── Test: simplified score (no vol-normalized components) ──
def score_simplified(df, e3_w=0.85, e4_w=0.15):
    p75 = df["abs_reaction_p75_rolling"].fillna(df["abs_reaction_p75"])
    e3 = (p75 / 0.12).clip(0, 1)
    e4 = np.clip(df["reaction_entropy"], 0, 1)
    exp = 100 * np.clip(e3_w * e3 + e4_w * e4, 0, 1)
    vol_gate = (
        1.0
        + 0.4 * (df["stock_vs_sector_vol"].fillna(1) > 1).astype(float)
        + 0.3 * df["vol_stress_extreme"].fillna(0).astype(float)
    )
    return (exp * vol_gate).clip(0, 100)

df["_simplified"] = score_simplified(df)
corr_s = avg_oos_corr(df, df["_simplified"])
simple_latest = {}
for s in spot_stocks:
    sdf = df[(df["stock"] == s) & (df["is_earnings_day"] == 1)]
    simple_latest[s] = f"{sdf['_simplified'].iloc[-1]:.0f}" if not sdf.empty else "n/a"
df.drop(columns=["_simplified"], inplace=True)
print(f"\nSimplified (no e1/e2): avg_oos_corr={round(corr_s,4)}  {simple_latest}")


exit()
##################################
# FREE-FORM TESTING 
##################################

df = df.drop(columns=["momentum_fragility_score", "risk_score", "momentum_pressure_regime"] )
df["label_3pct"] = (df["abs_reaction_3d"] >= 0.03).astype(int)
df["label_5pct"] = (df["abs_reaction_3d"] >= 0.05).astype(int)
pre_earnings = df[(df["days_to_earnings"] >= 1) & (df["days_to_earnings"] <= 10)]
earnings_df = df[df["is_earnings_day"] == 1].copy()

##################################
# Backtesting features
##################################
print("##################################\nBacktesting features\n##################################")
feature = "earnings_explosiveness_score"

test_score_df = df[["stock", "date","earnings_date","abs_reaction_3d", feature]].dropna().copy()
pre_2015 = test_score_df[test_score_df["date"] < "2015-01-01"].copy()
post_2015 = test_score_df[test_score_df["date"] >= "2015-01-01"].copy()

def normalize_with_train(s, train_min, train_max):
    denom = train_max - train_min
    if denom == 0:
        return pd.Series(50, index=s.index)
    return 100 * (s - train_min) / denom

train_min = pre_2015[feature].min()
train_max = pre_2015[feature].max()
pre_2015["score_oos"] = normalize_with_train(
    pre_2015[feature], train_min, train_max
)

post_2015["score_oos"] = normalize_with_train(
    post_2015[feature], train_min, train_max
).clip(0, 100)
pre_corr = pre_2015[["score_oos", "abs_reaction_3d"]].corr().iloc[0,1]
print("Train corr:", pre_corr)

pre_2015["bucket"] = pd.qcut(pre_2015["score_oos"], q=10, labels=False)
print(pre_2015.groupby("bucket")["abs_reaction_3d"].mean())
train_edges = pd.qcut(
    pre_2015["score_oos"],
    q=10,
    retbins=True,
    labels=False
)[1]

post_2015["bucket"] = pd.cut(
    post_2015["score_oos"],
    bins=train_edges,
    labels=False,
    include_lowest=True
)
post_corr = post_2015[["score_oos", "abs_reaction_3d"]].corr().iloc[0,1]
print("Test corr:", post_corr)
print(post_2015.groupby("bucket")["abs_reaction_3d"].mean())

def testing_scores(df):
    print("Running Score Testing...\n--------------------")

    stock_list = pd.read_csv("data/stock_list.csv")
    first_30_stocks = stock_list.iloc[1:31,0]

    global_earnings_df = df[df["is_earnings_day"] == 1].copy()
    
    P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"].mean()
    bucket_stats = pd.DataFrame({
        "global_hist_prob": P_extreme_given_bucket,
        "global_risk_lift_vs_baseline": P_extreme_given_bucket / P_extreme_global
    })
    
    report_txt = open("output/report_txt.txt", "w")
    for stock in first_30_stocks:
        print(f"{stock}")
        stock_df = df[df["stock"] == stock]
        earnings_df = stock_df[stock_df["is_earnings_day"] == 1]
        latest_row = earnings_df.iloc[-1]
        # Bayesian shrinkage: (n_stock * p_stock + prior_strength * p_global) / (n_stock + prior_strength)
        prior_strength = 20
        # Stock historical bucket stats
        earnings_explosiveness_buckets= (
            earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
            .agg(extreme_count="sum", event_count="count")
        )
        earnings_explosiveness_buckets["shrunk_prob"] = (
            earnings_explosiveness_buckets["extreme_count"] +
            prior_strength * bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"] # type: ignore
        ) / (
            earnings_explosiveness_buckets["event_count"] + prior_strength
        )
        earnings_explosiveness_buckets["global_hist_prob"] = bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"]
        # Lift relative to global baseline
        earnings_explosiveness_buckets["lift_vs_baseline"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / P_extreme_global
        )
        # Lift relative to global same-bucket risk
        earnings_explosiveness_buckets["lift_vs_same_bucket_global"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / earnings_explosiveness_buckets["global_hist_prob"]
        )

        current_bucket  = latest_row["earnings_explosiveness_bucket"]

        if type(current_bucket)!= str:
            latest_row = earnings_df.iloc[-2]
            current_bucket = latest_row["earnings_explosiveness_bucket"]

        earnings_explosiveness_score = f"{latest_row["earnings_explosiveness_score"]:.0f}"
        current_earnings_date = latest_row["earnings_date"]
        P_extreme_global = round(P_extreme_global, 3)
        current_bucket_prob = f"{earnings_explosiveness_buckets.loc[current_bucket, "shrunk_prob"]:.3f}"
        current_lift_vs_baseline = f"{earnings_explosiveness_buckets.loc[current_bucket, "lift_vs_baseline"]:.3f}"
        current_lift_vs_same_bucket_global = f"{earnings_explosiveness_buckets.loc[current_bucket, "lift_vs_same_bucket_global"]:.3f}"
        earnings_explosiveness_buckets = earnings_explosiveness_buckets.reset_index()
        # write to file
        report_txt.write(f"\n---------\n{stock}:\n")
        report_txt.write(f"Earnings Date: {current_earnings_date}\n")
        report_txt.write(f"Tail Risk Score: {earnings_explosiveness_score}\n")
        report_txt.write(f"risk_level, {current_bucket}\n")
        report_txt.write(f"base_extreme_prob, {P_extreme_global}\n")
        report_txt.write(f"hist_extreme_prob, {current_bucket_prob}\n")
        report_txt.write(f"current_lift_vs_baseline, {current_lift_vs_baseline}\n")
        report_txt.write(f"current_lift_vs_same_bucket_global, {current_lift_vs_same_bucket_global}\n")

def features_test(df):
    print("Running Feature Testing...\n--------------------")

    earnings_df = df[df["is_earnings_day"] == 1]
    earnings_df["label_3pct"] = (earnings_df["abs_reaction_3d"] >= 0.03).astype(int)
    earnings_df["label_5pct"] = (earnings_df["abs_reaction_3d"] >= 0.05).astype(int)

    earnings_df["rank"] = earnings_df.groupby("earnings_date")["risk_score"].rank(pct=True)
    earnings_df.to_csv("earnings_df_ranked.csv", index=False)

    top = earnings_df[earnings_df["rank"] >= 0.9]   # top 10%
    top["abs_reaction_3d"].mean()
    (top["abs_reaction_3d"] >= 0.05).mean()

    earnings_df["final_signal"] = earnings_df["momentum_fragility_score"]

    extreme_regime_df = earnings_df[earnings_df["earnings_explosiveness_score"] > 85].copy()  # only extreme regime
    extreme_regime_df.to_csv("extreme_regime_df.csv",index=False)

    # Testing best weights for the final risk score
    best_auc = 0
    best_w = None

    for w in np.linspace(0, 1, 21):  # 0.0 → 1.0
        score = w * earnings_df["earnings_explosiveness_score"] + (1 - w) * earnings_df["momentum_fragility_score"]
        
        data = pd.DataFrame({
            "score": score,
            "label": earnings_df["label_5pct"]
        }).dropna()
        
        auc = roc_auc_score(data["label"], data["score"])
        
        if auc > best_auc:
            best_auc = auc
            best_w = w

    # print(best_auc, best_w)

    def evaluate_numeric_feature(df, feature, label_col):
        data = df[[feature, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if data[label_col].nunique() < 2:
            return None
        
        corr = data[feature].corr(data[label_col])
        
        try:
            auc = roc_auc_score(data[label_col], data[feature])
        except:
            auc = np.nan
        
        return corr, auc

    numeric_features = [
        "vol_ratio_cross_sectional_pct",
        "sector_vol_ratio_pct",
        "earnings_explosiveness_z",
        "earnings_tail_z",
        "proximity_score",
        "vol_expansion_score",
        "momentum_fragility_score",
        "earnings_explosiveness_score",
        "risk_score"
    ]

    for feature in numeric_features:
        res3 = evaluate_numeric_feature(earnings_df, feature, "label_3pct")
        res5 = evaluate_numeric_feature(earnings_df, feature, "label_5pct")
        
        print(f"{feature}")
        print(f"  3% -> corr: {res3[0]:.3f}, AUC: {res3[1]:.3f}") # type:ignore
        print(f"  5% -> corr: {res5[0]:.3f}, AUC: {res5[1]:.3f}") # type:ignore

    cat_features = [
        "vol_stress_elevated",
        "vol_stress_extreme",
        "sector_vol_stress_high",
        "momentum_pressure_regime",
        "earnings_explosiveness_bucket"
    ]
    def evaluate_categorical_feature(df, feature, label_col):
        data = df[[feature, label_col]].dropna()
        
        stats = (
            data.groupby(feature)[label_col]
            .agg(events="count", event_rate="mean")
            .sort_values("event_rate", ascending=False)
        )
        
        return stats
    
    # for feature in cat_features:
    #     print(f"\n{feature} (3%)")
    #     print(evaluate_categorical_feature(earnings_df, feature, "label_3pct"))
        
    #     print(f"\n{feature} (5%)")
    #     print(evaluate_categorical_feature(earnings_df, feature, "label_5pct"))

    def bin_analysis(df, feature, label_col, n_bins=10):
        data = df[[feature, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        
        data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")
        
        stats = (
            data.groupby("bin")[label_col]
            .agg(events="count", event_rate="mean")
        )
        
        return stats
    print(bin_analysis(earnings_df, "momentum_fragility_score", "label_5pct"))
    # print(bin_analysis(earnings_df, "earnings_explosiveness_score", "label_5pct"))
    # print(bin_analysis(earnings_df, "risk_score", "label_5pct") )


earnings_df["fragility_pct"] = (
    earnings_df.groupby("date")["momentum_fragility_score"]
    .rank(pct=True)
)

earnings_df["fragility_display_score"] = 100 * earnings_df["fragility_pct"]
earnings_df["bucket"] = pd.qcut(earnings_df["momentum_fragility_score"], 10, labels=False)
earnings_df["rank"] = earnings_df.groupby("earnings_date")["momentum_fragility_score"].rank(pct=True)
top = earnings_df[earnings_df["rank"] >= 0.9]   # top 10%
# print( earnings_df.groupby("bucket")["label_5pct"].mean() )
# print(earnings_df["fragility_pct"])
