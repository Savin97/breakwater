# backtesting/testing_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
    Backtesting & Regime Evaluation Utilities for Breakwater

    This module contains diagnostic, validation, and regime-testing functions
    used to evaluate Breakwater's earnings-period risk framework.

    Purpose
    -------
    Provide tools to:
    • Validate individual feature predictive power
    • Test multi-factor "high-risk" regimes
    • Compare regime performance vs volatility-only baselines
    • Measure precision/recall of extreme-move detection
    • Run train/test and walk-forward robustness checks
    • Inspect distributional stability and lift statistics

    ---------------------------------------------------------------------------
    FEATURE DIAGNOSTICS
    ---------------------------------------------------------------------------

    check_explosiveness_feature(df)
        Tests earnings_explosiveness_score via quantile bucketing and compares
        large/extreme move rates across buckets.

    check_timing_danger_connection_to_earnings_move_bucket(df)
        Decile analysis of timing_danger vs large/extreme move outcomes,
        including lift vs baseline and confidence intervals.

    check_corr_of_features(df)
        Prints correlation matrix between core components:
        proximity, volatility expansion, momentum fragility,
        explosiveness, and timing_danger.

    check_timing_danger_score_metric(df)
        Evaluates distribution, dispersion, and bucketed reaction behavior
        of the timing_danger score (including normalization diagnostics).

    ---------------------------------------------------------------------------
    REGIME TESTING
    ---------------------------------------------------------------------------

    three_way_regime_test(df)
        Tests the 3-factor regime hypothesis:
        high timing_danger + elevated idiosyncratic vol +
        high explosiveness, and compares hit rates vs baseline.

    breakwater_regime_test(df)
        Computes regime statistics (event count, large/extreme rates)
        for the defined Breakwater regime mask.

    volatility_only_regime_test(df)
        Baseline comparison using volatility-only conditions
        (high vol_30d + elevated idiosyncratic vol).

    evaluate_high_risk_earnings_regime(df)
        Defines a stricter "high-risk" regime and runs threshold grids
        to evaluate sensitivity of extreme-move concentration.

    comparing_regime_results_to_volatility_only(df)
        Stress-tests regime performance vs volatility-only signals,
        including shuffled-label checks and stability diagnostics.

    ---------------------------------------------------------------------------
    CLASSIFICATION METRICS
    ---------------------------------------------------------------------------


    (Threshold sweep block)
        Iterates multiple quantile thresholds and evaluates confusion
        metrics for each regime definition.

    ---------------------------------------------------------------------------
    ROBUSTNESS & OUT-OF-SAMPLE VALIDATION
    ---------------------------------------------------------------------------

    check_timing_danger_train_test(df)
        Pre/post split validation ensuring normalization and bucket
        edges are trained only on prior data.

    yearly_oos_report(df, ...)
        Walk-forward yearly OOS test computing correlation,
        bucket separation, and lift statistics per year.

    
    ---------------------------------------------------------------------------
    Forward Evaluating
    ---------------------------------------------------------------------------
    forward_eval_onefactor

    forward_eval_twofactor_and

    All evaluations are performed on earnings-day events unless otherwise specified.
"""

def check_explosiveness_feature(df):
    """ 
        Extreme move rate: 0.10421515386604603
        explosive_bucket
        (18.035, 46.526]    0.066775
        (46.526, 57.006]    0.035889
        (57.006, 68.114]    0.037520
        (68.114, 80.606]    0.089723
        (80.606, 100.0]     0.307818

        The highest explosiveness bucket gives:
        P(extreme move | top explosiveness) ≈ 31%
        vs baseline:
        P(extreme move) ≈ 10%

        That's roughly a 3x risk multiplier.
        That's strong signal for a single feature. Very strong, actually.

        This gives results:
        "Stocks with historically explosive earnings behavior are about 
        three times more likely to experience an extreme earnings move."

        Why this validates your explosiveness design
        It confirms:

        median reaction component is useful
        p75 tail component is useful
        normalization by vol_30d works
        scoring compression didn't kill signal
        requiring 8 historical earnings events was reasonable

        All of that survived testing.

        You just discovered something real about earnings risk:
        Extreme earnings moves are heavily stock-dependent.
        Some stocks are simply "earnings explosives."

    """
    # Check correlation structure of timing_danger and its components
    subset = df[df["is_earnings_day"] == True].copy()
    print("Extreme move rate:", subset["is_extreme_reaction"].mean())

    subset["explosive_bucket"] = pd.qcut(
        subset["earnings_explosiveness_score"],
        5,
        duplicates="drop"
    )

    print(subset.groupby("explosive_bucket")["is_extreme_reaction"].mean())
    print(subset.groupby("explosive_bucket").size())

    top = subset[subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(0.8)]
    bottom = subset[subset["earnings_explosiveness_score"] <= subset["earnings_explosiveness_score"].quantile(0.2)]

    print("Top bucket extreme rate:", top["is_extreme_reaction"].mean())
    print("Bottom bucket extreme rate:", bottom["is_extreme_reaction"].mean())

def three_way_regime_test(df, score_feature):
    """
        This is a temporary function to test the three-way regime hypothesis:
        Tail risk increases materially when:
        timing_danger is high
        stock_vs_sector_vol ≥ 1
        earnings_explosiveness_score is in the top decile

        We will check the hit rates for large_plus and extreme earnings moves across different regimes.
    """
    # separate back testing df to only earnings days
    bt = df[df["is_earnings_day"] == 1].copy()

    # Define the three regime conditions:
    # High timing danger (top decile)
    danger_cutoff = bt[score_feature].quantile(0.9)
    bt["high_danger"] = (bt[score_feature] >= danger_cutoff).astype(int)

    # High individual volatility
    bt["high_individual_vol"] = (bt["stock_vs_sector_vol"] >= 1).astype(int)

    # High explosiveness (top decile)
    exp_cut = bt["earnings_explosiveness_score"].quantile(0.9)
    bt["high_explosiveness"] = (
        bt["earnings_explosiveness_score"] >= exp_cut
    ).astype(int)

    baseline = {
        "group": "ALL",
        "n": len(bt),
        "p_large_plus": bt["is_large_reaction"].mean(),
        "p_extreme": bt["is_extreme_reaction"].mean(),
    }

    mask = (
        (bt["high_danger"] == 1) &
        (bt["high_individual_vol"] == 1) &
        (bt["high_explosiveness"] == 1)
        )

    three_way = {
        "group": "High danger + individual vol + explosiveness",
        "n": mask.sum(),
        "p_large_plus": bt.loc[mask, "is_large_reaction"].mean(),
        "p_extreme": bt.loc[mask, "is_extreme_reaction"].mean(),
    }
    print( pd.DataFrame([baseline, three_way]) )

def conditional_hit_rate_analysis(df, score_feature):
    """ 
        We already proved something important:
        Unconditional timing_danger ≠ large move predictor
        So now we ask the correct question:
        In which regimes does earnings risk actually turn into realized large moves?
        That means conditioning.
    """
    bt = df[df["is_earnings_day"] == 1].copy()

    # We want:
    # vol_stress_extreme == 1
    # Sector earnings crowding: sector_earnings_density >= median
    # Stock-specific volatility dominance: stock_vs_sector_vol >= 1
    # median split for density
    density_med = bt["sector_earnings_density"].median()
    bt["dense_earnings"] = (bt["sector_earnings_density"] >= density_med).astype(int)

    bt["high_individual_vol"] = (bt["stock_vs_sector_vol"] >= 1).astype(int)

    danger_cut = bt[score_feature].quantile(0.9)
    bt["high_danger"] = (bt[score_feature] >= danger_cut).astype(int)

    def cond_table(df, mask, label):
        sub = df[mask]
        return {
            "group": label,
            "n": len(sub),
            "p_large_plus": sub["is_large_reaction"].mean(),
            "p_extreme": sub["is_extreme_reaction"].mean(),
        }

    rows = []

    # baseline
    rows.append(cond_table(bt, bt.index == bt.index, "ALL"))

    # danger only
    rows.append(cond_table(bt, bt["high_danger"] == 1, "High danger"))

    # danger + vol stress
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["vol_stress_extreme"] == 1),
        "High danger + vol stress extreme"
    ))

    # danger + low density
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["dense_earnings"] == 0),
        "High danger + low earnings density"
    ))

    # danger + idio vol
    rows.append(cond_table(
        bt,
        (bt["high_danger"] == 1) & (bt["high_individual_vol"] == 1),
        "High danger + high idio vol"
    ))
    print(pd.DataFrame(rows))

def check_feature_connection_to_large_reacion_metric(df, feature):
    
    ## Diagnostic: timing_danger connection to earnings_move_bucket
    # Earnings-only rows (use your actual column name)
    bt = df[df["is_earnings_day"] == 1].copy()

    # Safety: drop rows missing what you need
    #bt = bt.dropna(subset=["timing_danger", "earnings_move_bucket"])
    bt = bt.dropna(subset=[feature, "reaction_3d"])

    # 10 deciles: 1..10
    bt[f"{feature}_decile"] = pd.qcut(
        bt[feature],
        q=10,
        labels=False,
        duplicates="drop" # avoids errors if timing_danger has repeated values
    ) + 1

    # Define two targets:
        # large+ = bucket >= 1
        # extreme = bucket == 2

        # Interpretation rules:
        # p_large_plus should generally increase from D1 -> D10.
        # lift_large_plus in D10:
        # ~1.0 = no signal
        # 1.3–1.7 = usable
        # 2.0+ = strong
        # p_extreme is rarer, so it'll be noisier; lift matters more than smoothness.

    decile_table = (
        bt.groupby(f"{feature}_decile")
        .agg(
            n=(feature, "size"),
            avg_danger=(feature, "mean"),
            p_large_plus=("is_large_reaction", "mean"),
            p_extreme=("is_extreme_reaction", "mean"),
        )
        .reset_index()
    )

    # Add lift vs overall baseline
    base_large = bt["is_large_reaction"].mean()
    base_ext   = bt["is_extreme_reaction"].mean()

    decile_table["lift_large_plus"] = decile_table["p_large_plus"] / base_large
    decile_table["lift_extreme"]    = decile_table["p_extreme"] / base_ext

    decile_table.sort_values(f"{feature}_decile")

    from math import sqrt
    import numpy as np
    def wilson_ci(p, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        half = (z * sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
        return (center - half, center + half)

    rows = []
    for _, r in decile_table.iterrows():
        n = int(r["n"])
        lo, hi = wilson_ci(r["p_large_plus"], n)
        rows.append((lo, hi))

    decile_table["p_large_plus_ci_lo"] = [x[0] for x in rows]
    decile_table["p_large_plus_ci_hi"] = [x[1] for x in rows]

    print(decile_table)
    print("----------------------")

    print(bt[feature].describe())
    print(bt[feature].nunique())

def check_corr_of_features(df):
    # only earnings ??? 
    df_to_check = df[df["is_earnings_day"] == 1].copy()
    print(df_to_check[
        ["proximity_score",
        "vol_expansion_score",
        "momentum_fragility_score",
        "earnings_explosiveness_score"]
    ].corr())

def breakwater_regime_test(df, score_feature):
    bt = df[df["is_earnings_day"] == 1].copy()

    # timing danger top decile
    danger_cutoff = bt[score_feature].quantile(0.9)
    bt["high_danger"] = (bt[score_feature] >= danger_cutoff)

    # explosiveness top decile
    explosiveness_cutoff = bt["earnings_explosiveness_score"].quantile(0.9)
    bt["high_explosive"] = (bt["earnings_explosiveness_score"] >= explosiveness_cutoff)
    # stock vol > sector vol
    bt["vol_vs_sector"] = (bt["stock_vs_sector_vol"] >= 1)

    regime = (
        bt["vol_vs_sector"] &
        bt["high_explosive"] &
        bt["high_danger"]
    )

    subset = bt[regime]
    results = {
        "n_events": len(subset),
        "extreme_rate": subset["is_extreme_reaction"].mean(),
        "large_plus_rate": subset["is_large_reaction"].mean(),
    }
    print(results)

def volatility_only_regime_test(df):
    """
        Test whether volatility alone can generate high extreme-move rates.
    """
    bt = df[df["is_earnings_day"] == 1].copy()
    # --- Define volatility regime ---
    # Top decile volatility
    vol_cutoff = bt["vol_30d"].quantile(0.9)
    bt["high_vol"] = (bt["vol_30d"] >= vol_cutoff)
    # Stock moving more than sector
    bt["vol_vs_sector"] = (bt["stock_vs_sector_vol"] >= 1)

    regime = bt["high_vol"] & bt["vol_vs_sector"]
    subset = bt[regime]

    results = {
        "n_events": len(subset),
        "extreme_rate": subset["is_extreme_reaction"].mean(),
        "large_plus_rate": subset["is_large_reaction"].mean(),
    }
    print(results)

def evaluate_high_risk_earnings_regime(df, score_feature):
    """
        Define a "high-risk regime" where all three conditions hold:
        timing_danger in top 20%
        stock_vs_sector_vol >= 1
        earnings_explosiveness_score in top 10%

        Prints regime statistics and robustness checks.
    """
    subset = df[df["is_earnings_day"] == True].copy()

    cond = (
        (subset[score_feature] >= subset[score_feature].quantile(0.8)) &
        (subset["stock_vs_sector_vol"] >= 1) &
        (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(0.9))
    )

    print("Sample size:", cond.sum())
    print("Large move rate:", subset.loc[cond, "is_large_reaction"].mean())
    print("Extreme move rate:", subset.loc[cond, "is_extreme_reaction"].mean())

    print("Baseline extreme:", subset["is_extreme_reaction"].mean())
    print("Baseline large:", subset["is_large_reaction"].mean())

    print(subset.loc[cond, "abs_reaction_3d"].describe())
    print(subset.loc[cond].groupby("stock").size().describe())

    for td_q in [0.7, 0.8]:
        for ex_q in [0.8, 0.9]:
            cond = (
                (subset[score_feature] >= subset[score_feature].quantile(td_q)) &
                (subset["earnings_explosiveness_score"] >= subset["earnings_explosiveness_score"].quantile(ex_q)) &
                (subset["stock_vs_sector_vol"] >= 1)
            )

            if cond.sum() > 30:
                print(td_q, ex_q,
                    cond.sum(),
                    subset.loc[cond, "is_extreme_reaction"].mean())

def testing_precision_recall(input_df):
    df = input_df.copy()
    print(f"\n--- Precision & Recall tests ---")
    earnings = df[df["is_earnings_day"] == 1].copy()

    extreme = earnings["is_extreme_reaction"] == 1
    regime = earnings["is_joint_regime"] == 1

    TP = ((regime) & (extreme)).sum()
    FN = ((~regime) & (extreme)).sum()
    FP = ((regime) & (~extreme)).sum()
    TN = ((~regime) & (~extreme)).sum()

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    print("TN:", TN)
    print("recall:", recall)
    print("precision:", precision)
    print(regime.sum(), "events flagged")
    print(len(earnings), "total earnings events")

def comparing_regime_results_to_volatility_only(df):
    earnings_df = df[df.is_earnings_day == 1].sort_values(["stock","date"]).copy()

    # For each stock, compare current p75 with previous event p75
    earnings_df["p75_diff"] = (
        earnings_df.groupby("stock")["abs_reaction_p75"]
        .diff()
    )
    print(earnings_df[["stock","date","abs_reaction_3d","abs_reaction_p75","p75_diff"]].head(20))

    # Test 2 
    earnings_df["extreme_shuffled"] = np.random.permutation(earnings_df["is_extreme_reaction"].values)
    threshold = 0.9
    exp_thr = earnings_df["earnings_explosiveness_score"].quantile(threshold)
    frag_thr = earnings_df["momentum_fragility_score"].quantile(threshold)

    earnings_df["is_joint_regime"] = (
        (earnings_df["earnings_explosiveness_score"] >= exp_thr) &
        (earnings_df["momentum_fragility_score"] >= frag_thr) &
        (earnings_df["is_earnings_day"])
    ).astype(int)

    regime = earnings_df["is_joint_regime"] == 1

    TP = ((regime) & (earnings_df["extreme_shuffled"] == 1)).sum()
    FP = ((regime) & (earnings_df["extreme_shuffled"] == 0)).sum()

    print("Shuffled precision:", TP / (TP + FP))

    # Test 3
    median_year = earnings_df["date"].dt.year.median()

    first_half = earnings_df[earnings_df["date"].dt.year <= median_year]
    second_half = earnings_df[earnings_df["date"].dt.year > median_year]

    def regime_precision(sub):
        regime = sub["is_joint_regime"] == 1
        extreme = sub["is_extreme_reaction"] == 1
        TP = ((regime) & (extreme)).sum()
        FP = ((regime) & (~extreme)).sum()
        return TP / (TP + FP)

    print("First half precision:", regime_precision(first_half))
    print("Second half precision:", regime_precision(second_half))

    # Top volatility test
    threshold = 0.98

    vol_thr = earnings_df["vol_30d"].quantile(threshold)
    regime_vol_top2 = earnings_df["vol_30d"] >= vol_thr
    print(regime_vol_top2)

    regime_vol_top2 = earnings_df["vol_30d"] >= vol_thr
    extreme = earnings_df["is_extreme_reaction"] == 1

    TP = ((regime_vol_top2) & (extreme)).sum()
    FP = ((regime_vol_top2) & (~extreme)).sum()

    precision = TP / (TP + FP)

    print("Top 2% vol events:", regime_vol_top2.sum())
    print("Extreme rate (precision):", precision)

def check_score_metric(input_df, score_feature):
    df = input_df.copy()
    df["abs_reaction_3d"] = df["reaction_3d"].abs()
    df[f"{score_feature}_score"] = (
            100 * (df[score_feature] - df[score_feature].min()) /
            (df[score_feature].max() - df[score_feature].min())
        )
    test_score_df = df[["stock", "date","earnings_date","abs_reaction_3d", f"{score_feature}_score"]].dropna()
    s = test_score_df[f"{score_feature}_score"]
    stats = {
        "count": s.count(),
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "p10": s.quantile(0.10),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
        "p90": s.quantile(0.90),
        "p95": s.quantile(0.95),
    }
    print(stats)
    counts, bins = np.histogram(s, bins=30)

    for i in range(len(counts)):
        print(f"{bins[i]:.4f} to {bins[i+1]:.4f} : {counts[i]}")
    per_day_dispersion = (
        test_score_df
        .groupby("earnings_date")[f"{score_feature}_score"]
        .agg(["mean", "std", "min", "max", "count"])
    )
    per_stock = (
        test_score_df
        .groupby("stock")[f"{score_feature}_score"]
        .agg(["mean", "std", "count"])
    )
    top_10 = s.quantile(0.90)
    bottom_10 = s.quantile(0.10)

    high_tail_rate = (s >= top_10).mean()
    print(high_tail_rate)
    low_tail_rate = (s <= bottom_10).mean()
    print(low_tail_rate)
    corr = test_score_df[[f"{score_feature}_score", "abs_reaction_3d"]].corr()
    print(corr)

    test_score_df["bucket"] = pd.qcut(
        test_score_df[f"{score_feature}_score"], 
        q=5, 
        labels=False
    )

    bucket_stats = test_score_df.groupby("bucket")["abs_reaction_3d"].mean()
    print(bucket_stats)

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title(f"{score_feature} score distribution")
    plt.xlabel(f"{score_feature}_score")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        test_score_df[f"{score_feature}_score"],
        test_score_df["abs_reaction_3d"],
        alpha=0.25
    )
    plt.title(f"{score_feature} score vs abs reaction 3d")
    plt.xlabel(f"{score_feature}_score")
    plt.ylabel("abs_reaction_3d")
    plt.show()

    plt.figure(figsize=(8, 5))
    bucket_stats.plot(kind="bar")
    plt.title(f"Mean abs_reaction_3d by {score_feature} bucket")
    plt.xlabel("Score bucket, low to high")
    plt.ylabel("Mean abs_reaction_3d")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(per_day_dispersion["std"].dropna(), bins=30)
    plt.title(f"Per-earnings-date dispersion of {score_feature} score")
    plt.xlabel("Daily cross-sectional std")
    plt.ylabel("Count")
    plt.show()

def check_feature_train_test(df, feature):
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

    pre_2015["bucket"] = pd.qcut(pre_2015["score_oos"], q=5, labels=False)
    print(pre_2015.groupby("bucket")["abs_reaction_3d"].mean())
    train_edges = pd.qcut(
        pre_2015["score_oos"],
        q=5,
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

def yearly_oos_report(df, date_col, score_feature, target_col, q=5):
    d = df[df["is_earnings_day"]==1]
    d = d[[date_col, score_feature, target_col]].dropna()
    d[date_col] = pd.to_datetime(d[date_col])

    years = sorted(d[date_col].dt.year.unique())
    rows = []

    for y in years[1:]:  # start from 2nd year (need prior data to train)
        train = d[d[date_col] < pd.Timestamp(f"{y}-01-01")]
        test  = d[(d[date_col] >= pd.Timestamp(f"{y}-01-01")) & (d[date_col] < pd.Timestamp(f"{y+1}-01-01"))]

        if len(train) < 500 or len(test) < 100:
            continue

        train_min = train[score_feature].min()
        train_max = train[score_feature].max()
        denom = train_max - train_min
        if denom == 0:
            continue

        # Normalize using train params only
        train_score = 100 * (train[score_feature] - train_min) / denom
        test_score  = 100 * (test[score_feature]  - train_min) / denom

        # Clip to avoid out-of-range due to new extremes
        train_score = train_score.clip(0, 100)
        test_score  = test_score.clip(0, 100)

        # Freeze bucket edges from train
        _, edges = pd.qcut(train_score, q=q, retbins=True, duplicates="drop")

        test_bucket = pd.cut(test_score, bins=edges, labels=False, include_lowest=True)

        test_eval = pd.DataFrame({
            "score": test_score,
            "bucket": test_bucket,
            "target": test[target_col].values
        }).dropna(subset=["bucket"])

        if len(test_eval) < 100:
            continue

        corr = test_eval[["score", "target"]].corr().iloc[0, 1]

        bucket_means = test_eval.groupby("bucket")["target"].mean()
        # separation metrics
        top = bucket_means.max()
        bot = bucket_means.min()
        lift_abs = top - bot
        lift_ratio = (top / bot) if bot != 0 else np.nan

        rows.append({
            "year": y,
            "n_test": len(test_eval),
            "corr": corr,
            "bot_bucket_mean": bot,
            "top_bucket_mean": top,
            "lift_abs": lift_abs,
            "lift_ratio": lift_ratio,
        })

    yearly = pd.DataFrame(rows).sort_values("year")
    print(yearly)
    print("Avg corr:", yearly["corr"].mean())
    print("Median corr:", yearly["corr"].median())
    print("Pct years corr>0.15:", (yearly["corr"] > 0.15).mean())

# ------------------------------------------------------
# Forward Evaluating
# ------------------------------------------------------
"""
    Add this when using forward eval:
    # Expanding prior only
    prior_stats, prior_thr = forward_eval_onefactor(df, "abs_reaction_p75", q=0.90)
    print("-------------------------------\nPRIOR thr:", prior_thr)
    print(prior_stats[prior_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))
    # Rolling prior only 
    roll_stats, roll_thr = forward_eval_onefactor(df, "abs_reaction_p75_rolling", q=0.90)
    print("-------------------------------\nROLL thr:", roll_thr)
    print(roll_stats[roll_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))
    # Prior + fragility (current core)
    pf_stats, (p_thr, f_thr) = forward_eval_twofactor(df, "abs_reaction_p75", "momentum_fragility_score", q=0.90)
    print("-------------------------------\nPRIOR thrs:", p_thr, "\tFRAG thrs: ", f_thr)
    print(pf_stats[pf_stats["split"]=="TEST"][["year","n_regime","regime_extreme_rate","lift","regime_capture_of_extremes"]].to_string(index=False))


"""
def forward_eval_onefactor(
        df,
        eval_feature,
        train_years=range(2005, 2011),
        test_years=range(2011, 2026),
        q=0.90,
        label_col="is_extreme_reaction",
        earn_col="is_earnings_day",
        date_col="date",
    ):
    earn = df[df[earn_col] == 1].dropna(subset=[date_col, label_col, eval_feature]).copy()
    earn["year"] = pd.to_datetime(earn[date_col]).dt.year
    earn["y"] = earn[label_col].astype(int)

    train = earn[earn["year"].isin(train_years)].copy()
    test  = earn[earn["year"].isin(test_years)].copy()

    thr = float(train[eval_feature].quantile(q))

    def add_regime(sub):
        out = sub.copy()
        out["is_regime"] = out[eval_feature] >= thr
        return out

    def stats(sub, split):
        rows = []
        for y, g in add_regime(sub).groupby("year"):
            N = len(g)
            K = int(g["y"].sum())
            base = K / N if N else np.nan

            r = g[g["is_regime"]]
            n = len(r)
            k = int(r["y"].sum())
            reg = k / n if n else np.nan
            lift = (reg / base) if (base and np.isfinite(reg)) else np.nan

            rows.append({
                "split": split,
                "year": int(y),
                "N_earnings": int(N),
                "baseline_extreme_rate": base,
                "n_regime": int(n),
                "regime_extreme_rate": reg,
                "lift": lift,
                "regime_share_of_events": (n / N) if N else np.nan,
                "regime_capture_of_extremes": (k / K) if K else np.nan,
            })
        return pd.DataFrame(rows).sort_values("year")
    return pd.concat([stats(train, "TRAIN"), stats(test, "TEST")], ignore_index=True), thr


def forward_eval_twofactor(
        df,
        feat1, feat2,
        train_years=range(2005, 2011),
        test_years=range(2011, 2026),
        q=0.90,
        label_col="is_extreme_reaction",
        earn_col="is_earnings_day",
        date_col="date",
    ):
    earn = df[df[earn_col] == 1].dropna(subset=[date_col, label_col, feat1, feat2]).copy()
    earn["year"] = pd.to_datetime(earn[date_col]).dt.year
    earn["y"] = earn[label_col].astype(int)

    train = earn[earn["year"].isin(train_years)].copy()
    test  = earn[earn["year"].isin(test_years)].copy()

    thr1 = float(train[feat1].quantile(q))
    thr2 = float(train[feat2].quantile(q))

    def add_regime(sub):
        out = sub.copy()
        out["is_regime"] = (out[feat1] >= thr1) & (out[feat2] >= thr2)
        return out

    def stats(sub, split):
        rows = []
        for y, g in add_regime(sub).groupby("year"):
            N = len(g)
            K = int(g["y"].sum())
            base = K / N if N else np.nan

            r = g[g["is_regime"]]
            n = len(r)
            k = int(r["y"].sum())
            reg = k / n if n else np.nan
            lift = (reg / base) if (base and np.isfinite(reg)) else np.nan

            rows.append({
                "split": split,
                "year": int(y),
                "N_earnings": int(N),
                "baseline_extreme_rate": base,
                "n_regime": int(n),
                "regime_extreme_rate": reg,
                "lift": lift,
                "regime_share_of_events": (n / N) if N else np.nan,
                "regime_capture_of_extremes": (k / K) if K else np.nan,
            })
        return pd.DataFrame(rows).sort_values("year")

    return pd.concat([stats(train, "TRAIN"), stats(test, "TEST")], ignore_index=True), (thr1, thr2)


def high_conviction_regime_test(df, test_years=range(2011, 2026)):
    """
        Testing whether adding the pre_earnings_drift_flag 
        condition on top of "High Alert" actually improve prediction, or is it noise?
    """
    earn = df[df["is_earnings_day"] == 1].copy()
    earn["year"] = pd.to_datetime(earn["date"]).dt.year
    earn["y"] = earn["is_extreme_reaction"].astype(int)
    earn["high_alert"] = earn["earnings_explosiveness_bucket"] == "High Alert"
    drift = earn["pre_earnings_drift_flag"].fillna("") != ""
    surprise = earn["surprise_momentum_flag"].fillna("") != ""

    regimes = {
        "High Alert":                earn["high_alert"],
        "HA + Drift flag":           earn["high_alert"] & drift,
        "HA + Any surprise flag":    earn["high_alert"] & surprise,
        "HA + Beat Streak":          earn["high_alert"] & (earn["surprise_momentum_flag"] == "Beat Streak"),
        "HA + Miss Streak":          earn["high_alert"] & (earn["surprise_momentum_flag"] == "Miss Streak"),
        "HA + Overdue Miss":         earn["high_alert"] & (earn["surprise_momentum_flag"] == "Overdue Miss"),
        "HA + Erratic":              earn["high_alert"] & (earn["surprise_momentum_flag"] == "Erratic"),
        "HA + Drift OR Surprise":    earn["high_alert"] & (drift | surprise),
        "HA + Drift AND Surprise":   earn["high_alert"] & drift & surprise,
        "HA + Drift OR (Surprise ex-OM)": earn["high_alert"] & (
            drift |
            earn["surprise_momentum_flag"].isin(["Beat Streak", "Miss Streak", "Erratic"])
        ),
    }

    test = earn[earn["year"].isin(test_years)].copy()

    rows = []
    for y, g in test.groupby("year"):
        N = len(g)
        K = int(g["y"].sum())
        base = K / N if N else np.nan
        for label, full_mask in regimes.items():
            mask = full_mask[g.index]
            sub = g[mask]
            n = len(sub)
            if n == 0:
                continue
            k = int(sub["y"].sum())
            rate = k / n
            rows.append({"regime": label, "year": int(y), "n": n,
                         "extreme_rate": rate, "lift": rate / base if base else np.nan,
                         "capture": k / K if K else np.nan})

    result = pd.DataFrame(rows).sort_values(["regime", "year"])
    cols = ["year", "n", "extreme_rate", "lift", "capture"]
    base_rate = test["y"].mean()

    print("\n--- Summary (all OOS years pooled) ---")
    summary_rows = []
    for label in regimes:
        grp = result[result["regime"] == label]
        if grp.empty:
            continue
        tot_n = grp["n"].sum()
        avg_extreme = (grp["extreme_rate"] * grp["n"]).sum() / tot_n
        summary_rows.append({
            "regime": label, "total_n": tot_n,
            "n_per_yr": round(tot_n / len(grp), 0),
            "extreme_rate": round(avg_extreme, 3),
            "lift": round(avg_extreme / base_rate, 2),
            "avg_capture": round(grp["capture"].mean(), 3),
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nBaseline extreme rate: {base_rate:.3f}")

    print("\n--- Year-by-year: HA + Any surprise flag ---")
    grp = result[result["regime"] == "HA + Any surprise flag"]
    print(grp[cols].to_string(index=False))

    print("\n--- Year-by-year: HA + Drift OR Surprise ---")
    grp = result[result["regime"] == "HA + Drift OR Surprise"]
    print(grp[cols].to_string(index=False))
