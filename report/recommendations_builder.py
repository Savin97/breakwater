def build_recommendation(risk_level, hist_extreme_prob, base_extreme_prob,
                         lift, surprise_flag, drift_flag, high_conviction,
                         stock, earnings_date):
    hist_pct = round(float(hist_extreme_prob) * 100, 1)
    base_pct = round(float(base_extreme_prob) * 100, 1)
    lift_x   = round(float(lift), 1)

    flag_lines = []
    if drift_flag == "Extended":
        flag_lines.append(
            "Pre-earnings drift is Extended — the stock has been trending upward into earnings, "
            "raising the bar for a positive surprise and increasing the risk of a sharp reversal."
        )
    elif drift_flag == "Compressed":
        flag_lines.append(
            "Pre-earnings drift is Compressed — unusually quiet price action ahead of earnings "
            "often precedes a sharp move in either direction."
        )

    if surprise_flag == "Beat Streak":
        flag_lines.append(
            "Beat Streak — the stock has beaten consensus estimates in recent quarters. "
            "Elevated expectations increase sensitivity to any earnings disappointment."
        )
    elif surprise_flag == "Miss Streak":
        flag_lines.append(
            "Miss Streak — consecutive estimate misses signal persistent downside pressure "
            "heading into this event."
        )
    elif surprise_flag == "Erratic":
        flag_lines.append(
            "Erratic surprise pattern — inconsistent results make the earnings outcome "
            "harder to predict and the price reaction harder to anticipate."
        )
    elif surprise_flag == "Overdue Miss":
        flag_lines.append(
            "Overdue Miss — after a run of beats, the stock may face a higher risk "
            "of a negative surprise this quarter."
        )

    if risk_level == "Normal":
        headline = "Normal — No Special Action Required"
        body = (
            f"{stock} shows no elevated tail risk ahead of earnings on {earnings_date}. "
            f"The historical probability of an extreme move (>8% in 3 days) is {hist_pct}%, "
            f"in line with the {base_pct}% market average."
        )
        action = "Standard position management applies. No special action required before earnings."
        flag_lines = []

    elif risk_level == "Elevated":
        headline = "Elevated Risk — Light Caution Advised"
        body = (
            f"{stock} shows moderately elevated tail risk ahead of earnings on {earnings_date}. "
            f"The historical probability of an extreme move (>8% in 3 days) is {hist_pct}% — "
            f"{lift_x}x the market average of {base_pct}%."
        )
        action = (
            "Review your position size ahead of earnings. "
            "If you hold a concentrated position, light hedging may be appropriate."
        )

    elif risk_level == "High Alert" and not high_conviction:
        headline = "High Alert — Heightened Caution Warranted"
        body = (
            f"{stock} has a structurally elevated earnings jump risk. "
            f"The historical probability of an extreme move (>8% in 3 days) is {hist_pct}% — "
            f"{lift_x}x the market average of {base_pct}%."
        )
        action = (
            "Consider reducing exposure or adding downside protection before earnings. "
            "Monitor closely in the days leading up to the event."
        )

    else:  # High Alert + high conviction
        headline = "High Alert ★ Highest Priority"
        body = (
            f"{stock} combines a historically explosive earnings profile with active timing signals "
            f"this quarter. The historical probability of an extreme move (>8% in 3 days) is "
            f"{hist_pct}% — {lift_x}x the market average of {base_pct}%."
        )
        action = (
            "Strong case for hedging or reducing exposure before earnings. "
            "This is the highest-priority flag in the Breakwater framework."
        )

    return {
        "headline": headline,
        "body":     body,
        "action":   action,
        "flag_lines": flag_lines,
    }
