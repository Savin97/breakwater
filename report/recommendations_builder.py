# report/recommendations_builder.py
"""Customer wording for Model B — one place, shared by the PDF, the calendar, the digest
and the dashboard so no surface drifts into saying something the model does not know.

Model B knows one thing: how large this stock's corrected earnings moves have been
historically, ranked against a frozen reference distribution. It is magnitude risk only.
Nothing here may state a probability or chance, a multiple of a market rate, an expected
move, or a direction (testing/test_customer_outputs.py enforces this).
"""
from scoring.model_b import HIGH_ALERT, ELEVATED, NORMAL, UNSCORED, MIN_PRIOR

PERCENTILE_NAME = "Breakwater Risk Percentile"

PERCENTILE_EXPLAINER = (
    "Breakwater Risk Percentile compares this stock's structural earnings-move risk with "
    "the frozen Model B reference distribution. It is a rank, not a probability.")

SIGNAL_EXPLAINER = (
    "The score is built from one input: the average absolute 3-session price move on this "
    "stock's past earnings reports, measured from the last close before each announcement "
    "and counting only reports with a verified announcement time. It ranks the size of a "
    "potential move, never its direction.")

TIER_BANDS = {
    HIGH_ALERT: "top decile of the reference distribution (90th percentile and above)",
    ELEVATED:   "80th–90th percentile of the reference distribution",
    NORMAL:     "below the 80th percentile of the reference distribution",
    UNSCORED:   f"fewer than {MIN_PRIOR} usable prior corrected earnings reactions",
}

# Unscored is a coverage statement, never a risk level. Every surface that shows an
# Unscored event, or leaves one out, says so in these words.
UNSCORED_NOTE = (
    "Unscored means there is not enough verified earnings history to evaluate this event "
    f"with Model B (fewer than {MIN_PRIOR} usable past reports) — it is not a low-risk rating.")

_HEADLINES = {
    HIGH_ALERT: "High Alert — top decile of historical earnings-move size",
    ELEVATED:   "Elevated — 80th to 90th percentile of historical earnings-move size",
    NORMAL:     "Normal — below the 80th percentile",
    UNSCORED:   "Unscored — not enough verified earnings history",
}

_ACTIONS = {
    HIGH_ALERT: ("This stock's past earnings reports have produced some of the largest moves "
                 "in the S&P 500 reference set. Review position size and any hedges before "
                 "the report; a large move in either direction is in character for it."),
    ELEVATED:   ("This stock's past earnings moves have been larger than most. Worth "
                 "checking position size before the report."),
    NORMAL:     "Nothing in this stock's earnings history marks it out for an unusually "
                "large move. Standard position management applies.",
    UNSCORED:   ("Breakwater does not score this event. That is not a statement that the "
                 "risk is low — there is not enough verified history to rank it."),
}


def build_recommendation(risk_tier, stock, earnings_date, n_prior, hist_mean_abs):
    """Headline / body / action for the PDF's summary block."""
    if risk_tier == UNSCORED:
        body = (f"{stock} reports on {earnings_date}. It has {int(n_prior)} usable prior "
                f"corrected earnings reaction{'s' if int(n_prior) != 1 else ''}; Model B "
                f"needs at least {MIN_PRIOR} to score an event.")
    else:
        body = (f"{stock} reports on {earnings_date}. Across its {int(n_prior)} usable past "
                f"earnings reports, the stock moved {hist_mean_abs * 100:.1f}% on average "
                f"over the 3 sessions after the announcement (either direction). That places "
                f"it in the {risk_tier} tier: {TIER_BANDS[risk_tier]}.")
    return {"headline": _HEADLINES[risk_tier], "body": body, "action": _ACTIONS[risk_tier]}
