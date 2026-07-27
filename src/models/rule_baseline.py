"""
Rule-based clinical threshold classifier -- the sanity-floor baseline.

CLAUDE.md is explicit: build this FIRST, always, before any ML model. The
reasoning: if XGBoost or the LSTM can't clearly beat a handful of if/else
rules written from docs/lls_mapping.md, that's a signal something is wrong
with the features or labels, not evidence the ML approach doesn't work. This
mirrors CineMind's own popularity-baseline pattern (a sister project in this
workspace) -- always have a dumb-but-principled floor to beat.

This is deliberately simple: no learned weights, no interactions beyond what
a clinician's mental checklist would use. It reads spo2_delta (deviation
from altitude-expected SpO2) and hr elevation as the primary signals, with
the negative-trend check specifically for catching HAPE/HACE risk per
docs/lls_mapping.md's note that trend matters as much as absolute value
there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HR_NORMAL_RANGE, SEVERITY_INDEX


def classify_row(spo2_delta: float, hr: float, spo2_trend_5min: float) -> int:
    """
    Apply the docs/lls_mapping.md thresholds to a single reading's features.

    Returns an integer severity index (0=Normal .. 4=HACE risk).

    hr_elevation is computed relative to the population-normal band's
    midpoint rather than a per-subject resting HR, because -- unlike the
    ML models, which get individual baseline patterns for free from
    training data -- a rule-based baseline has no way to know a specific
    subject's true resting HR at inference time. Using the population
    normal midpoint is the fair, information-equivalent comparison.
    """
    hr_normal_mid = sum(HR_NORMAL_RANGE) / 2
    hr_elevation = hr - hr_normal_mid

    # HAPE/HACE risk: severe SpO2 deviation OR steep negative trend even
    # before the absolute deviation looks extreme (docs/lls_mapping.md:
    # "early HAPE can desaturate fast" -- trend precedes magnitude).
    steep_negative_trend = spo2_trend_5min < -0.15  # % SpO2 lost per second, sustained

    # Cascading if/elif-style checks from most to least severe: each branch
    # only runs if every stricter branch above it already failed to match,
    # so the thresholds are naturally nested (e.g. reaching the "Severe AMS"
    # check already means the row failed the harder HAPE/HACE check above).
    if spo2_delta >= 16 or (spo2_delta >= 10 and steep_negative_trend):
        # Distinguish HACE from HAPE using the more extreme band + higher HR
        # elevation, per docs/lls_mapping.md's tier table -- HACE is modeled
        # as the more decompensated end of this same danger zone.
        if spo2_delta >= 20 and hr_elevation >= 35:
            return SEVERITY_INDEX["HACE risk"]
        return SEVERITY_INDEX["HAPE risk"]

    if spo2_delta >= 8 or hr_elevation >= 20:
        return SEVERITY_INDEX["Severe AMS"]

    if spo2_delta >= 4 or hr_elevation >= 10:
        return SEVERITY_INDEX["Mild AMS"]

    return SEVERITY_INDEX["Normal"]  # neither SpO2 nor HR deviated enough to trip any tier above


def predict(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized-ish wrapper: apply classify_row across a DataFrame that has
    already been through src.data.feature_engineering.engineer_features
    (needs spo2_delta, hr, spo2_trend_5min columns).
    """
    required = {"spo2_delta", "hr", "spo2_trend_5min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"rule_baseline.predict() requires columns {required}, missing {missing}. "
            "Run src.data.feature_engineering.engineer_features first."
        )
    return df.apply(
        lambda row: classify_row(row["spo2_delta"], row["hr"], row["spo2_trend_5min"]),
        axis=1,
    ).to_numpy()  # .to_numpy() so callers get a plain int array, matching what evaluate() expects from every model (not a pandas Series)


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    from src.models.metrics import evaluate, print_report

    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} not found -- run `python -m src.data.feature_engineering` first."
        )
    test_df = pd.read_csv(test_path)

    y_pred = predict(test_df)
    y_true = test_df["severity_index"].to_numpy()

    result = evaluate(y_true, y_pred, model_name="Rule-Based Baseline (test set)")
    print_report(result)
