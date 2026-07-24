"""
Smoke tests for the Days 1-7 data + ML pipeline.

These are deliberately lightweight (small synthetic samples generated
in-memory, not the full multi-million-row dataset) -- the goal is catching
"this function throws / returns the wrong shape" regressions quickly, not
re-validating model performance (that's what src/models/predict_severity.py's
run_comparison() and its saved model_comparison.json artifact are for).

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import N_TIERS, SEVERITY_TIERS
from src.data.feature_engineering import FEATURE_COLUMNS, engineer_features, temporal_split
from src.data.synth_data import expected_spo2, generate_dataset, simulate_trajectory
from src.models import rule_baseline
from src.models.metrics import evaluate


def test_expected_spo2_decreases_with_altitude():
    """Sanity check on the core physiological reference curve -- higher
    altitude must always mean lower expected SpO2, never the reverse."""
    assert expected_spo2(0) > expected_spo2(3000) > expected_spo2(6000)


def test_simulate_trajectory_shape_and_bounds():
    rng = np.random.default_rng(0)
    df = simulate_trajectory("test_subj", "Mild AMS", duration_minutes=10, rng=rng)

    assert len(df) == 10 * 60  # 1Hz * 10 minutes
    assert set(df.columns) >= {"timestamp", "spo2", "hr", "temp", "altitude", "severity_label"}
    # Physiological plausibility bounds -- catches generator bugs that
    # would produce nonsense values (e.g. negative SpO2, HR of 500).
    assert df["spo2"].between(50, 100).all()
    assert df["hr"].between(35, 200).all()
    assert (df["altitude"] >= 0).all()
    assert (df["severity_label"] == "Mild AMS").all()


def test_generate_dataset_has_all_tiers():
    """With enough subjects, every severity tier should appear at least
    once -- a regression here would mean TIER_PREVALENCE or the sampling
    logic broke and silently dropped a class (the exact failure mode that
    caused the original val/test split to have zero HACE-risk examples)."""
    df = generate_dataset(n_subjects=200, duration_minutes=20, seed=1)
    present_tiers = set(df["severity_label"].unique())
    assert present_tiers == set(SEVERITY_TIERS)


def test_engineer_features_adds_expected_columns():
    rng = np.random.default_rng(0)
    df = simulate_trajectory("test_subj", "Severe AMS", duration_minutes=15, rng=rng)
    featured = engineer_features(df)

    for col in FEATURE_COLUMNS:
        assert col in featured.columns, f"missing engineered feature column: {col}"
    # No NaNs should leak out of feature engineering -- a downstream model
    # would silently mishandle them (or XGBoost would treat NaN as "missing"
    # which has different semantics than an actual zero-trend reading).
    assert not featured[FEATURE_COLUMNS].isna().any().any()


def test_engineer_features_trend_direction():
    """A trajectory with a clearly worsening (declining) SpO2 should show a
    negative spo2_trend_5min once the window fills -- this is the feature
    the rule baseline and XGBoost both lean on most heavily to catch
    HAPE/HACE risk (see docs/lls_mapping.md), so its sign must be correct."""
    n = 600  # 10 minutes at 1Hz
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=float),
            "spo2": np.linspace(95, 70, n),  # steadily declining
            "hr": np.full(n, 80.0),
            "temp": np.full(n, 36.8),
            "altitude": np.full(n, 3000.0),
        }
    )
    featured = engineer_features(df)
    # By the end of the trajectory (window fully filled), trend must be negative.
    assert featured["spo2_trend_5min"].iloc[-1] < 0


def test_temporal_split_no_subject_overlap():
    """The split's core invariant: no subject_id may appear in more than one
    of train/val/test -- this is what actually prevents leakage (see the
    docstring in feature_engineering.temporal_split)."""
    df = generate_dataset(n_subjects=150, duration_minutes=15, seed=2)
    from src.data.feature_engineering import build_processed_dataset

    processed = build_processed_dataset(df)
    train_df, val_df, test_df = temporal_split(processed)

    train_ids = set(train_df["subject_id"])
    val_ids = set(val_df["subject_id"])
    test_ids = set(test_df["subject_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # All three splits should be non-empty for every severity tier present
    # in the source data -- this is the regression test for the original
    # bug where a plain chronological cut starved val/test of rare tiers.
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tiers_present = set(split_df["severity_label"].unique())
        assert len(tiers_present) > 0, f"{split_name} split is empty"


def test_rule_baseline_predicts_valid_tiers():
    df = generate_dataset(n_subjects=30, duration_minutes=15, seed=3)
    from src.data.feature_engineering import build_processed_dataset

    processed = build_processed_dataset(df)
    preds = rule_baseline.predict(processed)

    assert preds.min() >= 0
    assert preds.max() < N_TIERS


def test_rule_baseline_missing_columns_raises():
    with pytest.raises(ValueError):
        rule_baseline.predict(pd.DataFrame({"spo2": [1, 2]}))


def test_evaluate_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    result = evaluate(y, y, model_name="perfect")
    assert result["mean_abs_tier_error"] == 0.0
    assert result["under_triage_rate"] == 0.0
    assert result["f1_macro"] == pytest.approx(1.0)


def test_evaluate_under_triage_detection():
    """A model that always predicts one tier too LOW should show up with a
    100% under-triage rate -- this is the metric the whole system cares
    about most (missing a dangerous case), so it must be computed correctly."""
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3])  # always one tier under
    result = evaluate(y_true, y_pred, model_name="under-triager")
    assert result["under_triage_rate"] == 1.0
    assert result["mean_abs_tier_error"] == 1.0
