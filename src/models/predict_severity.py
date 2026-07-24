"""
Model selection + the single inference entry point every downstream stage
(hysteresis gate, LLM interpreter, Streamlit dashboard) calls.

CLAUDE.md Section 6: "keep whichever performs better in practice --
XGBoost is favored default for small datasets like this." This module runs
both trained models against the SAME held-out test set through the SAME
metrics.py, picks the winner by mean_abs_tier_error (the ordinal-aware
metric that best reflects clinical cost -- see metrics.py's reasoning), and
writes a small model_comparison.json artifact recording the decision so
it's traceable later (not just a comment in code that can drift from what
was actually run).

predict_severity() is intentionally the ONLY function the rest of the
pipeline (Stage 4 in the data flow diagram) needs to import -- it hides
which underlying model is active so swapping the winner later (e.g. if
future work improves the LSTM) doesn't require touching alerts/hysteresis,
llm/, or app/ code at all.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, SEVERITY_TIERS
from src.data.feature_engineering import FEATURE_COLUMNS
from src.models import lstm_model, rule_baseline, xgb_ordinal
from src.models.metrics import compare_models, evaluate, print_report

COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
SELECTED_MODEL_PATH = MODELS_DIR / "selected_model.json"


def run_comparison() -> dict:
    """
    Train/evaluate/compare rule baseline, XGBoost, and LSTM on the same
    test set. Returns the comparison dict and writes it to disk.

    Assumes xgb_ordinal.py and lstm_model.py have already been run once
    (their artifacts exist in MODELS_DIR) -- this function LOADS the saved
    models rather than retraining, so re-running comparison after tweaking
    e.g. hysteresis config doesn't waste an XGBoost/LSTM training pass.
    """
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_true = test_df["severity_index"].to_numpy()

    results = []

    baseline_pred = rule_baseline.predict(test_df)
    results.append(evaluate(y_true, baseline_pred, model_name="Rule-Based Baseline"))

    xgb_model_obj, thresholds = xgb_ordinal.load()
    xgb_pred = xgb_ordinal.predict(xgb_model_obj, thresholds, test_df)
    results.append(evaluate(y_true, xgb_pred, model_name="XGBoost Ordinal"))

    lstm_model_obj, mean, std = lstm_model.load()
    lstm_pred, lstm_true = lstm_model.predict(lstm_model_obj, mean, std, test_df)
    # LSTM's y_true comes back from build_windows (one label per WINDOW, not
    # per row) so it's a different length than y_true above -- evaluate
    # against lstm_true, not the row-level y_true, to compare apples to
    # apples with what the LSTM was actually asked to predict.
    results.append(evaluate(lstm_true, lstm_pred, model_name="LSTM"))

    for r in results:
        print_report(r)

    comparison_table = compare_models(results)
    print("\nSide-by-side comparison:")
    print(comparison_table.to_string(index=False))

    # Selection rule: lowest mean_abs_tier_error among the two REAL models
    # (baseline is the sanity floor, never eligible to be "selected" as the
    # production model even if it happened to win on a metric by chance --
    # CLAUDE.md's baseline is a check on the others, not a candidate itself).
    candidates = [r for r in results if r["model_name"] in ("XGBoost Ordinal", "LSTM")]
    winner = min(candidates, key=lambda r: r["mean_abs_tier_error"])

    baseline_result = next(r for r in results if r["model_name"] == "Rule-Based Baseline")
    beats_baseline = winner["mean_abs_tier_error"] < baseline_result["mean_abs_tier_error"]
    if not beats_baseline:
        print(
            "\n*** WARNING: winning model does NOT beat the rule-based baseline on "
            "mean_abs_tier_error. Per CLAUDE.md, this signals a real problem with "
            "features or labels that should be investigated before trusting this model. ***"
        )

    comparison_record = {
        "test_set_size": len(test_df),
        "results_summary": comparison_table.to_dict(orient="records"),
        "winner": winner["model_name"],
        "winner_beats_baseline": beats_baseline,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison_record, f, indent=2)
    with open(SELECTED_MODEL_PATH, "w") as f:
        json.dump({"selected_model": winner["model_name"]}, f, indent=2)

    print(f"\nSelected model: {winner['model_name']}  (beats baseline: {beats_baseline})")
    print(f"Saved comparison -> {COMPARISON_PATH}")
    print(f"Saved selection -> {SELECTED_MODEL_PATH}")

    return comparison_record


# ---------------------------------------------------------------------------
# Live inference entry point
# ---------------------------------------------------------------------------

_cached_selection: str | None = None
_cached_xgb = None
_cached_lstm = None


def _get_selected_model_name() -> str:
    global _cached_selection
    if _cached_selection is None:
        if not SELECTED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{SELECTED_MODEL_PATH} not found -- run "
                "`python -m src.models.predict_severity` (run_comparison) first "
                "to train/compare/select a model."
            )
        with open(SELECTED_MODEL_PATH) as f:
            _cached_selection = json.load(f)["selected_model"]
    return _cached_selection


def predict_severity(buffer_df: pd.DataFrame) -> dict:
    """
    THE inference entry point for Stage 4 of the live pipeline (data flow
    diagram). Takes a single subject's rolling buffer -- already run
    through src.data.feature_engineering.engineer_features -- and returns
    the CURRENT severity classification for the most recent reading.

    Returns a dict (not a bare int) because the hysteresis gate, LLM
    prompt, and dashboard all want more than just the tier index: the tier
    NAME (for display/prompts), and a confidence proxy. This is the
    "Output: 5-tier severity score" contract shown in the system
    architecture diagram's ML Severity Classifier node.

    Always operates on the LATEST row of buffer_df -- earlier rows exist
    only to give trend features (spo2_trend_5min etc.) and, for the LSTM
    path, the raw window, something to compute over.
    """
    if buffer_df.empty:
        raise ValueError("predict_severity() called with an empty buffer")

    selected = _get_selected_model_name()
    latest_row = buffer_df.iloc[[-1]]

    if selected == "XGBoost Ordinal":
        global _cached_xgb
        if _cached_xgb is None:
            _cached_xgb = xgb_ordinal.load()
        model, thresholds = _cached_xgb
        continuous = float(model.predict(latest_row[FEATURE_COLUMNS])[0])
        tier_index = int(xgb_ordinal.bin_predictions(np.array([continuous]), thresholds)[0])
        # A simple, honest confidence proxy: how far the continuous
        # prediction sits from the NEAREST bin edge, normalized to [0,1].
        # Not a calibrated probability -- XGBoost regression doesn't
        # natively produce one -- but a useful "how borderline is this"
        # signal for the LLM prompt and dashboard to surface.
        edges = [0.0] + list(thresholds) + [4.0]
        lo, hi = edges[tier_index], edges[tier_index + 1]
        span = max(hi - lo, 1e-6)
        center = (lo + hi) / 2
        confidence = float(1.0 - min(abs(continuous - center) / (span / 2), 1.0) * 0.5)

    elif selected == "LSTM":
        global _cached_lstm
        if _cached_lstm is None:
            _cached_lstm = lstm_model.load()
        model, mean, std = _cached_lstm
        if len(buffer_df) < lstm_model.WINDOW_SAMPLES:
            raise ValueError(
                f"LSTM model needs at least {lstm_model.WINDOW_SAMPLES} buffered readings "
                f"(60s at 1Hz), got {len(buffer_df)}. Not enough history yet -- caller "
                "should fall back to the rule baseline until the buffer fills."
            )
        window = buffer_df[lstm_model.LSTM_RAW_COLUMNS].to_numpy(dtype=np.float32)[
            -lstm_model.WINDOW_SAMPLES :
        ]
        window_norm = (window - mean) / std
        import torch

        with torch.no_grad():
            logits = model(torch.from_numpy(window_norm).float().unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        tier_index = int(probs.argmax())
        confidence = float(probs[tier_index])

    else:
        raise ValueError(f"Unknown selected model: {selected!r}")

    return {
        "severity_index": tier_index,
        "severity_label": SEVERITY_TIERS[tier_index],
        "confidence": round(confidence, 3),
        "model_used": selected,
    }


if __name__ == "__main__":
    run_comparison()
