"""
Shared evaluation utilities for every severity classifier (rule baseline,
XGBoost, LSTM). Living in one module guarantees all three are scored with
literally the same code -- if the baseline and XGBoost each computed their
own precision/recall, a subtle difference in averaging method could make a
"fair" model comparison meaningless without anyone noticing.

Ordinal-aware metrics matter here specifically because plain accuracy
treats "predicted Normal, actual HACE risk" (very bad) the same as
"predicted Mild AMS, actual Severe AMS" (still wrong, much less bad). We
report both standard classification metrics AND mean absolute tier error
(how many tiers off, on average) so a model that's "close but wrong" is
visibly rewarded over one that's wildly wrong -- see CLAUDE.md Section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import SEVERITY_TIERS


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> dict:
    """
    Compute the full metrics suite for a set of ordinal predictions.

    y_true / y_pred are integer severity indices (0=Normal .. 4=HACE risk),
    NOT string labels -- keeps this function usable by every model
    regardless of whether it natively predicts strings, ints, or a
    continuous score that's already been threshold-binned by the caller.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # macro averaging: each class contributes equally to the score
    # regardless of how common it is. This matters because Normal dominates
    # the dataset (CLAUDE.md's expected class imbalance) -- a model that
    # nails Normal and ignores HACE risk would still get a great
    # micro/weighted F1, which would hide exactly the failure mode we care
    # most about (missing the dangerous, rare cases).
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Mean Absolute Tier Error: average |predicted_tier - true_tier|. This
    # is the ordinal-specific metric -- 0 means perfect, 1 means predictions
    # are typically one tier off, etc. Directly answers "how clinically
    # costly are this model's mistakes on average," which accuracy alone
    # can't.
    mate = float(np.mean(np.abs(y_pred - y_true)))

    # Under-triage rate: fraction of predictions that UNDERESTIMATE severity
    # (predicted tier < true tier). This is the dangerous direction of error
    # for a medical alert system -- predicting "Normal" for an actual HACE
    # case could mean a missed alert. Over-triage (false alarms) is
    # annoying; under-triage is the failure mode the hysteresis gate and
    # this whole system exist to avoid, so we track it explicitly rather
    # than letting it hide inside an aggregate accuracy number.
    under_triage_rate = float(np.mean(y_pred < y_true))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SEVERITY_TIERS))))  # labels= pins the row/col order to all 5 tiers even if a rare class is absent from this particular y_true/y_pred batch
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(SEVERITY_TIERS))),  # same reasoning as confusion_matrix's labels= above -- keeps per-class rows present/aligned even for an unseen-in-this-batch tier
        target_names=SEVERITY_TIERS,  # display tier names instead of raw integer indices in the report
        zero_division=0,  # a class with zero predicted/true samples would otherwise raise a warning and return NaN for precision/recall -- report 0 instead
        output_dict=True,  # return a dict (not a printed string) so callers/print_report can format it themselves
    )

    return {
        "model_name": model_name,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "mean_abs_tier_error": mate,
        "under_triage_rate": under_triage_rate,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_report(result: dict) -> None:
    """Human-readable console summary -- used identically by all three model scripts."""
    print(f"\n{'=' * 60}")
    print(f"  {result['model_name']}")
    print(f"{'=' * 60}")
    print(f"  Precision (macro):       {result['precision_macro']:.4f}")
    print(f"  Recall (macro):          {result['recall_macro']:.4f}")
    print(f"  F1 (macro):              {result['f1_macro']:.4f}")
    print(f"  Mean Abs Tier Error:     {result['mean_abs_tier_error']:.4f}  (0 = perfect)")
    print(f"  Under-triage rate:       {result['under_triage_rate']:.4f}  (predicted too mild)")
    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    cm_df = pd.DataFrame(result["confusion_matrix"], index=SEVERITY_TIERS, columns=SEVERITY_TIERS)
    print(cm_df.to_string())
    print()


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Side-by-side comparison table across the baseline/XGBoost/LSTM results."""
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["model_name"],
                "precision_macro": round(r["precision_macro"], 4),
                "recall_macro": round(r["recall_macro"], 4),
                "f1_macro": round(r["f1_macro"], 4),
                "mean_abs_tier_error": round(r["mean_abs_tier_error"], 4),
                "under_triage_rate": round(r["under_triage_rate"], 4),
            }
        )
    return pd.DataFrame(rows)
