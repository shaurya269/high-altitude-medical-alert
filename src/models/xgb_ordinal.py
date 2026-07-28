"""
XGBoost ordinal severity classifier -- the primary model (CLAUDE.md Section 6).

Approach: regression + threshold binning (CLAUDE.md's chosen default over a
cascade of binary "at least this severe?" classifiers -- simpler to
implement and tune, still ordinal-aware).

Why regression instead of plain multi-class XGBoost:
  Plain multi-class classification (`objective="multi:softmax"`) treats the
  5 tiers as unrelated categories -- to the loss function, predicting
  "Normal" when the truth is "HACE risk" is exactly as wrong as predicting
  "Mild AMS" when the truth is "Severe AMS". That throws away the entire
  point of ordinal data (see config.py's comment on SEVERITY_TIERS).
  Instead, we train XGBoost to REGRESS the severity index as a continuous
  target (0.0 to 4.0). A prediction of 3.7 for a true HACE-risk (4) case is
  a small loss; a prediction of 0.2 is a big loss -- the loss function
  itself now understands ordering. We then bin the continuous output back
  into discrete tiers using thresholds tuned on the validation set (not
  just naive rounding) so tier boundaries can be shifted to balance
  precision/recall per class, particularly to reduce under-triage
  (predicting too mild -- see metrics.py) of the rare, dangerous classes.

Class imbalance handling: sample weights inversely proportional to class
frequency, computed on the TRAIN split only (never peek at val/test
distributions) -- CLAUDE.md requires this, not just accepting a model that
always predicts "Normal".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODELS_DIR, N_TIERS, PROCESSED_DIR, RANDOM_SEED, SEVERITY_TIERS
from src.data.feature_engineering import FEATURE_COLUMNS
from src.models.metrics import evaluate, print_report

MODEL_PATH = MODELS_DIR / "xgb_ordinal.json"
THRESHOLDS_PATH = MODELS_DIR / "xgb_thresholds.json"


def bin_predictions(continuous_preds: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """
    Convert continuous severity-index predictions into discrete tiers using
    tuned cut points, e.g. thresholds=[0.5, 1.5, 2.5, 3.5] means:
      pred < 0.5           -> tier 0 (Normal)
      0.5 <= pred < 1.5     -> tier 1 (Mild AMS)
      ... etc.

    np.digitize does exactly this bucketing; N_TIERS-1 thresholds always
    produce N_TIERS bins, which is why we assert the length below -- a
    mismatched threshold count would silently produce the wrong number of
    output classes.
    """
    assert len(thresholds) == N_TIERS - 1, (
        f"Expected {N_TIERS - 1} thresholds for {N_TIERS} tiers, got {len(thresholds)}"
    )
    # np.digitize returns, for each value, how many thresholds it's >= to --
    # e.g. thresholds=[0.5,1.5,2.5,3.5] and pred=2.1 -> 2 (past 0.5 and 1.5,
    # not past 2.5) -- which is exactly the tier index we want. .clip is a
    # defensive guard against out-of-range predictions (e.g. a stray
    # negative or >4.0 continuous output) rather than something expected
    # to trigger often.
    return np.digitize(continuous_preds, thresholds).clip(0, N_TIERS - 1)



# How much more we penalize an under-triage error (predicted tier < true
# tier) than an equally-sized over-triage error, when tuning thresholds.
#
# Day 13 testing found that plain MATE-minimization (weight=1.0) left 45%
# of true HAPE/HACE-risk rows predicted BELOW the hysteresis gate's
# alert-eligible tier (Severe AMS) -- the alert would never even be
# attempted for nearly half the most dangerous cases. Weighting under-
# triage more heavily was tried to fix this (weight=3.0, then a gentler
# weight=1.5) -- both reduced that 45% figure substantially, BUT both also
# introduced a new, unacceptable failure: a genuinely Normal-severity demo
# scenario started reliably triggering a real Telegram alert (10/10 random
# seeds at weight=3.0; the first 3 tested at weight=1.5 also fired,
# stopped there once the pattern was clearly systematic rather than
# noise). The mechanism: any asymmetric bias toward predicting HIGHER
# severity pushes ordinary sensor noise on a Normal trajectory across the
# Severe-AMS threshold often enough to sustain hysteresis's consecutive-
# reading requirement -- this isn't a tunable edge case, it's a direct
# consequence of biasing the SAME thresholds that gate real (Normal-vs-
# elevated) classifications, not just the dangerous tiers specifically.
#
# Reverted to weight=1.0 (plain, symmetric MATE -- identical to the
# pre-Day-13 behavior) as the safe, well-tested default. The underlying
# under-triage finding is real and worth addressing, but not via this
# mechanism -- see the project owner's decision log / README for the
# chosen path forward (e.g. lowering HYSTERESIS_ALERT_TIER instead, which
# changes what severity the GATE watches for without distorting the
# model's own tier boundaries).
UNDER_TRIAGE_PENALTY_WEIGHT = 1.0


def _asymmetric_tier_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Like mean absolute tier error, but an under-triage error (pred < true)
    counts UNDER_TRIAGE_PENALTY_WEIGHT times as much as an over-triage
    error of the same size. This is the objective tune_thresholds()
    actually searches over -- see UNDER_TRIAGE_PENALTY_WEIGHT's comment
    for why the two directions aren't treated symmetrically.
    """
    error = y_pred - y_true
    weights = np.where(error < 0, UNDER_TRIAGE_PENALTY_WEIGHT, 1.0)
    return float(np.mean(np.abs(error) * weights))


def tune_thresholds(y_val_true: np.ndarray, val_continuous_preds: np.ndarray) -> list[float]:
    """
    Search for threshold cut points that minimize the ASYMMETRIC tier
    error (see _asymmetric_tier_error) on the validation set, rather than
    naive rounding OR plain (symmetric) mean absolute tier error.

    Why not just round(), and why not plain MATE either? Rounding
    implicitly assumes each tier's continuous predictions are symmetric
    around its center, which isn't guaranteed -- e.g. if the model
    systematically under-predicts severity for the rare HACE class (a
    real risk given class imbalance), shifting that tier's lower
    threshold down catches more true positives. Plain MATE fixes the
    "symmetric around center" assumption but still treats under- and
    over-triage as equally costly, which Day 13 testing showed leaves a
    dangerous gap (see UNDER_TRIAGE_PENALTY_WEIGHT). This is a small grid
    search per threshold, optimizing directly for the asymmetric metric
    that actually reflects what this system should be biased toward.
    """
    best_thresholds = [0.5, 1.5, 2.5, 3.5]  # naive-rounding starting point
    best_cost = _asymmetric_tier_error(
        bin_predictions(val_continuous_preds, best_thresholds), y_val_true
    )

    # Coordinate-descent-style search: adjust one threshold at a time over a
    # local grid, keep it if it improves the asymmetric cost. Cheap (few
    # hundred evaluations) and avoids a full N-dimensional grid search over
    # all 4 thresholds jointly, which isn't necessary here.
    candidate_offsets = np.arange(-0.4, 0.41, 0.05)
    for _ in range(3):  # a few passes over all thresholds tends to converge
        for i in range(len(best_thresholds)):
            for offset in candidate_offsets:
                trial = list(best_thresholds)  # copy, not mutate best_thresholds -- this is a candidate to test, not yet a commitment
                trial[i] = trial[i] + offset
                # Keep thresholds monotonically increasing -- otherwise
                # np.digitize's bucket ordering breaks down. Checks the
                # neighbor on each side (skipping the check entirely at the
                # first/last index, since there's no neighbor there) with a
                # small 0.05 buffer so thresholds can't collapse to be
                # equal or crossed.
                if not (
                    (i == 0 or trial[i] > trial[i - 1] + 0.05)
                    and (i == len(trial) - 1 or trial[i] < trial[i + 1] - 0.05)
                ):
                    continue
                preds = bin_predictions(val_continuous_preds, trial)
                cost = _asymmetric_tier_error(preds, y_val_true)
                if cost < best_cost:
                    best_cost = cost
                    best_thresholds = trial

    return best_thresholds


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> tuple[xgb.XGBRegressor, list[float]]:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["severity_index"].astype(float)  # cast int tier labels to float -- this is the continuous regression target, per the module docstring
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["severity_index"].astype(float)

    # Sample weights inversely proportional to class frequency, computed
    # from TRAIN labels only -- this is the "class weighting" CLAUDE.md
    # requires so the model doesn't just learn to always predict the
    # dominant "Normal" class. compute_sample_weight('balanced', ...) gives
    # each class weight = n_samples / (n_classes * class_count), so rare
    # classes (HAPE/HACE risk) get proportionally larger weight per example.
    sample_weight = compute_sample_weight("balanced", y_train.round().astype(int))

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,  # max number of boosting trees to build (300 default); early_stopping_rounds below can halt before reaching this
        max_depth=max_depth,  # max depth per tree (5 default) -- caps how many feature interactions one tree can encode, guards against overfitting a modestly-sized dataset
        learning_rate=learning_rate,  # shrinkage applied to each tree's contribution (0.05 default) -- smaller values need more trees but generalize better
        subsample=0.8,  # fraction of TRAINING ROWS sampled (without replacement) per tree -- adds randomness across trees to reduce overfitting/variance
        colsample_bytree=0.8,  # fraction of FEATURE COLUMNS sampled per tree -- same variance-reduction idea as subsample, applied to columns instead of rows
        objective="reg:squarederror",  # regression objective (minimize squared error) -- this is what makes the model predict a continuous severity score instead of a class, per the module docstring
        random_state=RANDOM_SEED,  # fixes the row/column subsampling randomness so training is reproducible
        early_stopping_rounds=20,  # stop boosting if val eval_metric hasn't improved for 20 rounds -- prevents overfitting past the point val performance plateaus
        eval_metric="mae",  # metric early stopping/eval_set actually watches (mean absolute error on the continuous score, not the binned tiers)
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,  # per-row weights from compute_sample_weight above, so rare severity classes count more toward the loss
        eval_set=[(X_val, y_val)],  # validation fold XGBoost checks each round against, purely to drive early_stopping_rounds -- never used to fit weights
        verbose=False,  # suppress XGBoost's per-round eval printout; we print our own summary after training instead
    )

    val_continuous_preds = model.predict(X_val)  # continuous severity scores (e.g. 2.3), not yet binned to a discrete tier
    thresholds = tune_thresholds(y_val.to_numpy(), val_continuous_preds)

    return model, thresholds


def save(model: xgb.XGBRegressor, thresholds: list[float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump({"thresholds": thresholds, "feature_columns": FEATURE_COLUMNS}, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved thresholds -> {THRESHOLDS_PATH}")


def load() -> tuple[xgb.XGBRegressor, list[float]]:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    with open(THRESHOLDS_PATH) as f:
        meta = json.load(f)
    return model, meta["thresholds"]


def predict(model: xgb.XGBRegressor, thresholds: list[float], df: pd.DataFrame) -> np.ndarray:
    continuous = model.predict(df[FEATURE_COLUMNS])
    return bin_predictions(continuous, thresholds)


if __name__ == "__main__":
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run `python -m src.data.feature_engineering` first."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("Training XGBoost ordinal regressor...")
    model, thresholds = train(train_df, val_df)
    print(f"Tuned thresholds: {[round(t, 3) for t in thresholds]}")

    save(model, thresholds)

    y_pred = predict(model, thresholds, test_df)
    y_true = test_df["severity_index"].to_numpy()
    result = evaluate(y_true, y_pred, model_name="XGBoost Ordinal (test set)")
    print_report(result)

    # Feature importance -- useful both for the owner's learning goal
    # (CLAUDE.md: "wants things explained") and as a sanity check that the
    # model is leaning on clinically sensible features (spo2_delta, trend)
    # rather than something spurious.
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
        ascending=False
    )
    print("Feature importances:")
    print(importances.round(4).to_string())
