"""
Feature engineering + temporal train/val/test split.

This is Stage 3 of the data flow diagram (Architecture_Diagrams/03_data_flow_diagram.html):
turns a raw per-subject vitals stream into the trend/derived features the ML
models actually train on. The SAME functions here are reused at inference
time (Stage 2-3 of the live pipeline) so training and serving can never
silently drift apart into two different feature definitions.

Why a temporal split, never a random shuffle (CLAUDE.md, repeated because
it's the single easiest mistake to make with time-series data): if we
randomly shuffled rows into train/test, a test-set row at t=500 could sit
right next to a train-set row at t=499 from the SAME trajectory. The model
would effectively be tested on data it has already seen the immediate
neighbors of -- performance would look great and be a lie. Instead we split
whole subjects/trajectories by time-of-generation order, so validation/test
genuinely represent "sessions the model has never seen any part of."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    SEVERITY_INDEX,
    SYNTHETIC_DIR,
    TEST_FRACTION,
    TRAIN_FRACTION,
    TREND_WINDOW_MINUTES,
    VAL_FRACTION,
)
from src.data.synth_data import expected_spo2, synthesize_temperature

# Trend window expressed in samples (rows), not minutes, since all the
# rolling-window pandas calls need an integer window size. At 1Hz sampling
# this is just minutes*60, but computing it from SAMPLE_RATE_HZ (rather than
# hardcoding *60) means it stays correct if the sample rate config changes.
from src.config import SAMPLE_RATE_HZ

TREND_WINDOW_SAMPLES = TREND_WINDOW_MINUTES * 60 * SAMPLE_RATE_HZ


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a tidy per-subject vitals DataFrame.

    Must be called per-subject (or with subject boundaries respected) --
    rolling windows must never blend two different subjects' readings
    together. This function assumes `df` is a SINGLE subject's trajectory,
    sorted by timestamp; `build_processed_dataset` below handles the
    per-subject grouping so callers don't have to remember that rule.

    Features added (matching Stage 3 of the data flow diagram):
      - spo2_trend_5min:  slope of SpO2 over the trailing 5-minute window
                          (negative = worsening) -- HAPE/HACE risk shows up
                          as a steep negative trend even before the absolute
                          value looks alarming (docs/lls_mapping.md)
      - hr_trend_5min:    same idea for heart rate
      - ascent_rate:      meters/minute of altitude change (rapid ascent is
                          itself a risk factor independent of current vitals)
      - expected_spo2:    the altitude-adjusted reference curve (see
                          synth_data.expected_spo2)
      - spo2_delta:       actual spo2 minus expected_spo2 -- the deviation
                          that actually matters clinically, not raw SpO2
                          (see docs/lls_mapping.md "why deltas, not raw")
      - time_at_altitude: minutes since altitude first exceeded the AMS
                          risk-onset threshold (config.ALTITUDE_RISK_ONSET_M)
                          -- symptoms build up the LONGER someone stays high,
                          not just based on how high they currently are
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Trend = simple linear slope (last - first) / window_span rather than a
    # full least-squares fit -- cheaper to compute and good enough for a
    # monotonic-ish physiological trend; np.polyfit per row would be
    # needlessly expensive at 1Hz over long recordings.
    #
    # Vectorized via diff() against a shift() of the SAME distance, rather
    # than a Python-level rolling().apply(_slope) callback -- the latter is
    # O(n * window_size) with per-window Python function-call overhead,
    # which is fine for a quick demo but becomes a real bottleneck once the
    # dataset has hundreds of subjects at 1Hz over 90-minute trajectories
    # (millions of rows). diff()/shift() do the identical "last - first over
    # a fixed lag" computation in optimized pandas C code instead.
    #
    # min_periods behavior is replicated by computing the slope over
    # whatever lag is actually available near the start of a trajectory
    # (growing window) rather than requiring the full TREND_WINDOW_SAMPLES
    # -- same "use the partial window you have so far" reasoning as before,
    # since a real device can't look back further than it's been running.
    row_idx = pd.Series(np.arange(len(df)), index=df.index)  # 0, 1, 2, ... one integer per row -- "how far into the trajectory is this row"
    effective_lag = row_idx.clip(upper=TREND_WINDOW_SAMPLES - 1)  # cap each row's lag at the full window size, so early rows use a SMALLER lag

    def _vectorized_slope(series: pd.Series) -> pd.Series:
        first_full = series.shift(TREND_WINDOW_SAMPLES - 1)  # this row's value, shifted DOWN by the full window size -> gives "the value N rows ago" aligned to the current row
        span_full = float(TREND_WINDOW_SAMPLES - 1)  # the time gap (in samples) between "N rows ago" and "now"
        slope_full = (series - first_full) / span_full  # (now - then) / elapsed_samples = average change per sample = the slope

        # For the still-filling-up lead-in rows (fewer than
        # TREND_WINDOW_SAMPLES samples seen so far), fall back to a slope
        # against the very first sample of the trajectory (index 0) instead
        # -- growing-window behavior matching the old min_periods=2 case.
        first_ever = series.iloc[0]  # the very first value in this subject's whole trajectory
        span_growing = effective_lag.replace(0, np.nan)  # avoid /0 on row 0 (row 0 has lag 0, which would be a division by zero)
        slope_growing = (series - first_ever) / span_growing  # slope computed against the trajectory's start, for rows too early to have a full window yet

        # .where(condition, other): keep slope_full wherever the condition
        # is True (row is far enough in for a full window), otherwise use
        # slope_growing (row is still in the early "not enough history yet"
        # period) -- this stitches the two calculations into one column.
        slope = slope_full.where(row_idx >= TREND_WINDOW_SAMPLES - 1, slope_growing)
        return slope.fillna(0.0)  # any remaining NaN (e.g. row 0's slope_growing, which had span_growing=NaN) becomes a neutral 0

    df["spo2_trend_5min"] = _vectorized_slope(df["spo2"])
    df["hr_trend_5min"] = _vectorized_slope(df["hr"])

    # Ascent rate: meters/minute, using a 1-minute-back diff (60 samples at
    # 1Hz). fillna(0) for the first minute of any trajectory, where there's
    # no prior sample to diff against yet.
    altitude_diff = df["altitude"].diff(periods=60 * SAMPLE_RATE_HZ)  # this row's altitude minus the altitude 60 samples (1 minute) earlier
    df["ascent_rate"] = (altitude_diff / TREND_WINDOW_MINUTES).fillna(0.0)  # approx m/min

    df["expected_spo2"] = expected_spo2(df["altitude"])  # the altitude-adjusted reference curve for every row
    df["spo2_delta"] = df["expected_spo2"] - df["spo2"]  # positive = worse than expected

    from src.config import ALTITUDE_RISK_ONSET_M

    above_risk_altitude = df["altitude"] >= ALTITUDE_RISK_ONSET_M  # a True/False column: True for every row at/above the risk-relevant altitude
    # cumsum trick: count seconds where above_risk_altitude is True, but
    # ONLY counting a contiguous run from the first time it becomes True
    # (not resetting on brief dips) -- once you've climbed above the risk
    # threshold, "time at altitude" should keep accumulating even through
    # small fluctuations, matching how altitude illness risk actually
    # accumulates with continued exposure.
    # idxmax() on a boolean Series returns the index of the FIRST True value
    # (pandas treats True as 1, False as 0, and idxmax finds the first max).
    first_risk_idx = above_risk_altitude.idxmax() if above_risk_altitude.any() else None
    if first_risk_idx is not None and above_risk_altitude.any():
        # How many seconds have elapsed since the altitude first crossed the
        # risk threshold; .clip(lower=0) guards against any negative value
        # for rows that happen to come before that point in an edge case.
        seconds_since = (df["timestamp"] - df.loc[first_risk_idx, "timestamp"]).clip(lower=0)
        df["time_at_altitude_min"] = np.where(
            # np.where(condition, value_if_true, value_if_false): rows at or
            # after the risk-onset point get seconds_since converted to
            # minutes; rows before it (still ascending) get a flat 0.
            df["timestamp"] >= df.loc[first_risk_idx, "timestamp"], seconds_since / 60.0, 0.0
        )
    else:
        df["time_at_altitude_min"] = 0.0  # this subject's whole trajectory never reached the risk altitude at all

    return df


FEATURE_COLUMNS = [
    "spo2",
    "hr",
    "temp",
    "altitude",
    "spo2_trend_5min",
    "hr_trend_5min",
    "ascent_rate",
    "spo2_delta",
    "time_at_altitude_min",
]


def build_processed_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply engineer_features() per-subject (never blending rolling windows
    across subjects) and encode the ordinal severity label as an integer.
    """
    # include_groups=False (pandas >=2.2) excludes the group key column from
    # what's passed to engineer_features -- but we need subject_id back on
    # the output, so re-attach it per-group inside the lambda rather than
    # relying on the now-removed include_groups=True behavior.
    def _process_group(group: pd.DataFrame) -> pd.DataFrame:
        subject_id = group.name  # pandas sets .name to the current group's key value (the subject_id) inside a groupby callback
        result = engineer_features(group)
        result["subject_id"] = subject_id
        return result

    # group_keys=False: don't add an extra subject_id index level on top of
    # each group's own row index (we're re-attaching subject_id as a plain
    # column ourselves inside _process_group instead).
    processed = raw.groupby("subject_id", group_keys=False).apply(
        _process_group, include_groups=False
    )
    processed["severity_index"] = processed["severity_label"].map(SEVERITY_INDEX)  # string tier name -> its ordinal int (e.g. "Severe AMS" -> 2), via the config dict
    return processed.reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by SUBJECT, ordered by subject_id (which, for synthetic data, is
    generation order -- a stand-in for "time" since these aren't literally
    one continuous recording). Whole subjects go entirely into one split,
    never partial trajectories -- this is what actually prevents leakage:
    even a temporal split on ROWS within a mixed subject pool could still
    let train and test see different time-slices of the same person's
    trajectory, which leaks their individual physiology baseline across the
    split. Splitting on whole subjects avoids that entirely.

    Stratified-by-tier, temporal-within-tier: with only a couple dozen
    subjects per rare tier (HAPE/HACE risk), a single chronological cut
    across ALL subjects can starve val/test of an entire class purely by
    bad luck -- e.g. all HACE-risk subjects happening to land in train by
    chance of generation order. That's not a modeling problem, it's a
    sampling problem, and it silently makes val/test metrics meaningless
    for the missing class (a model literally cannot be evaluated on a tier
    with zero examples). So instead we group subjects by severity_label
    FIRST, then apply the same chronological train/val/test cut WITHIN each
    tier group. This still respects "no shuffling, val/test represent
    later-in-time than train" (the actual point of a temporal split) --
    it just guarantees that guarantee holds independently per class instead
    of being at the mercy of which classes happened to cluster where in
    the overall subject order.

    Once real Harespod data is merged in (harespod_loader.py), the same
    function applies: sort combined subjects by their recording start time
    (or synthetic generation index as today) and cut by fraction -- nothing
    else about this function needs to change.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9, (
        "Split fractions must sum to 1.0 -- check config.py"
    )

    # One row per subject with its label, so we can group by tier without
    # scanning the full (large) row-level dataframe repeatedly.
    subject_labels = df[["subject_id", "severity_label"]].drop_duplicates("subject_id")

    train_ids: set = set()
    val_ids: set = set()
    test_ids: set = set()

    for _tier, group in subject_labels.groupby("severity_label"):
        subject_ids = sorted(group["subject_id"])  # deterministic "time" order within tier
        n = len(subject_ids)
        n_train = max(1, int(n * train_frac)) if n >= 3 else n
        n_val = max(1, int(n * val_frac)) if n - n_train >= 2 else max(0, n - n_train)
        # For very small tier groups (n<3) everything goes to train -- with
        # so few examples there's no sane way to carve out val/test without
        # one of them being empty anyway; this just makes that explicit
        # rather than leaving an empty-but-silent split.
        train_ids.update(subject_ids[:n_train])
        val_ids.update(subject_ids[n_train : n_train + n_val])
        test_ids.update(subject_ids[n_train + n_val :])

    train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


def load_real_data_for_training() -> pd.DataFrame:
    """
    Load and rule-label every available real dataset, returning a
    DataFrame with the same schema as build_processed_dataset()'s
    synthetic output -- ready to concat onto the synthetic TRAIN split
    (never val/test -- see the long comment in run_pipeline() for why).

    Currently loads Harespod only. The pilot altitude dataset
    (pilot_altitude_loader.py) is deliberately NOT included here: it was
    investigated and found to use an oxygen-enriched protocol (~40-45% O2
    throughout, not ambient air), a fundamentally different physiological
    scenario than the ambient-air hypoxia this project models
    (docs/lls_mapping.md) -- rule-labeling it with ambient-air thresholds
    would actively mislabel it, not just approximate it. It stays loaded
    and available (pilot_altitude_loader.py) for exploration/visualization
    only. If a future oxygen-adjusted expected_spo2 curve is ever built
    (using the dataset's own O2/N2/CO2 columns), this function is where
    it would be added as a second real-data source.

    Returns an empty DataFrame (not an error) if no real dataset is
    downloaded yet -- run_pipeline() treats that as "proceed with
    synthetic-only training," matching CLAUDE.md's synthetic-first design.
    """
    from src.data import harespod_loader
    from src.data.label_real_data import label_real_dataframe

    if not harespod_loader.has_harespod_data():
        return pd.DataFrame()

    raw = harespod_loader.load_all_subjects()
    labeled = label_real_dataframe(raw)

    # Harespod has no temperature sensor (see harespod_loader.py docstring)
    # -- synthesize one from the same literature-based per-tier
    # distributions synth_data.py uses for pure-synthetic trajectories, so
    # FEATURE_COLUMNS' "temp" slot is never silently empty/NaN for real data.
    rng = np.random.default_rng(RANDOM_SEED)
    labeled["temp"] = synthesize_temperature(labeled["severity_label"], rng)

    return labeled


def run_pipeline(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load synthetic data -> engineer features -> temporal split -> merge in
    real data (train only) -> (optionally) save.

    Real data (Harespod, rule-labeled -- see load_real_data_for_training())
    is added ONLY to the train split, never val/test. This is a
    deliberate choice, not an oversight: Harespod's severity labels are
    generated by the SAME rule-based thresholds that rule_baseline.py uses
    as a model. If rule-labeled rows ended up in val/test, evaluating the
    rule baseline against them would be circular -- it would trivially
    "predict" labels it produced itself, making model_comparison.json
    misleading. Keeping val/test 100% synthetic (whose labels come from
    synth_data.py's own generation logic, not classify_row()) keeps every
    model's evaluation honest, while real data still gets to CONTRIBUTE
    more realistic training signal to XGBoost/LSTM.

    (A documented alternative -- splitting real data across all three sets
    and having run_comparison() score the rule baseline on synthetic-only
    rows while scoring XGBoost/LSTM on the full mixed set -- was considered
    and intentionally not built; see label_real_data.py's docstring for
    the tradeoff if this is ever revisited.)
    """
    raw_path = SYNTHETIC_DIR / "synthetic_vitals.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found -- run `python -m src.data.synth_data` first "
            "to generate the synthetic dataset."
        )
    raw = pd.read_csv(raw_path)

    processed = build_processed_dataset(raw)
    train_df, val_df, test_df = temporal_split(processed)

    # Tag synthetic rows explicitly rather than leaving data_source implicit
    # (NaN) for them -- every row in the final train/val/test files should
    # be traceable to its source, same reasoning as harespod_loader.py and
    # pilot_altitude_loader.py tagging their own output.
    train_df["data_source"] = "synthetic"
    val_df["data_source"] = "synthetic"
    test_df["data_source"] = "synthetic"

    real_train_rows = load_real_data_for_training()
    if not real_train_rows.empty:
        # Align columns explicitly (rather than a raw pd.concat) so a
        # schema drift in either source -- e.g. a new engineered feature
        # added to one but not the other -- fails loudly here instead of
        # silently producing NaN-filled columns in train.csv.
        missing_in_real = set(train_df.columns) - set(real_train_rows.columns)
        if missing_in_real:
            raise ValueError(
                f"Real training data is missing columns {missing_in_real} that "
                "the synthetic train split has -- fix load_real_data_for_training() "
                "before merging, rather than silently concatenating mismatched schemas."
            )
        real_train_rows = real_train_rows[train_df.columns]
        train_df = pd.concat([train_df, real_train_rows], ignore_index=True)
        print(
            f"Merged {len(real_train_rows):,} real-data rows "
            f"({real_train_rows['subject_id'].nunique()} subjects) into train split "
            "(val/test remain 100% synthetic -- see run_pipeline()'s docstring)"
        )

    print(f"Train: {len(train_df):,} rows / {train_df['subject_id'].nunique()} subjects")
    print(f"Val:   {len(val_df):,} rows / {val_df['subject_id'].nunique()} subjects")
    print(f"Test:  {len(test_df):,} rows / {test_df['subject_id'].nunique()} subjects")
    print("\nClass balance (train):")
    print(train_df["severity_label"].value_counts(normalize=True).round(3))

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
        print(f"\nSaved train/val/test.csv -> {PROCESSED_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
