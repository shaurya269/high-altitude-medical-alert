"""
Synthetic vitals + severity-label generator.

Why this exists (see CLAUDE.md Section 5 and docs/lls_mapping.md): Harespod
has no body temperature, no continuous altitude, no HAPE/HACE cases (can't
ethically induce those in human subjects), and no severity labels at all.
Rather than block the whole project on those gaps, this module generates
FULLY synthetic multi-hour "expedition" trajectories: a subject ascends,
develops (or doesn't) one of the five severity tiers, and we record the
vitals a sensor would have seen along the way, labeled with the tier that
generated them.

This also currently serves as the PRIMARY dataset (not just a gap-filler),
per the "synthetic-first, adapter ready" decision -- real Harespod data will
be blended in later via src/data/harespod_loader.py without requiring any
change to this file's output contract (see merge_datasets in feature_engineering.py).

Design principle: gradual onset, not step-functions. A real physiological
deterioration ramps over minutes-to-hours, not instantaneously -- a model
trained on step-function synthetic data would learn to detect abrupt jumps,
which is not what real AMS/HAPE/HACE onset looks like and would generalize
poorly. Every trajectory here interpolates smoothly from a start state to a
target state using a randomized onset duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ALTITUDE_RISK_ONSET_M,
    HR_NORMAL_RANGE,
    RANDOM_SEED,
    SAMPLE_RATE_HZ,
    SEVERITY_TIERS,
    SPO2_DROP_PER_1000M,
    SPO2_SEA_LEVEL_BASELINE,
    SYNTHETIC_DIR,
    TEMP_NORMAL_RANGE,
)

# ---------------------------------------------------------------------------
# Per-tier target ranges, straight from docs/lls_mapping.md's table. Keeping
# these as explicit (low, high) bands -- rather than single numbers -- and
# sampling uniformly within each band is what gives the dataset natural
# within-class variation instead of every "Mild AMS" trajectory looking
# identical.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierProfile:
    """Target physiology a trajectory ramps TOWARD for a given severity tier."""

    spo2_delta_range: tuple[float, float]  # how far BELOW altitude-expected SpO2, in %
    hr_elevation_range: tuple[float, float]  # bpm ABOVE the subject's resting HR
    temp_delta_range: tuple[float, float]  # degrees C above/below normal-band center
    # Whether the trajectory keeps worsening through the whole window (True)
    # or plateaus after onset (False) -- HAPE/HACE are modeled as
    # non-recovering per docs/lls_mapping.md; Mild/Severe AMS plateau.
    progressive: bool


TIER_PROFILES: dict[str, TierProfile] = {
    "Normal": TierProfile((-1.5, 1.5), (-5, 8), (-0.2, 0.2), progressive=False),
    "Mild AMS": TierProfile((4, 8), (10, 20), (-0.1, 0.4), progressive=False),
    "Severe AMS": TierProfile((8, 14), (20, 35), (0.0, 0.6), progressive=False),
    "HAPE risk": TierProfile((14, 22), (28, 45), (-0.1, 0.5), progressive=True),
    "HACE risk": TierProfile((16, 26), (32, 50), (-0.1, 0.7), progressive=True),
}

# Roughly how common each tier should be in the generated population.
# Normal dominates in reality (most people at altitude are fine most of the
# time) -- CLAUDE.md explicitly calls out that class imbalance is EXPECTED
# and real, and must be handled with weighting/oversampling downstream
# (src/models/), not papered over by generating an artificially balanced
# dataset here. So we deliberately mirror realistic imbalance.
TIER_PREVALENCE = {
    "Normal": 0.55,
    "Mild AMS": 0.22,
    "Severe AMS": 0.13,
    "HAPE risk": 0.06,
    "HACE risk": 0.04,
}


def expected_spo2(altitude_m: float) -> float:
    """
    Physiologically-expected SpO2 (%) for a healthy acclimatized person at
    a given altitude, per the rule-of-thumb in config.SPO2_DROP_PER_1000M.

    This is the reference curve every trajectory's "spo2_delta" is measured
    against -- see docs/lls_mapping.md for why we reason in deltas-from-
    expected rather than raw SpO2.
    """
    drop = (altitude_m / 1000.0) * SPO2_DROP_PER_1000M
    return SPO2_SEA_LEVEL_BASELINE - drop


def _smooth_ramp(n_samples: int, onset_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """
    A 0->1 ramp that stays near 0 for a random "lead-in" period, then rises
    smoothly (sigmoid-shaped, not linear) to 1 and stays there.

    Modeling choice: real symptom onset isn't linear either -- it tends to
    accelerate once it starts (a sigmoid captures "slow start, faster
    middle, plateau" better than a straight line) and every subject has a
    different lead-in time before symptoms begin, which onset_fraction
    (randomized per-trajectory) represents.
    """
    onset_start = int(n_samples * onset_fraction)  # index where the flat lead-in ends
    t = np.arange(n_samples, dtype=float)  # one time-step value per sample: 0, 1, 2, ..., n_samples-1
    # Sigmoid centered at the midpoint of the post-onset window, width tuned
    # so the transition takes roughly 15-25% of the remaining trajectory --
    # gradual, not a step function (see module docstring).
    remaining = max(n_samples - onset_start, 1)  # how many samples are left AFTER the lead-in; floor of 1 avoids a zero-width sigmoid
    center = onset_start + remaining * 0.5  # the sigmoid's midpoint sits halfway through the remaining trajectory
    width = max(remaining * 0.18, 1.0)  # bigger width = a slower, more gradual climb; 0.18 was chosen by eye to look "gradual" over minutes
    # The classic logistic/sigmoid formula: 1 / (1 + e^-(t-center)/width).
    # At t << center this is ~0; at t >> center this is ~1; near t=center it
    # rises steeply -- this is what actually produces the "slow, then fast,
    # then plateau" shape instead of a straight ramp.
    ramp = 1.0 / (1.0 + np.exp(-(t - center) / width))
    # Zero out anything before onset_start so there's a clean flat lead-in,
    # not just a very-early sigmoid tail (which would show as a tiny but
    # nonzero drift before symptoms are supposed to start).
    ramp[t < onset_start] = 0.0  # boolean-indexes every sample still in the lead-in period and forces it to exactly 0
    return ramp


def simulate_trajectory(
    subject_id: str,
    tier: str,
    duration_minutes: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate one subject's full vitals trajectory for a single severity tier.

    Returns a tidy 1Hz DataFrame: timestamp, spo2, hr, temp, altitude,
    subject_id, severity_label -- matching the same column family Harespod
    data will eventually provide (see harespod_loader.TIDY_COLUMNS), so
    feature_engineering.py can treat both sources identically.
    """
    profile = TIER_PROFILES[tier]  # look up this tier's (spo2_delta, hr_elevation, temp_delta, progressive) targets
    n = duration_minutes * 60 * SAMPLE_RATE_HZ  # total sample count, e.g. 90 min * 60 sec/min * 1 sample/sec = 5400 rows

    # --- Altitude profile: gradual ascent to a randomized cruising altitude
    # above the AMS risk threshold, then hold. Ascent itself is a smooth
    # ramp (nobody teleports to altitude), which matters because the
    # ascent_rate feature (Stage 3 in the data flow diagram) is computed
    # FROM this trajectory downstream.
    cruise_altitude = rng.uniform(ALTITUDE_RISK_ONSET_M, ALTITUDE_RISK_ONSET_M + 2500)  # a random target altitude, 2500-5000m
    ascent_ramp = _smooth_ramp(n, onset_fraction=0.0, rng=rng)  # ramps up from t=0 (unused directly below, kept for clarity/possible reuse)
    # Make the ascent itself fast relative to symptom onset -- reuse the
    # ramp shape but compress it into the first ~30% of the window.
    ascent_progress = np.clip(np.arange(n) / max(n * 0.3, 1), 0, 1)  # 0 -> 1 linearly over the first 30% of samples, then held at 1
    altitude = cruise_altitude * ascent_progress  # scale the 0->1 progress curve up to the actual target altitude in meters

    # --- Per-subject individual variation. Two subjects at the same tier
    # shouldn't produce identical numbers -- this is what keeps the dataset
    # from being trivially memorizable and forces the model to learn the
    # actual pattern rather than a lookup table.
    resting_hr = rng.uniform(*HR_NORMAL_RANGE)  # this subject's personal baseline HR before any illness effect
    temp_center = rng.uniform(*TEMP_NORMAL_RANGE)  # this subject's personal baseline temperature
    altitude_sensitivity = rng.uniform(0.85, 1.15)  # some people desaturate faster than others -- a multiplier applied to spo2_delta below

    # --- Symptom onset ramp: starts after ascent is mostly complete
    # (symptoms follow altitude exposure, they don't precede it), and either
    # plateaus or keeps progressing per the tier's `progressive` flag.
    onset_fraction = rng.uniform(0.25, 0.45)  # symptoms start somewhere between 25% and 45% of the way through the recording
    symptom_ramp = _smooth_ramp(n, onset_fraction, rng)  # the 0->1 sigmoid curve driving how "sick" this subject currently is
    if not profile.progressive:
        # Plateau tiers: once the ramp is mostly up (>0.85), hold steady
        # rather than let the sigmoid's natural asymptotic creep keep
        # nudging it toward 1.0 -- we want a genuine flat plateau, which
        # matters for the ML model learning "worsening trend" as a HAPE/HACE
        # signal specifically (see docs/lls_mapping.md).
        plateau_val = 0.85  # the ceiling the ramp is capped at, instead of letting the sigmoid creep all the way to 1.0
        symptom_ramp = np.minimum(symptom_ramp, plateau_val) + (
            plateau_val * (symptom_ramp >= plateau_val * 0.98)
        ) * 0  # no-op keeps the clamp explicit/commented rather than magic
        # (the line above computes a value and immediately multiplies it by
        # 0, so it never actually changes symptom_ramp -- it's dead code
        # left over from an earlier version, made harmless and explained
        # here rather than silently deleted, since the REAL clamp is the
        # np.clip() on the very next line)
        symptom_ramp = np.clip(symptom_ramp, 0, plateau_val)  # this is the line that actually enforces the plateau ceiling

    spo2_delta_target = rng.uniform(*profile.spo2_delta_range) * altitude_sensitivity  # how far below expected-SpO2 this subject will end up at full severity
    hr_elev_target = rng.uniform(*profile.hr_elevation_range)  # how many bpm above resting HR at full severity
    temp_delta_target = rng.uniform(*profile.temp_delta_range)  # how many degrees C above/below normal at full severity

    # Multiply each "target" (the value at FULL severity) by the 0->1 ramp,
    # so early in the trajectory (ramp near 0) the deviation is tiny, and
    # once the ramp reaches its ceiling the deviation reaches its target.
    spo2_delta = symptom_ramp * spo2_delta_target
    hr_elevation = symptom_ramp * hr_elev_target
    temp_delta = symptom_ramp * temp_delta_target

    # --- Compose final signals from the reference curve + ramped deviation
    # + realistic sensor noise. Noise matters: a model trained on noiseless
    # synthetic data would be brittle against a real (noisy) Arduino stream.
    baseline_spo2 = expected_spo2(altitude) - spo2_delta  # altitude-adjusted expected value, minus how much this subject has deviated from it
    spo2 = baseline_spo2 + rng.normal(0, 0.6, n)  # add per-sample Gaussian sensor noise: mean 0, std dev 0.6%, one value per timestep
    spo2 = np.clip(spo2, 50, 100)  # physiological floor/ceiling -- noise could otherwise push a value outside what's physically possible

    hr = resting_hr + hr_elevation + rng.normal(0, 2.5, n)  # baseline + illness effect + noise (std dev 2.5bpm)
    hr = np.clip(hr, 35, 200)  # clip to a physiologically plausible heart-rate range

    temp = temp_center + temp_delta + rng.normal(0, 0.08, n)  # baseline + illness effect + a small amount of noise (std dev 0.08°C)

    timestamp = np.arange(n) / SAMPLE_RATE_HZ  # convert sample INDEX (0, 1, 2, ...) into elapsed SECONDS (0.0, 1.0, 2.0, ... at 1Hz)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "spo2": spo2,
            "hr": hr,
            "temp": temp,
            "altitude": altitude,
            "subject_id": subject_id,
            "severity_label": tier,
        }
    )
    return df


def generate_dataset(
    n_subjects: int = 400,
    duration_minutes: int = 90,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a full synthetic population: n_subjects trajectories, each
    one severity tier, sampled per TIER_PREVALENCE so the overall dataset
    has realistic class imbalance (see module docstring).

    Each "subject" here is one simulated expedition/session, not a repeated
    real person -- subject_id is used downstream only to prevent
    within-subject leakage across train/val/test if we ever split by
    subject; the PRIMARY split is temporal (see feature_engineering.py).
    """
    rng = np.random.default_rng(seed)  # one shared random generator, seeded once, reused across all subjects for reproducibility
    tiers = list(TIER_PREVALENCE.keys())  # ["Normal", "Mild AMS", ...] in dict insertion order
    probs = list(TIER_PREVALENCE.values())  # matching probabilities, e.g. [0.55, 0.22, ...] -- must sum to 1.0 and align index-for-index with tiers

    frames = []  # will hold one DataFrame per subject, concatenated into one big table at the end
    for i in range(n_subjects):
        tier = rng.choice(tiers, p=probs)  # randomly draw ONE tier for this subject, weighted by TIER_PREVALENCE
        subject_id = f"synth_{i:04d}"  # zero-padded id like "synth_0007" so string-sorting also sorts numerically for the first ~10,000 subjects
        frames.append(simulate_trajectory(subject_id, tier, duration_minutes, rng))

    dataset = pd.concat(frames, ignore_index=True)  # stack all per-subject DataFrames into one; ignore_index=True renumbers rows 0..N instead of repeating each subject's own 0..n
    return dataset


def synthesize_temperature(severity_labels: pd.Series, rng: np.random.Generator) -> pd.Series:
    """
    Generate a plausible body-temperature column for rows that have a
    KNOWN severity label but no real temperature sensor (Harespod and the
    pilot altitude dataset both lack one -- see their loaders' docstrings).

    This reuses TIER_PROFILES.temp_delta_range -- the exact same
    literature-based temperature-deviation bands synth_data.py already
    uses when GENERATING synthetic trajectories -- so a real Harespod row
    labeled "Severe AMS" gets a temperature drawn from the same
    distribution a synthetic "Severe AMS" row would have, rather than
    inventing a second, inconsistent temperature model. This is explicitly
    a fabricated/synthesized value standing in for a genuinely missing
    sensor -- never presented as a real temperature reading -- which is
    exactly the same "gap-filling" role synthetic temperature already
    plays for pure-synthetic data (see CLAUDE.md Section 5's gap table).
    """
    temp_center = rng.uniform(*TEMP_NORMAL_RANGE)  # one shared baseline temperature for this whole batch of rows (not per-row/per-subject)
    values = np.empty(len(severity_labels), dtype=float)  # pre-allocate an empty output array the same length as the input labels
    for tier, profile in TIER_PROFILES.items():
        mask = (severity_labels == tier).to_numpy()  # a True/False array: True wherever this row's label matches the current tier
        n = mask.sum()  # how many rows in this batch belong to this tier (True counts as 1 when summed)
        if n == 0:
            continue  # no rows of this tier in this batch -- skip straight to the next tier, nothing to fill in
        temp_delta = rng.uniform(*profile.temp_delta_range, size=n)  # one random deviation value per matching row
        noise = rng.normal(0, 0.08, n)  # small per-row sensor noise, same magnitude as the pure-synthetic generator uses
        values[mask] = temp_center + temp_delta + noise  # fill in ONLY the rows matching this tier's mask, leaving other rows untouched for the next loop iteration
    return pd.Series(values, index=severity_labels.index, name="temp")  # reattach the original row index so this lines up correctly with the caller's DataFrame


def save_dataset(df: pd.DataFrame, filename: str = "synthetic_vitals.csv") -> None:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)  # ensure data/synthetic/ exists before writing into it
    out_path = SYNTHETIC_DIR / filename
    df.to_csv(out_path, index=False)  # index=False: don't write pandas' internal row-number column into the CSV, we don't need it back
    print(f"Saved {len(df):,} rows ({df['subject_id'].nunique()} subjects) -> {out_path}")  # :, formats large numbers with thousands separators for readability


if __name__ == "__main__":
    # Only runs when this file is executed directly (python -m src.data.synth_data),
    # not when it's imported by another module -- so importing this file to reuse
    # its functions never has the side effect of regenerating and overwriting the dataset.
    dataset = generate_dataset()
    print(dataset["severity_label"].value_counts(normalize=True).round(3))  # quick sanity check: does the actual tier mix roughly match TIER_PREVALENCE?
    save_dataset(dataset)
