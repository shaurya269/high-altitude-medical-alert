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
    onset_start = int(n_samples * onset_fraction)
    t = np.arange(n_samples, dtype=float)
    # Sigmoid centered at the midpoint of the post-onset window, width tuned
    # so the transition takes roughly 15-25% of the remaining trajectory --
    # gradual, not a step function (see module docstring).
    remaining = max(n_samples - onset_start, 1)
    center = onset_start + remaining * 0.5
    width = max(remaining * 0.18, 1.0)
    ramp = 1.0 / (1.0 + np.exp(-(t - center) / width))
    # Zero out anything before onset_start so there's a clean flat lead-in,
    # not just a very-early sigmoid tail (which would show as a tiny but
    # nonzero drift before symptoms are supposed to start).
    ramp[t < onset_start] = 0.0
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
    profile = TIER_PROFILES[tier]
    n = duration_minutes * 60 * SAMPLE_RATE_HZ

    # --- Altitude profile: gradual ascent to a randomized cruising altitude
    # above the AMS risk threshold, then hold. Ascent itself is a smooth
    # ramp (nobody teleports to altitude), which matters because the
    # ascent_rate feature (Stage 3 in the data flow diagram) is computed
    # FROM this trajectory downstream.
    cruise_altitude = rng.uniform(ALTITUDE_RISK_ONSET_M, ALTITUDE_RISK_ONSET_M + 2500)
    ascent_ramp = _smooth_ramp(n, onset_fraction=0.0, rng=rng)  # ramps up from t=0
    # Make the ascent itself fast relative to symptom onset -- reuse the
    # ramp shape but compress it into the first ~30% of the window.
    ascent_progress = np.clip(np.arange(n) / max(n * 0.3, 1), 0, 1)
    altitude = cruise_altitude * ascent_progress

    # --- Per-subject individual variation. Two subjects at the same tier
    # shouldn't produce identical numbers -- this is what keeps the dataset
    # from being trivially memorizable and forces the model to learn the
    # actual pattern rather than a lookup table.
    resting_hr = rng.uniform(*HR_NORMAL_RANGE)
    temp_center = rng.uniform(*TEMP_NORMAL_RANGE)
    altitude_sensitivity = rng.uniform(0.85, 1.15)  # some people desaturate faster than others

    # --- Symptom onset ramp: starts after ascent is mostly complete
    # (symptoms follow altitude exposure, they don't precede it), and either
    # plateaus or keeps progressing per the tier's `progressive` flag.
    onset_fraction = rng.uniform(0.25, 0.45)
    symptom_ramp = _smooth_ramp(n, onset_fraction, rng)
    if not profile.progressive:
        # Plateau tiers: once the ramp is mostly up (>0.85), hold steady
        # rather than let the sigmoid's natural asymptotic creep keep
        # nudging it toward 1.0 -- we want a genuine flat plateau, which
        # matters for the ML model learning "worsening trend" as a HAPE/HACE
        # signal specifically (see docs/lls_mapping.md).
        plateau_val = 0.85
        symptom_ramp = np.minimum(symptom_ramp, plateau_val) + (
            plateau_val * (symptom_ramp >= plateau_val * 0.98)
        ) * 0  # no-op keeps the clamp explicit/commented rather than magic
        symptom_ramp = np.clip(symptom_ramp, 0, plateau_val)

    spo2_delta_target = rng.uniform(*profile.spo2_delta_range) * altitude_sensitivity
    hr_elev_target = rng.uniform(*profile.hr_elevation_range)
    temp_delta_target = rng.uniform(*profile.temp_delta_range)

    spo2_delta = symptom_ramp * spo2_delta_target
    hr_elevation = symptom_ramp * hr_elev_target
    temp_delta = symptom_ramp * temp_delta_target

    # --- Compose final signals from the reference curve + ramped deviation
    # + realistic sensor noise. Noise matters: a model trained on noiseless
    # synthetic data would be brittle against a real (noisy) Arduino stream.
    baseline_spo2 = expected_spo2(altitude) - spo2_delta
    spo2 = baseline_spo2 + rng.normal(0, 0.6, n)
    spo2 = np.clip(spo2, 50, 100)  # physiological floor/ceiling

    hr = resting_hr + hr_elevation + rng.normal(0, 2.5, n)
    hr = np.clip(hr, 35, 200)

    temp = temp_center + temp_delta + rng.normal(0, 0.08, n)

    timestamp = np.arange(n) / SAMPLE_RATE_HZ

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
    rng = np.random.default_rng(seed)
    tiers = list(TIER_PREVALENCE.keys())
    probs = list(TIER_PREVALENCE.values())

    frames = []
    for i in range(n_subjects):
        tier = rng.choice(tiers, p=probs)
        subject_id = f"synth_{i:04d}"
        frames.append(simulate_trajectory(subject_id, tier, duration_minutes, rng))

    dataset = pd.concat(frames, ignore_index=True)
    return dataset


def save_dataset(df: pd.DataFrame, filename: str = "synthetic_vitals.csv") -> None:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows ({df['subject_id'].nunique()} subjects) -> {out_path}")


if __name__ == "__main__":
    dataset = generate_dataset()
    print(dataset["severity_label"].value_counts(normalize=True).round(3))
    save_dataset(dataset)
