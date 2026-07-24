"""
Loader/adapter for the Harespod hypobaric-chamber dataset.

Harespod (Figshare DOI 10.6084/m9.figshare.c.6623344.v1, companion code at
https://github.com/oca-john/Harespod) records SpO2, heart rate, and
respiration at 100Hz from 15 subjects in a hypobaric (altitude-simulating)
chamber. It has NO body temperature and NO continuous altitude/pressure
signal -- only discrete altitude-change markers -- see CLAUDE.md Section 5
for the full gap list and how each is filled downstream.

This module is a placeholder/adapter, written before the actual files have
been downloaded. That's intentional (see CLAUDE.md decision: "synthetic-first,
adapter ready") -- it means:
  1. The rest of the pipeline (Day 2 onward) is NOT blocked on a manual
     browser download of a Figshare archive.
  2. The moment the real files land in data/raw/harespod/, this loader
     starts producing real data with zero changes needed anywhere else --
     every downstream stage (synthetic augmentation, feature engineering,
     labeling) consumes the same tidy DataFrame shape regardless of source.

If you (the project owner) download Harespod, unzip it into
data/raw/harespod/ and re-run `has_harespod_data()` -- if the expected files
aren't found yet, everything upstream from this module raises a clear,
actionable error rather than silently returning garbage or crashing deep
inside pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HARESPOD_DIR, SAMPLE_RATE_HZ

# Harespod's raw sample rate per the dataset documentation. We downsample to
# SAMPLE_RATE_HZ (1Hz) because that's a realistic Arduino polling rate --
# no point training on time resolution the eventual hardware can't produce.
HARESPOD_RAW_HZ = 100

# Expected tidy columns once a subject's file is loaded and downsampled.
# timestamp: seconds since recording start
# spo2:      blood oxygen saturation, %
# hr:        heart rate, bpm
# resp:      respiration rate, breaths/min (Harespod has this; we don't use
#            it as a model feature yet, but keep it -- it's free signal that
#            may prove useful for feature engineering later)
# altitude:  meters, INTERPOLATED from Harespod's discrete change markers,
#            not a native continuous column (see interpolate_altitude below)
# subject_id: which of the 15 subjects this recording is from
TIDY_COLUMNS = ["timestamp", "spo2", "hr", "resp", "altitude", "subject_id"]


def has_harespod_data() -> bool:
    """
    Cheap existence check other modules call before attempting to load.

    Returns False (not an exception) so callers -- e.g. a data-merge script
    that wants to "use Harespod if present, else pure synthetic" -- can
    branch on this without a try/except around a heavier load call.
    """
    if not HARESPOD_DIR.exists():
        return False
    # Harespod's companion repo ships per-subject files; we don't know the
    # exact final filenames until the real archive is downloaded, so this
    # check is intentionally loose: "is there anything here at all?"
    return any(HARESPOD_DIR.iterdir())


def _raise_not_downloaded() -> None:
    raise FileNotFoundError(
        "Harespod data not found in data/raw/harespod/.\n\n"
        "This is expected if you haven't downloaded it yet -- it's a manual "
        "browser download, not something this pipeline fetches automatically.\n\n"
        "To get it:\n"
        "  1. Visit https://doi.org/10.6084/m9.figshare.c.6623344.v1\n"
        "  2. Download the dataset archive and unzip its contents into\n"
        f"     {HARESPOD_DIR}\n"
        "  3. (Optional) companion loading code: https://github.com/oca-john/Harespod\n\n"
        "Until then, the rest of the pipeline runs on the synthetic generator "
        "(src/data/synth_data.py) instead -- see CLAUDE.md Section 5."
    )


def interpolate_altitude(
    df: pd.DataFrame, altitude_markers: list[tuple[float, float]]
) -> pd.Series:
    """
    Interpolate a continuous altitude column from discrete change markers.

    Harespod records altitude as step-change events (e.g. "at t=120s,
    chamber altitude set to 4000m") rather than a continuous sensor trace,
    because it's a hypobaric CHAMBER, not a real ascent -- the operators
    change pressure in discrete steps. Real ascents (and our synthetic data)
    are gradual, so we linearly interpolate BETWEEN markers to approximate
    a continuous trajectory. This is a modeling simplification: it assumes
    a steady climb/descent rate between two known points, which is a
    reasonable approximation for chamber protocols that ramp pressure over
    a few minutes rather than stepping instantly.

    Args:
        df: tidy dataframe with a 'timestamp' column (seconds)
        altitude_markers: list of (timestamp_seconds, altitude_meters) pairs,
            sorted by timestamp, marking known altitude at known times.

    Returns:
        pd.Series of interpolated altitude (meters), same length/index as df.
    """
    if not altitude_markers:
        raise ValueError(
            "No altitude markers provided -- Harespod's raw files should "
            "include chamber-altitude-change events; without them we have "
            "no basis to interpolate a continuous altitude trace."
        )
    marker_times = np.array([t for t, _ in altitude_markers], dtype=float)
    marker_alts = np.array([a for _, a in altitude_markers], dtype=float)
    # np.interp clamps to the first/last marker value outside the known
    # range, which is the right behavior here (altitude doesn't change
    # before the first marker or after the last one is recorded).
    return pd.Series(
        np.interp(df["timestamp"].to_numpy(), marker_times, marker_alts),
        index=df.index,
        name="altitude",
    )


def load_subject(subject_id: str) -> pd.DataFrame:
    """
    Load one subject's Harespod recording as a tidy 1Hz DataFrame.

    NOTE: this function's internals are a placeholder until the real
    Harespod file format is confirmed post-download -- the companion repo
    (github.com/oca-john/Harespod) documents subject-level file naming that
    we haven't inspected yet since the archive isn't downloaded. When you
    add the real files, check their loading code and adjust the file-read
    logic below (the tidy OUTPUT shape / TIDY_COLUMNS should not need to
    change, since everything downstream is written against that contract).
    """
    if not has_harespod_data():
        _raise_not_downloaded()

    # Placeholder path pattern -- adjust once real filenames are known.
    candidate = HARESPOD_DIR / f"{subject_id}.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected Harespod subject file at {candidate} but it doesn't "
            "exist. The exact naming convention will be confirmed once the "
            "real archive is unzipped -- update this path pattern then."
        )

    raw = pd.read_csv(candidate)
    # Downsample 100Hz -> 1Hz by taking every Nth row. A simple stride
    # (rather than averaging) preserves the true sensor noise character,
    # which matters for training a model that will see equally-noisy real
    # sensor data -- averaging would make the training data cleaner than
    # what the Arduino will actually produce.
    stride = HARESPOD_RAW_HZ // SAMPLE_RATE_HZ
    downsampled = raw.iloc[::stride].reset_index(drop=True)
    downsampled["subject_id"] = subject_id
    return downsampled


def load_all_subjects() -> pd.DataFrame:
    """Load and concatenate every available Harespod subject recording."""
    if not has_harespod_data():
        _raise_not_downloaded()

    subject_files = sorted(HARESPOD_DIR.glob("*.csv"))
    if not subject_files:
        _raise_not_downloaded()

    frames = [load_subject(f.stem) for f in subject_files]
    return pd.concat(frames, ignore_index=True)
