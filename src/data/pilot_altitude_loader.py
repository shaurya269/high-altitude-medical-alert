"""
Loader for the "High-Altitude Pilot Physiological Monitoring Dataset"
(Jia, Yang, Zhao; Sci Data 2026, DOI 10.1038/s41597-025-06508-1,
PMC12886808; data on Figshare DOI 10.6084/m9.figshare.29947679, companion
code https://github.com/Garethjia/Dataset).

Why this dataset exists in this project alongside Harespod
(harespod_loader.py): while investigating Harespod's absolute-unit gap (see
that module's docstring), this dataset was found and turned out to be a
better fit -- SpO2/HR/respiratory rate values here are in real physical
units (percent, bpm, breaths/min) with no normalization ambiguity, verified
directly by inspecting the downloaded CSVs (SpO2 88-100%, HR 43-179bpm --
physiologically sane numbers, not a [0,1] range). It also covers a HIGHER
altitude range (4500-7500m) than Harespod (1500-4000m), better matching the
HAPE/HACE risk tiers this project cares about most.

Protocol shape -- worth understanding, since it differs from a mountaineer's
ASCENT: this is a hypobaric-chamber DECOMPRESSION study for aviation
physiology (rapid altitude increase then a stepped descent: 7500 -> 7000 ->
6500 -> 6000 -> 5500 -> 5000 -> [4500]m, confirmed identical direction
across all 20 downloaded subject files). The direction (descending vs a
climber's ascending) doesn't matter for what this project trains on --
severity classification cares about the relationship between CURRENT
altitude/vitals and how long you've been exposed, not which direction you
got there. But it's worth knowing this is a decompression/descent profile,
not an ascent, if the data is ever visualized or described in the README.

Structure actually found (verified by opening the files, not assumed):
  data/raw/pilot_altitude/Data/<MMDDx>.csv     -- 20 subject files, e.g. "0609b.csv"
    Columns (with header row): Altitude,Time,SPO2,Heart rate,Respiratory rate,
                                O2,N2,CO2,other
    Altitude   -- meters, integer, changes in discrete steps (chamber protocol)
    Time       -- HH:MM:SS, 1Hz, no date component (verified no midnight
                  wraparound in any of the 20 files, so safe to treat as a
                  same-day session)
    SPO2       -- percent, float, real units (Yanpai 9500 sensor, ±2% accuracy
                  per the paper -- NOT normalized, unlike Harespod)
    Heart rate -- bpm, int (TE-4000 monitor, ±5% accuracy per the paper)
    Respiratory rate, O2/N2/CO2/other -- present but not currently used by
                  this project's FEATURE_COLUMNS; loaded and kept in case
                  useful for a future feature (e.g. gas composition as a
                  confound check)
  Data/Code/ + Dataset-main/*.py                -- the paper's own analysis
                  scripts (box plots, heatmaps, Pearson correlation) --
                  reference only, not used by this loader

No severity labels are included (same gap as Harespod) -- filled the same
way, via the LLS-derived rules in docs/lls_mapping.md, applied post-hoc in
feature_engineering.py once this data is merged with the synthetic set.

No temperature column either (same gap as Harespod) -- still filled via
synth_data.py's literature-based temperature synthesis for any blending
that needs it; this loader does not fabricate a temp column itself so real
vs. synthesized values stay distinguishable at the source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

PILOT_ALTITUDE_DIR = RAW_DIR / "pilot_altitude"
PILOT_DATA_DIR = PILOT_ALTITUDE_DIR / "Data"

TIDY_COLUMNS = ["timestamp", "spo2", "hr", "respiratory_rate", "altitude", "subject_id", "data_source"]


def has_pilot_altitude_data() -> bool:
    """Cheap existence check -- True once Data/*.csv files are present."""
    if not PILOT_DATA_DIR.exists():
        return False
    return any(PILOT_DATA_DIR.glob("*.csv"))


def _raise_not_downloaded() -> None:
    raise FileNotFoundError(
        "Pilot altitude dataset not found in data/raw/pilot_altitude/Data/.\n\n"
        "This is a manual browser download, not something this pipeline fetches "
        "automatically.\n\n"
        "To get it:\n"
        "  1. Visit https://doi.org/10.6084/m9.figshare.29947679\n"
        "  2. Download the dataset archive and unzip its contents into\n"
        f"     {PILOT_ALTITUDE_DIR}\n"
        "  3. Companion code: https://github.com/Garethjia/Dataset\n\n"
        "Until then, the rest of the pipeline runs on the synthetic generator "
        "(src/data/synth_data.py) and/or Harespod (harespod_loader.py) instead."
    )


def list_subject_ids() -> list[str]:
    """Every subject file's stem, e.g. ['0609b', '0610b', ...]."""
    if not has_pilot_altitude_data():
        _raise_not_downloaded()
    return sorted(p.stem for p in PILOT_DATA_DIR.glob("*.csv"))


def load_subject(subject_id: str) -> pd.DataFrame:
    """
    Load one subject's recording as a tidy 1Hz DataFrame, matching
    TIDY_COLUMNS -- the same general shape synth_data.py and
    harespod_loader.py produce, so feature_engineering.py can treat all
    three sources through the same downstream code.

    Unlike harespod_loader.py, no rescaling is applied here -- SPO2 and
    Heart rate are already in real physical units (see module docstring),
    so this loader is a straight column rename + timestamp normalization,
    not an approximation. data_source is still tagged "pilot_altitude" so
    it stays traceable and distinguishable from synthetic data, even though
    (unlike Harespod) there's no known accuracy caveat attached to it beyond
    the sensors' own documented ±2%/±5% accuracy specs.
    """
    if not has_pilot_altitude_data():
        _raise_not_downloaded()

    csv_path = PILOT_DATA_DIR / f"{subject_id}.csv"
    if not csv_path.exists():
        available = list_subject_ids()
        raise FileNotFoundError(
            f"No pilot altitude subject file at {csv_path}. Available subjects: {available}"
        )

    raw = pd.read_csv(csv_path)
    required = {"Altitude", "Time", "SPO2", "Heart rate"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected columns {missing} -- the dataset's "
            "column names may differ from what this loader assumes. Inspect the "
            "file directly before trusting this subject."
        )

    # Time is HH:MM:SS with no date -- parse as a time-of-day, then convert
    # to seconds-from-recording-start (matching synth_data.py's convention
    # of timestamp starting at 0.0), rather than keeping wall-clock time.
    # Verified (module docstring) that none of the 20 files cross midnight,
    # so a same-day parse is safe here -- if that assumption were ever
    # violated for a new file, the resulting negative/huge timestamp deltas
    # would be obviously wrong and easy to catch, not silently corrupted.
    time_parsed = pd.to_datetime(raw["Time"], format="%H:%M:%S")
    seconds = (time_parsed - time_parsed.iloc[0]).dt.total_seconds()

    tidy = pd.DataFrame(
        {
            "timestamp": seconds,
            "spo2": raw["SPO2"].astype(float),
            "hr": raw["Heart rate"].astype(float),
            "respiratory_rate": raw.get("Respiratory rate", pd.Series(dtype=float)),
            "altitude": raw["Altitude"].astype(float),
            "subject_id": f"pilot_{subject_id}",
            "data_source": "pilot_altitude",
        }
    )
    return tidy.sort_values("timestamp").reset_index(drop=True)


def load_all_subjects() -> pd.DataFrame:
    """Load and concatenate every subject's recording."""
    subject_ids = list_subject_ids()
    frames = [load_subject(sid) for sid in subject_ids]
    return pd.concat(frames, ignore_index=True)
