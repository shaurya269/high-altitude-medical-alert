"""
Loader/adapter for the real, downloaded Harespod hypobaric-chamber dataset.

Harespod (Figshare DOI 10.6084/m9.figshare.c.6623344.v1, companion code at
https://github.com/oca-john/Harespod, described in Zhang et al. 2024, Sci
Data, https://doi.org/10.1038/s41597-024-03065-x / PMC10899206) records
SpO2, heart rate, pulse rate, and respiration from 15 subjects in a
hypobaric (altitude-simulating) chamber, ramped through 1500/2000/2500/
3000/3500/4000m stages with a steady period at each. See CLAUDE.md Section
5 for the full gap list (no body temperature, no HAPE/HACE cases, no
severity labels) and how each is filled by synth_data.py.

Downloaded layout actually found in data/raw/harespod/ (verified against
the dataset's own README/Instruction.md and the paper -- NOT a guess):
  Data_Cons/<subject_id>/         15 usable subjects, e.g. "217a", "218c"
    hr_5cut.csv                   1Hz, columns: timestamp, heart_rate (normalized 0-1)
    prt_5cut.csv                  1Hz, columns: timestamp, pulse_rate (normalized 0-1)
    spv_5cut.csv                  1Hz, columns: timestamp, SpO2 saturation (normalized 0-1)
    spo_5cut.csv                  100Hz, SpO2 PLETHYSMOGRAPH WAVEFORM (not a %) -- unused here
    rsp_5cut.csv                  100Hz, respiration waveform
    key_timestamp.txt             6 wall-clock timestamps marking when the chamber
                                   reached each altitude stage (1500/2000/2500/3000/
                                   3500/4000m), used as interpolate_altitude() markers
  Data_Disc/<subject_id>/         same signals pre-split into per-stage files
                                   (hr_20.csv = the 2000m stage, etc.) -- not used here,
                                   Data_Cons's single continuous recording is simpler
                                   to feed through the same pipeline as synthetic data
  Data_Incomp/                    excluded by the dataset's own authors (see
                                   Instruction.md) -- not loaded

Column name -> physical meaning was NOT obvious from the CSVs alone (they're
headerless, unlabeled numbers) -- confirmed by reading the actual device
driver source in Codes_Arch/Berry_Related/berry_modified_pm6100.cpp, which
names the BerryMed PM6100 monitor's output fields directly:
  hr  -> m_HeartRate    (Heart Rate)
  prt -> m_PulseRate    (Pulse Rate -- a second, related HR-like signal)
  spv -> m_SPO2Sat      (SpO2 Saturation, i.e. the actual SpO2 percentage)
  spo -> SPO2 Wave      (raw plethysmograph waveform, NOT a percentage -- unused)
  rsp -> RESP Wave      (respiration waveform)
So `spv`, not `spo`, is what this loader treats as "SpO2%" -- a subtle but
important distinction that would have silently produced nonsense if guessed
wrong (the raw waveform int values look superficially like they could be
readings, but aren't on a 0-100 scale).

CRITICAL LIMITATION -- confirmed by reading the actual code (data_5cut.py
etc.) and the paper (PMC10899206) directly, not assumed:
  Every _5cut.csv signal is MIN-MAX NORMALIZED TO [0,1] PER SUBJECT, PER
  SIGNAL, before being saved -- and the pre-normalization files (with real
  bpm/percentage values) are NOT included in the downloaded archive. The
  paper's Methods section confirms normalization was applied "to minimize
  the impact of individual variances in tolerance to hypoxic conditions"
  but does not document a way to recover each subject's original min/max.
  This means we CANNOT know subject 218c's true worst SpO2 reading in %;
  we only know it was their personal minimum for that session.

  Rescaling was first tried onto the DEVICE's own documented collection
  range (SpO2/pulse rate 0-100, heart rate 0-250bpm, confirmed from the
  paper) but rejected after inspection: it stretched each subject's actual
  (much narrower) session variation across that entire range, producing
  clinically implausible numbers (e.g. one subject's rescaled mean heart
  rate came out to 174bpm at rest, with SpO2 hitting 0%). Per the project
  owner's decision, this loader instead rescales onto a PHYSIOLOGICALLY
  PLAUSIBLE range for a resting-to-distressed human (SpO2 70-100%, HR/pulse
  40-160bpm -- see SPO2_DEVICE_RANGE etc. below). This is still NOT the
  same as recovering each subject's true individual values -- two subjects
  who both show normalized value 0.0 for spv may have had genuinely
  different true SpO2 readings, since 0.0 means "this subject's own
  minimum for this session," not a specific real percentage. Every row
  loaded through this module is tagged data_source="harespod_rescaled"
  specifically so it is never silently blended with synthetic data (which
  has no such ambiguity) without that distinction being traceable downstream.

  A second real dataset was investigated (src/data/pilot_altitude_loader.py)
  and found to have genuine absolute units with no rescaling needed --
  but it turned out to be an oxygen-ENRICHED protocol (pilots breathing
  ~40-45% O2 throughout, not ambient air), a fundamentally different
  physiological scenario than the ambient-air hypoxia this project models
  (docs/lls_mapping.md). Harespod, by contrast, is confirmed ambient-air
  (pressure reduction only, real oxygen used solely as an emergency
  intervention per its paper's Methods) -- so despite its rescale
  uncertainty, it is the physiologically comparable real dataset, and the
  one actually used for severity labeling (label_real_data.py).

Data Use Agreement (data/raw/harespod/Harespod-main/Data_Use_Agreement.md):
  free to use, but publications using this data must cite the source, and
  it must not be redistributed as part of a new dataset without permission
  from the original research team -- keep this in mind if this project's
  processed/merged data is ever published or shared beyond this repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HARESPOD_DIR, SAMPLE_RATE_HZ

# The paper (PMC10899206) documents the DEVICE's collection range (SpO2 and
# pulse rate 0-100, heart rate 0-250bpm) -- but rescaling onto that full
# range was tried and rejected: it stretches each subject's actual (much
# narrower) session variation across the ENTIRE device range, producing
# clinically implausible numbers (e.g. one subject's rescaled mean HR came
# out to 174bpm at rest, with SpO2 hitting 0%). No public source documents
# each subject's true individual min/max, so an exact recovery is
# impossible (see the module docstring's CRITICAL LIMITATION section).
#
# Instead, per project owner's decision, we rescale onto a PHYSIOLOGICALLY
# PLAUSIBLE range for a resting-to-distressed human -- not the device's
# raw collection bounds. This is explicitly an approximation, not a
# recovered true value: two subjects who both show normalized value 0.0
# for spv may have had genuinely different true SpO2 readings, but at
# least the resulting numbers land somewhere a real pulse oximeter would
# plausibly report, rather than spanning a device's full theoretical range.
SPO2_DEVICE_RANGE = (70.0, 100.0)   # spv (SpO2 Sat) -- healthy-to-severely-hypoxic band
PULSE_RATE_DEVICE_RANGE = (40.0, 160.0)  # prt (Pulse Rate) -- resting-to-strenuous band
HEART_RATE_DEVICE_RANGE = (40.0, 160.0)  # hr (Heart Rate) -- resting-to-strenuous band

# The six wall-clock altitude-stage markers found in every subject's
# key_timestamp.txt, confirmed identical (in meters) across all 15 subjects.
ALTITUDE_STAGE_METERS = [1500, 2000, 2500, 3000, 3500, 4000]

TIDY_COLUMNS = ["timestamp", "spo2", "hr", "altitude", "subject_id", "data_source"]


def has_harespod_data() -> bool:
    """Cheap existence check -- True once Data_Cons/<subject>/ folders are present."""
    cons_dir = HARESPOD_DIR / "Data_Cons"
    if not cons_dir.exists():
        return False
    return any(p.is_dir() for p in cons_dir.iterdir())


def _raise_not_downloaded() -> None:
    raise FileNotFoundError(
        "Harespod data not found in data/raw/harespod/Data_Cons/.\n\n"
        "This is expected if you haven't downloaded it yet -- it's a manual "
        "browser download, not something this pipeline fetches automatically.\n\n"
        "To get it:\n"
        "  1. Visit https://doi.org/10.6084/m9.figshare.c.6623344.v1\n"
        "  2. Download the dataset archive and unzip its contents into\n"
        f"     {HARESPOD_DIR}\n"
        "  3. Companion loading code: https://github.com/oca-john/Harespod\n\n"
        "Until then, the rest of the pipeline runs on the synthetic generator "
        "(src/data/synth_data.py) instead -- see CLAUDE.md Section 5."
    )


def list_subject_ids() -> list[str]:
    """Every usable subject folder name under Data_Cons/, e.g. ['217a', '218c', ...]."""
    if not has_harespod_data():
        _raise_not_downloaded()
    cons_dir = HARESPOD_DIR / "Data_Cons"
    return sorted(p.name for p in cons_dir.iterdir() if p.is_dir())


def _rescale(normalized: pd.Series, device_range: tuple[float, float]) -> pd.Series:
    """
    Map a [0,1]-normalized signal onto the device's documented collection
    range. See the module docstring's CRITICAL LIMITATION -- this is an
    approximation using the DEVICE's known range, not a recovery of each
    subject's true individual min/max, which isn't preserved in the
    downloaded archive.
    """
    lo, hi = device_range
    return lo + normalized.clip(0, 1) * (hi - lo)


def _parse_key_timestamps(subject_dir: Path) -> list[tuple[str, int]]:
    """
    Parse key_timestamp.txt's `t1 = '...'  # 1500 / 1.27` lines into
    (wall_clock_str, altitude_m) pairs, in order. Regex-based rather than
    a fixed line-number read, since the file has variable leading comment
    lines -- this is robust to those comments changing without needing an
    update here.

    Two documented exceptions found by inspecting all 15 files directly:

    1. Subject 314c's t1 line reads `# Start to Rec` instead of
       `# 1500 / 1.27` -- every other subject's t1 marks the 1500m stage,
       and 314c's own "1.5k-2k Time" line below confirms its t1 means the
       same thing, just with different comment text. Filled in as 1500m.

    2. Subject 328a never reached the 4000m stage: its t6 line reads
       `t6 = '0000'      # Did not rise to this altitude` -- '0000' is a
       placeholder, not a real timestamp. This subject's protocol
       genuinely only has 5 real stages (1500-3500m), so this marker is
       DROPPED rather than forced to a 6th fake entry -- callers get back
       5 markers for this subject and interpolate/hold at 3500m for any
       reading after it, instead of extrapolating toward an altitude the
       subject never actually reached.

    Both are handled explicitly here rather than silently regex-matching
    around them, so these known exceptions stay visible instead of looking
    like coincidentally-correct behavior.
    """
    kt_path = subject_dir / "key_timestamp.txt"
    text = kt_path.read_text(encoding="utf-8")
    # Matches `# 1500 / 1.27` (normal), `# Start to Rec` (314c's t1
    # wording), or `# Did not rise to this altitude` (328a's dropped t6).
    pattern = re.compile(
        r"^t\d+\s*=\s*'([^']*)'\s*#\s*"
        r"(?:(\d+)(?:\s*/\s*[\d.]+)?|Start to Rec|Did not rise to this altitude)",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    # Drop any "Did not rise to this altitude" entries -- these have an
    # empty ts capture too ('0000' isn't parseable as a real timestamp so
    # we don't even want to try), identified by having no altitude digits
    # AND not being the "Start to Rec" case (which legitimately means 1500m).
    parsed = []
    for ts, alt in matches:
        if not alt and ts == "0000":
            continue  # the "never reached this altitude" case -- drop it
        parsed.append((ts, int(alt) if alt else ALTITUDE_STAGE_METERS[0]))

    expected_max = len(ALTITUDE_STAGE_METERS)
    if not (expected_max - 1 <= len(parsed) <= expected_max):
        raise ValueError(
            f"{kt_path} has {len(parsed)} usable timestamp markers, expected "
            f"{expected_max - 1} or {expected_max} ({ALTITUDE_STAGE_METERS}). The file "
            "format may differ for this subject -- inspect it directly before "
            "trusting this subject's altitude interpolation."
        )
    return parsed


def interpolate_altitude(timestamps: pd.Series, subject_dir: Path) -> pd.Series:
    """
    Build a continuous altitude-in-meters column for one subject by
    linearly interpolating between key_timestamp.txt's known chamber-stage
    markers. Harespod is a hypobaric CHAMBER study -- altitude changes in
    discrete operator-triggered steps, not a continuous ascent -- so
    "continuous altitude" is itself an approximation of what happened
    between two known points, same reasoning as in feature_engineering's
    synthetic ascent ramps.

    `timestamps` before the first marker are held at the first known
    altitude (0m is a reasonable prior -- the subject was at ambient
    pressure before the chamber started ramping); after the last marker,
    altitude is held at the final stage (4000m) since Harespod's protocol
    holds a steady period there rather than descending within the same
    recording.

    GMT-vs-local-time correction: every key_timestamp.txt's own header
    warns "device record time may be GMT, which is 8H different from local
    time" -- and checking all 15 subjects directly confirmed 6 of them
    (318a, 318b, 318c, 321a, 328a, 328b) actually have their markers
    recorded exactly 8 hours AHEAD of the vitals data's own timestamps,
    while the other 9 have no offset at all. Rather than hardcode that
    subject list (fragile if new subjects are ever added), this detects
    the offset per-subject: compare the first marker time to the data's
    own first timestamp, and if they're ~8 hours apart, shift the markers
    back by that amount before interpolating. Without this, those 6
    subjects' altitude would be computed from timestamps that don't
    overlap the vitals data at all, silently producing a constant 4000m
    (or 1500m) for the whole recording instead of a real ramp.
    """
    markers = _parse_key_timestamps(subject_dir)
    marker_times = pd.to_datetime([m[0] for m in markers])
    marker_alts = np.array([m[1] for m in markers], dtype=float)

    data_start = pd.to_datetime(timestamps).min()
    offset_hours = (marker_times[0] - data_start).total_seconds() / 3600
    if abs(offset_hours - 8) < 0.1:
        marker_times = marker_times - pd.Timedelta(hours=8)
    elif abs(offset_hours) > 0.1:
        raise ValueError(
            f"{subject_dir}: key_timestamp.txt markers are {offset_hours:.2f}h "
            "offset from the vitals data's own timestamps -- expected either "
            "~0h (no offset) or ~8h (the documented GMT/local-time case). "
            "Inspect this subject's files directly before trusting its altitude."
        )

    marker_times_epoch = marker_times.astype("int64") / 1e9  # -> unix seconds
    ts_seconds = pd.to_datetime(timestamps).astype("int64") / 1e9
    return pd.Series(
        np.interp(ts_seconds, marker_times_epoch, marker_alts),
        index=timestamps.index,
        name="altitude",
    )


def load_subject(subject_id: str) -> pd.DataFrame:
    """
    Load one subject's Harespod recording as a tidy 1Hz DataFrame, matching
    TIDY_COLUMNS -- the same shape synth_data.py produces, so
    feature_engineering.py can treat both sources identically (see
    build_processed_dataset in that module).

    Uses spv (SpO2 Sat) for spo2 and hr (Heart Rate) directly -- NOT the
    100Hz spo/rsp waveform files, which aren't percentages/rates at all
    (see module docstring). prt (pulse rate) and rsp (respiration
    waveform) are loaded but not currently surfaced in TIDY_COLUMNS, since
    the rest of this project's feature set (config.FEATURE_COLUMNS) has no
    slot for them yet -- available for a future extension if useful.
    """
    if not has_harespod_data():
        _raise_not_downloaded()

    subject_dir = HARESPOD_DIR / "Data_Cons" / subject_id
    if not subject_dir.exists():
        available = list_subject_ids()
        raise FileNotFoundError(
            f"No Harespod subject folder at {subject_dir}. "
            f"Available subjects: {available}"
        )

    hr_df = pd.read_csv(subject_dir / "hr_5cut.csv", header=None, names=["timestamp", "hr_norm"])
    spv_df = pd.read_csv(subject_dir / "spv_5cut.csv", header=None, names=["timestamp", "spv_norm"])

    hr_df["timestamp"] = pd.to_datetime(hr_df["timestamp"])
    spv_df["timestamp"] = pd.to_datetime(spv_df["timestamp"])

    # hr and spv are recorded by the same device in the same session and
    # both land on whole-second timestamps (verified: both files span the
    # identical wall-clock range with strict 1-second spacing), so an inner
    # merge on timestamp aligns them without needing resampling/interpolation.
    merged = pd.merge(hr_df, spv_df, on="timestamp", how="inner")
    if merged.empty:
        raise ValueError(
            f"hr_5cut.csv and spv_5cut.csv for subject {subject_id} share no "
            "common timestamps -- unexpected given both come from the same "
            "recording session. Inspect the raw files before trusting this subject."
        )

    merged["spo2"] = _rescale(merged["spv_norm"], SPO2_DEVICE_RANGE)
    merged["hr"] = _rescale(merged["hr_norm"], HEART_RATE_DEVICE_RANGE)
    merged["altitude"] = interpolate_altitude(merged["timestamp"], subject_dir)

    # timestamp as seconds-from-recording-start, matching synth_data.py's
    # convention (0.0, 1.0, 2.0, ...) rather than absolute wall-clock time --
    # feature_engineering.py's rolling windows assume this convention.
    start = merged["timestamp"].min()
    merged["timestamp"] = (merged["timestamp"] - start).dt.total_seconds()

    merged["subject_id"] = f"harespod_{subject_id}"
    merged["data_source"] = "harespod_rescaled"

    return merged[TIDY_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def load_all_subjects() -> pd.DataFrame:
    """Load and concatenate every usable Harespod subject recording."""
    subject_ids = list_subject_ids()
    frames = [load_subject(sid) for sid in subject_ids]
    return pd.concat(frames, ignore_index=True)
