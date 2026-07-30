"""
HarespodUploadDataSource -- replay a Harespod-format recording for ONE
patient dropped in via the Streamlit file uploader, instead of picking
from the 15 subjects already present in data/raw/harespod/.

Why this exists: the full Harespod archive is a ~905MB manual download,
correctly excluded from this repo (and from Streamlit Cloud, which has no
volume to keep it in). Someone who has a single subject's files locally
(their own hypobaric-chamber session, or one Harespod subject folder
copied out) shouldn't need the whole archive just to replay that one
recording -- this lets them upload exactly the 3 files
`harespod_loader.load_subject()` actually reads and get an instant replay.

Design: rather than duplicate or modify harespod_loader.py's already-built
and already-tested parsing/rescaling/altitude-interpolation logic, this
module writes the uploaded files into a throwaway temp directory and then
calls harespod_loader's own path-parameterized helpers (`_rescale()`,
`interpolate_altitude()`, both of which already accept an explicit
subject_dir rather than reading the module-level HARESPOD_DIR global)
directly against that temp directory. Deliberately NOT done by reassigning
`harespod_loader.HARESPOD_DIR` itself: that's a shared module-level global,
and Streamlit can serve multiple users' sessions from the same running
process, so mutating a global (even temporarily, even restored in a
finally block) would be a real race condition if two uploads landed at
the same moment -- one session's temp path could leak into another's
read. Working only through the already-path-parameterized helpers avoids
that risk entirely, at the cost of re-doing `load_subject()`'s small
merge/rescale sequence here instead of calling it directly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import harespod_loader
from src.data.synth_data import TEMP_NORMAL_RANGE
from src.datasource.base import DataSource, Reading

# The exact 3 files harespod_loader.load_subject() reads for one subject --
# see that function's body: hr_5cut.csv, spv_5cut.csv (both bare two-column
# CSVs, no header), and key_timestamp.txt (the chamber-stage altitude
# markers). Nothing else is required -- prt/rsp/spo exist in the real
# archive but are documented as unused by this project.
REQUIRED_FILENAMES = ["hr_5cut.csv", "spv_5cut.csv", "key_timestamp.txt"]


def load_uploaded_subject(uploaded_files: dict[str, bytes]) -> pd.DataFrame:
    """
    uploaded_files: {filename: raw_bytes} for exactly the 3 REQUIRED_FILENAMES
    (Streamlit's UploadedFile.getvalue() gives raw bytes per file).

    Returns the same tidy DataFrame shape harespod_loader.load_subject()
    always returns -- this mirrors that function's own body line for line
    (merge hr+spv on timestamp, rescale, interpolate altitude, convert to
    seconds-from-start, tag columns), just reading from a temp directory
    via explicit paths instead of HARESPOD_DIR / "Data_Cons" / subject_id,
    so every rescale/interpolation/bug-fix already validated in
    harespod_loader.py applies identically to an uploaded subject.
    """
    missing = [f for f in REQUIRED_FILENAMES if f not in uploaded_files]
    if missing:
        raise ValueError(
            f"Missing required file(s): {missing}. Need exactly {REQUIRED_FILENAMES} "
            "for one subject -- the same 3 files harespod_loader.py reads from a real "
            "Data_Cons/<subject_id>/ folder."
        )

    # A fresh temp dir per call (not reused across uploads, never shared
    # across sessions) so two different uploaded subjects -- even in two
    # concurrent Streamlit sessions on the same server process -- can never
    # collide or leak into each other's files.
    tmp_dir = Path(tempfile.mkdtemp(prefix="harespod_upload_"))
    try:
        for filename in REQUIRED_FILENAMES:
            (tmp_dir / filename).write_bytes(uploaded_files[filename])

        # From here down: the same sequence as harespod_loader.load_subject(),
        # just parameterized on tmp_dir instead of a Data_Cons/<id> path.
        hr_df = pd.read_csv(tmp_dir / "hr_5cut.csv", header=None, names=["timestamp", "hr_norm"])
        spv_df = pd.read_csv(tmp_dir / "spv_5cut.csv", header=None, names=["timestamp", "spv_norm"])
        hr_df["timestamp"] = pd.to_datetime(hr_df["timestamp"])
        spv_df["timestamp"] = pd.to_datetime(spv_df["timestamp"])

        merged = pd.merge(hr_df, spv_df, on="timestamp", how="inner")
        if merged.empty:
            raise ValueError(
                "The uploaded hr_5cut.csv and spv_5cut.csv share no common "
                "timestamps -- they may not be from the same recording session. "
                "Inspect the raw files before trusting this upload."
            )

        merged["spo2"] = harespod_loader._rescale(merged["spv_norm"], harespod_loader.SPO2_DEVICE_RANGE)
        merged["hr"] = harespod_loader._rescale(merged["hr_norm"], harespod_loader.HEART_RATE_DEVICE_RANGE)
        merged["altitude"] = harespod_loader.interpolate_altitude(merged["timestamp"], tmp_dir)

        start = merged["timestamp"].min()
        merged["timestamp"] = (merged["timestamp"] - start).dt.total_seconds()

        merged["subject_id"] = "harespod_uploaded"
        merged["data_source"] = "harespod_rescaled"

        return merged[harespod_loader.TIDY_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # scratch directory, safe to discard the instant we're done with it


class HarespodUploadDataSource(DataSource):
    """
    Same DataSource contract (next_reading()/reset()) as
    HarespodReplayDataSource, but built from an already-loaded, already-
    validated DataFrame (from load_uploaded_subject()) instead of reading
    subject_id off disk -- the upload happens once, in the Streamlit
    sidebar, before this object is even constructed.
    """

    def __init__(self, df: pd.DataFrame, seed: int | None = None):
        self._df = df
        self._seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        self._readings: list[Reading] = []
        self._index = 0
        self.reset()

    def _load(self) -> None:
        # Same neutral (non-severity-conditioned) temperature synthesis as
        # HarespodReplayDataSource -- see that module's docstring for why:
        # no real temp sensor exists in Harespod, and there's no severity
        # label to condition on ahead of time during a live replay.
        rng = np.random.default_rng(self._seed)
        temp_center = rng.uniform(*TEMP_NORMAL_RANGE)
        temp_noise = rng.normal(0, 0.15, len(self._df))
        df = self._df.assign(temp=temp_center + temp_noise)
        self._readings = df[["timestamp", "spo2", "hr", "temp", "altitude"]].to_dict(
            orient="records"
        )

    def next_reading(self) -> Reading | None:
        if self._index >= len(self._readings):
            return None
        reading = self._readings[self._index]
        self._index += 1
        return reading

    def reset(self) -> None:
        if not self._readings:
            self._load()
        self._index = 0

    def __len__(self) -> int:
        return len(self._readings)

    @property
    def progress(self) -> float:
        if not self._readings:
            return 0.0
        return self._index / len(self._readings)
