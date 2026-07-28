"""
HarespodReplayDataSource -- replays a real Harespod recording end to end
through the live pipeline (CLAUDE.md Section 2: "Replay a real Harespod
recording end to end"). This is what lets the demo show the exact same
Stage 4-7 pipeline (ML classification, hysteresis, LLM interpretation,
alerts) responding to genuinely real sensor data, not just synthetic
scenarios -- a meaningfully different demo story ("here's what the system
does with an actual hypobaric chamber recording") from the scenario player.

Reuses harespod_loader.py's already-built, already-tested loading +
rescaling + altitude-interpolation logic rather than re-reading the raw
CSVs here -- this module's only job is turning one subject's loaded
DataFrame into the DataSource.next_reading() contract. No temperature
column exists in Harespod's raw data (see harespod_loader.py's docstring),
so this source synthesizes one per reading using the SAME literature-based
per-tier distributions used elsewhere for that gap -- but since a replay
has no severity label to condition on ahead of time (that's what the LIVE
pipeline is being demonstrated computing), it draws from the "Normal" tier
profile as a neutral prior. This is a cosmetic fill-in for a genuinely
missing sensor, not a signal fed into classification.
"""

from __future__ import annotations

import numpy as np

from src.data import harespod_loader
from src.data.synth_data import TEMP_NORMAL_RANGE
from src.datasource.base import DataSource, Reading


class HarespodReplayDataSource(DataSource):
    def __init__(self, subject_id: str, seed: int | None = None):
        if not harespod_loader.has_harespod_data():
            raise FileNotFoundError(
                "Harespod data not downloaded -- HarespodReplayDataSource requires it. "
                "See harespod_loader.py's docstring for download steps, or use "
                "ScenarioDataSource / ManualDataSource instead, which need no download."
            )
        self.subject_id = subject_id
        self._seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        self._readings: list[Reading] = []
        self._index = 0
        self.reset()

    def _load(self) -> None:
        df = harespod_loader.load_subject(self.subject_id)  # already rescaled/altitude-interpolated by the loader
        rng = np.random.default_rng(self._seed)  # seeded -> same synthesized temp column every time this subject+seed replays
        # See module docstring: no real temp sensor exists, so this is a
        # neutral (not severity-conditioned) synthesized fill-in, distinct
        # from label_real_data.py's training-time temp synthesis which DOES
        # condition on a known severity label -- there is no such label
        # available here since replay is meant to demonstrate the live
        # classification pipeline computing one, not assume it in advance.
        temp_center = rng.uniform(*TEMP_NORMAL_RANGE)  # one fixed baseline for the whole recording, drawn from the normal range
        temp_noise = rng.normal(0, 0.15, len(df))  # small per-reading jitter (sigma=0.15C) so temp isn't perfectly flat
        df = df.assign(temp=temp_center + temp_noise)  # add the synthesized column without mutating the loader's original df
        self._readings = df[["timestamp", "spo2", "hr", "temp", "altitude"]].to_dict(
            orient="records"
        )

    def next_reading(self) -> Reading | None:
        if self._index >= len(self._readings):  # recording exhausted -- signal end per the DataSource contract
            return None
        reading = self._readings[self._index]
        self._index += 1  # advance the simulated clock by one tick
        return reading

    def reset(self) -> None:
        if not self._readings:  # only load once per instance -- reset() rewinds, it doesn't reload
            self._load()
        self._index = 0

    def __len__(self) -> int:
        return len(self._readings)

    @property
    def progress(self) -> float:
        if not self._readings:
            return 0.0
        return self._index / len(self._readings)

    @staticmethod
    def available_subjects() -> list[str]:
        """Subject IDs the UI can offer in a replay picker, e.g. ['217a', '218c', ...]."""
        if not harespod_loader.has_harespod_data():
            return []
        return harespod_loader.list_subject_ids()
