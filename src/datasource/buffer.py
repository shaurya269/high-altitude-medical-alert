"""
ReadingBuffer -- Stage 2 of the data flow diagram ("Buffer & Validate"):
sits between any DataSource and the feature engineering / ML stages,
maintaining the trailing ROLLING_BUFFER_MINUTES of readings and rejecting
out-of-range glitches before they ever reach engineer_features() or a
trained model.

This is the one piece of "live pipeline glue" that's genuinely new in Day
8 -- every DataSource (scenario, manual, replay, and eventually
arduino_reader.py) feeds through the SAME buffer, so validation and
windowing logic exists in exactly one place regardless of where readings
originate. Stages 3+ (feature engineering, ML classification) always
operate on `buffer.as_dataframe()`, never on a raw DataSource reading
directly -- a single reading has no trend to compute a spo2_trend_5min
from.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from src.config import ROLLING_BUFFER_MINUTES, SAMPLE_RATE_HZ
from src.datasource.base import Reading

# Physiologically-impossible bounds -- NOT the same as "abnormal" (85%
# SpO2 is abnormal but real; -5% SpO2 is a sensor glitch). This is Stage
# 2's "validate ranges (reject glitches/out-of-range values)" per the data
# flow diagram -- a coarse sanity filter, not a clinical judgment. The
# actual severity classification (Stage 4) is what judges "abnormal but
# plausible" readings; this stage only catches "not physically possible."
VALID_RANGES = {
    "spo2": (0.0, 100.0),
    "hr": (20.0, 300.0),
    "temp": (25.0, 45.0),
    "altitude": (-500.0, 9000.0),  # Dead Sea to just below Everest's summit
}

BUFFER_MAXLEN = ROLLING_BUFFER_MINUTES * 60 * SAMPLE_RATE_HZ


class ReadingBuffer:
    """
    A fixed-size trailing window of validated readings. `deque(maxlen=...)`
    automatically drops the oldest reading once full -- exactly the
    "rolling" behavior Stage 2 needs, with no manual trimming logic to get
    wrong.
    """

    def __init__(self, maxlen: int = BUFFER_MAXLEN):
        self._readings: deque[Reading] = deque(maxlen=maxlen)
        self.rejected_count = 0

    def add(self, reading: Reading) -> bool:
        """
        Validate and append one reading. Returns True if accepted, False
        if rejected as out-of-range (caller can log/surface this -- e.g.
        the dashboard flagging "N glitched readings dropped" -- without
        this method raising and interrupting the live loop over what's
        expected to be an occasional, tolerable event, not an exceptional
        one).
        """
        for field, (lo, hi) in VALID_RANGES.items():
            value = reading.get(field)
            if value is None or not (lo <= value <= hi):
                self.rejected_count += 1
                return False
        self._readings.append(reading)
        return True

    def as_dataframe(self) -> pd.DataFrame:
        """
        The buffer's contents as a DataFrame sorted by timestamp -- the
        shape engineer_features() and every model's predict() expect.
        Empty (not an error) if nothing has been added yet; callers should
        check `len(buffer) > 0` before feeding it to feature engineering,
        same as any other "not enough history yet" case in this pipeline.
        """
        return pd.DataFrame(self._readings).sort_values("timestamp").reset_index(drop=True)

    def clear(self) -> None:
        self._readings.clear()
        self.rejected_count = 0

    def __len__(self) -> int:
        return len(self._readings)

    @property
    def is_full(self) -> bool:
        return len(self._readings) == self._readings.maxlen
