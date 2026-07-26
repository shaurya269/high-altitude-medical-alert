"""
ManualDataSource -- the Demo Mode "manual sliders" panel from CLAUDE.md
Section 2: lets the user directly set SpO2/HR/temp/altitude and see the
ML classification + LLM interpretation update live, for exploring edge
cases (e.g. "what does the system say at exactly 85% SpO2, 4000m?").

Unlike ScenarioDataSource (which pre-generates a whole trajectory) and the
future replay source (which steps through a fixed recording), this source
has no notion of "the next reading" beyond whatever the caller last set --
`next_reading()` just returns the current slider values, stamped with a
freshly-advanced timestamp. This still satisfies the DataSource contract
(callers never need to know this source works differently internally),
which is exactly the point of building against the interface rather than
letting app/ code special-case "oh, this one's manual."
"""

from __future__ import annotations

from src.config import SAMPLE_RATE_HZ
from src.datasource.base import DataSource, Reading

# Reasonable slider defaults -- a healthy reading at a moderate altitude,
# so the demo doesn't open already showing an alarming state.
DEFAULT_SPO2 = 96.0
DEFAULT_HR = 75.0
DEFAULT_TEMP = 36.8
DEFAULT_ALTITUDE = 2000.0


class ManualDataSource(DataSource):
    """
    `set_values()` is called by the Streamlit slider callbacks; every
    subsequent `next_reading()` call returns whatever was last set,
    advancing only the timestamp. This lets the live pipeline (feature
    engineering's rolling windows in particular) build up a real buffer of
    "this value held roughly steady, then the user moved a slider" history,
    which is exactly what makes manual override useful for exploring
    trend-sensitive behavior (e.g. dragging SpO2 down over several ticks to
    see spo2_trend_5min react), not just single-point classification.
    """

    def __init__(self):
        self._timestamp = 0.0
        self._spo2 = DEFAULT_SPO2
        self._hr = DEFAULT_HR
        self._temp = DEFAULT_TEMP
        self._altitude = DEFAULT_ALTITUDE

    def set_values(
        self,
        spo2: float | None = None,
        hr: float | None = None,
        temp: float | None = None,
        altitude: float | None = None,
    ) -> None:
        """Update any subset of the four sliders -- None means "leave unchanged."""
        if spo2 is not None:
            self._spo2 = spo2
        if hr is not None:
            self._hr = hr
        if temp is not None:
            self._temp = temp
        if altitude is not None:
            self._altitude = altitude

    def next_reading(self) -> Reading | None:
        reading: Reading = {
            "timestamp": self._timestamp,
            "spo2": self._spo2,
            "hr": self._hr,
            "temp": self._temp,
            "altitude": self._altitude,
        }
        self._timestamp += 1.0 / SAMPLE_RATE_HZ
        return reading

    def reset(self) -> None:
        self._timestamp = 0.0
        self._spo2 = DEFAULT_SPO2
        self._hr = DEFAULT_HR
        self._temp = DEFAULT_TEMP
        self._altitude = DEFAULT_ALTITUDE
