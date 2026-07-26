"""
ScenarioDataSource -- plays a pre-built severity trajectory forward one
reading at a time, for the Demo Mode "pick a scenario and watch it play in
real time" panel (CLAUDE.md Section 2: Normal / Mild AMS / Severe AMS /
HAPE onset / HACE risk).

Deliberately reuses synth_data.simulate_trajectory() rather than writing a
second trajectory generator -- that function already encodes the clinical
reasoning from docs/lls_mapping.md (gradual sigmoid onset, per-tier
SpO2/HR/temp profiles) and is unit-tested (tests/test_pipeline.py). A demo
scenario and a training-data trajectory should look like the same kind of
data, since the whole point of Demo Mode is exercising the REAL pipeline
end to end, not a simplified stand-in for it.
"""

from __future__ import annotations

import numpy as np

from src.config import SEVERITY_TIERS
from src.data.synth_data import simulate_trajectory
from src.datasource.base import DataSource, Reading

# Scenario names shown in the UI picker, mapped to the severity tier that
# drives simulate_trajectory()'s generation. "HAPE onset" / "HACE risk" use
# CLAUDE.md's own wording (Section 2) even though the underlying tier name
# in config.SEVERITY_TIERS is "HAPE risk" -- kept as an explicit mapping so
# a future rename of either doesn't silently break the other.
SCENARIOS: dict[str, str] = {
    "Normal": "Normal",
    "Mild AMS": "Mild AMS",
    "Severe AMS": "Severe AMS",
    "HAPE onset": "HAPE risk",
    "HACE risk": "HACE risk",
}


class ScenarioDataSource(DataSource):
    """
    Generates one fixed trajectory at construction time (so re-reading the
    same tick twice, e.g. if the UI re-renders, returns identical data --
    important for Streamlit specifically, whose rerun model would
    otherwise regenerate a DIFFERENT random trajectory on every widget
    interaction) and steps through it one reading per `next_reading()` call.
    """

    def __init__(self, scenario_name: str, duration_minutes: int = 90, seed: int | None = None):
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario_name!r}. Choose from: {list(SCENARIOS)}"
            )
        self.scenario_name = scenario_name
        self._tier = SCENARIOS[scenario_name]
        self._duration_minutes = duration_minutes
        # A random (not fixed) seed by default -- so replaying the "Severe
        # AMS" scenario twice in a session shows different-but-plausible
        # trajectories, matching how synth_data.py generates natural
        # within-class variation for training. Pass an explicit seed for
        # reproducible demos/tests.
        self._seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        self._trajectory = None
        self._index = 0
        self.reset()

    def _generate(self) -> None:
        rng = np.random.default_rng(self._seed)
        df = simulate_trajectory(
            subject_id=f"demo_{self.scenario_name.replace(' ', '_')}",
            tier=self._tier,
            duration_minutes=self._duration_minutes,
            rng=rng,
        )
        self._trajectory = df[["timestamp", "spo2", "hr", "temp", "altitude"]].to_dict(
            orient="records"
        )

    def next_reading(self) -> Reading | None:
        if self._index >= len(self._trajectory):
            return None
        reading = self._trajectory[self._index]
        self._index += 1
        return reading

    def reset(self) -> None:
        if self._trajectory is None:
            self._generate()
        self._index = 0

    def __len__(self) -> int:
        return len(self._trajectory)

    @property
    def progress(self) -> float:
        """Fraction of the trajectory consumed so far, in [0, 1] -- for a UI progress bar."""
        if not self._trajectory:
            return 0.0
        return self._index / len(self._trajectory)
