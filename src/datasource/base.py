"""
The DataSource interface -- the one abstraction the rest of the pipeline
(feature engineering, ML classification, hysteresis gate, LLM
interpretation, Telegram alerts) is built against, per CLAUDE.md's
explicit instruction: "Don't hardcode the data source -- always go through
the DataSource interface so the future hardware phase is a drop-in swap,
not a rewrite."

Why an abstract base class at all, rather than just writing three separate
functions: every consumer downstream (the Streamlit app's live loop,
Stages 2-7 of the data flow diagram) needs to treat "where does this
reading come from" as a solved, swappable question. Today there are three
Demo Mode sources (scenario player, manual override, Harespod replay) and
a future hardware source (arduino_reader.py, NOT built in this phase per
CLAUDE.md Section 2A). All four need to expose the IDENTICAL
`.next_reading()` contract so app/ code never has an if/elif on "which
source am I talking to."

Reading shape (matches Stage 1 of the data flow diagram exactly):
    {"timestamp": float, "spo2": float, "hr": float, "temp": float, "altitude": float}
This is deliberately a plain dict, not a custom class -- it's the exact
row shape pandas.DataFrame.from_records() and feature_engineering.py's
per-subject frames already expect (see synth_data.py / harespod_loader.py's
TIDY_COLUMNS), so a buffer of readings converts to a DataFrame with zero
translation step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Reading(TypedDict):
    """One sensor sample -- the atomic unit every DataSource implementation emits."""

    timestamp: float
    spo2: float
    hr: float
    temp: float
    altitude: float


class DataSource(ABC):
    """
    Abstract base every data source (demo or, in a future phase, real
    hardware) must implement. Two methods only, kept deliberately minimal:

    - `next_reading()`: advance the simulated/replayed clock by one tick
      and return the new reading. Returns None when the source is
      exhausted (e.g. a replay reaching the end of its recording) --
      callers must handle this, not assume readings are infinite.
    - `reset()`: rewind to the start. Needed for the Streamlit UI's
      "restart this scenario" button without recreating the whole object
      (and losing whatever's subscribed to it, once that matters).

    A future `arduino_reader.py` (Phase 3, NOT built now -- see CLAUDE.md
    Section 2A) implements this same interface over a live pyserial
    connection; `next_reading()` would block briefly on the next serial
    packet instead of advancing a simulated clock, and `reset()` would be
    a no-op (you can't rewind a live sensor) or could raise
    NotImplementedError -- that decision is deferred to when hardware
    integration actually starts.
    """

    @abstractmethod
    def next_reading(self) -> Reading | None:
        """Return the next reading, or None if the source is exhausted."""

    @abstractmethod
    def reset(self) -> None:
        """Rewind this source back to its starting state."""
