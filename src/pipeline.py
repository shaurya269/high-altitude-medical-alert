"""
The full sensor-to-alert pipeline, Stages 1-7 of the data flow diagram,
wired together in one place. This is Day 11's "main loop": every other
module built in Days 1-10 (a DataSource, ReadingBuffer, engineer_features,
predict_severity, HysteresisGate, llm_chat, alert_bot) is a piece; this
module is what calls them in the right order, once per reading, for one
monitored subject/session.

Deliberately a plain class with a `.tick()` method, not a `while True`
loop with a `time.sleep()` inside it -- Streamlit (Day 12) needs to call
one tick per rerun/timer-interval itself, and a live hardware phase
(Phase 3, not built now) would drive its own loop against a real serial
connection. This class has NO opinion about what drives it or how fast;
it just does the right sequence of work for one reading and returns a
result the caller can display.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.alerts.alert_bot import send_alert
from src.alerts.hysteresis import HysteresisGate
from src.data.feature_engineering import engineer_features
from src.datasource.base import DataSource, Reading
from src.datasource.buffer import ReadingBuffer
from src.llm.llm_chat import explain_severity, generate_alert_content
from src.models.predict_severity import predict_severity


@dataclass
class TickResult:
    """
    Everything one `.tick()` call produced -- the dashboard (Day 12)
    renders directly from this, and it's also what Day 13's testing pass
    inspects to verify the pipeline behaves correctly end to end.
    """

    reading: Reading | None  # None if the source was exhausted this tick
    accepted: bool  # False if the buffer rejected this reading as out-of-range
    severity: dict | None  # predict_severity()'s output, or None if buffer isn't full enough yet
    explanation: str | None  # LLM interpretation, Stage 6A -- always computed once severity exists
    alert_fired: bool  # whether the hysteresis gate passed THIS tick
    alert_result: dict | None  # alert_bot.send_alert()'s return, only set if alert_fired
    exhausted: bool  # True once the DataSource has no more readings (replay/scenario ended)


class MedicalAlertPipeline:
    """
    One instance per monitored subject/session. Owns the buffer and the
    hysteresis gate (both genuinely stateful across ticks -- see
    hysteresis.py's docstring on why a fresh gate per call would never
    fire), and drives one DataSource through the full Stage 1-7 sequence.

    Swapping which DataSource is plugged in (scenario, manual, replay, or
    a future arduino_reader.py) requires zero changes here -- this class
    only calls `.next_reading()`, never anything source-specific. That's
    the entire point of building against the DataSource interface
    (CLAUDE.md: "always go through the DataSource interface").
    """

    def __init__(self, source: DataSource, min_readings_for_classification: int = 60):
        self.source = source
        self.buffer = ReadingBuffer()
        self.gate = HysteresisGate()
        # XGBoost (the currently-selected model, see predict_severity.py)
        # doesn't strictly need a full 60s window the way the LSTM would,
        # but requiring SOME minimum history before the first
        # classification avoids a meaningless prediction off a single
        # reading with a zero trend/ascent-rate (every trend feature is 0
        # until enough history exists to compute one) -- see
        # feature_engineering.engineer_features' growing-window behavior.
        self.min_readings_for_classification = min_readings_for_classification

        # Caches the last LLM explanation + which severity tier it was
        # for. Re-generating a fresh explanation on EVERY tick would mean
        # one Groq API call per second in a live 1Hz loop -- impractical
        # cost/latency/rate-limit-wise, and pointless anyway, since the
        # explanation for "still Severe AMS" doesn't need to change just
        # because another second passed with the same classification.
        # "Stage 6A - ALWAYS" in the data flow diagram means "always
        # eligible, independent of the hysteresis gate" (contrasted with
        # Stage 6B's conditional alert dispatch) -- not "regenerate at
        # literal 1Hz regardless of cost." A tier CHANGE always gets a
        # fresh explanation; an unchanged tier reuses the cached one.
        self._last_explained_tier: int | None = None
        self._cached_explanation: str | None = None

    def tick(self) -> TickResult:
        """Advance by exactly one reading and run it through the full pipeline."""
        reading = self.source.next_reading()
        if reading is None:
            return TickResult(
                reading=None,
                accepted=False,
                severity=None,
                explanation=None,
                alert_fired=False,
                alert_result=None,
                exhausted=True,
            )

        accepted = self.buffer.add(reading)  # Stage 2: rolling buffer; False = reading was out-of-range and rejected, not stored

        if len(self.buffer) < self.min_readings_for_classification:
            return TickResult(
                reading=reading,
                accepted=accepted,
                severity=None,
                explanation=None,
                alert_fired=False,
                alert_result=None,
                exhausted=False,
            )

        # Stage 3-4: feature engineering + ML classification
        featured = engineer_features(self.buffer.as_dataframe())
        severity = predict_severity(featured)

        # Stage 6A: LLM interpretation. Only regenerated when the severity
        # TIER changes (see the cache fields' docstring in __init__) --
        # still "always available/eligible" independent of the hysteresis
        # gate, just not re-called on every identical-tier tick.
        if severity["severity_index"] != self._last_explained_tier:
            self._cached_explanation = explain_severity(
                severity["severity_label"], severity["confidence"], reading
            )
            self._last_explained_tier = severity["severity_index"]
        explanation = self._cached_explanation

        # Stage 5: hysteresis gate
        alert_fired = self.gate.evaluate(severity["severity_index"])

        alert_result = None
        if alert_fired:
            # Stage 6B: structured JSON alert content, then dispatch.
            # Deliberately a SEPARATE LLM call from explain_severity's
            # free-text explanation (not a reuse of `explanation`) --
            # CLAUDE.md's structured-output requirement applies
            # specifically to alert generation, and asking for JSON
            # up front produces a materially different, more
            # machine-parseable response than reformatting free text
            # after the fact would.
            alert_content = generate_alert_content(
                severity["severity_label"], severity["confidence"], reading
            )
            alert_result = send_alert(
                severity["severity_label"], reading, alert_content["summary"]
            )

        return TickResult(
            reading=reading,
            accepted=accepted,
            severity=severity,
            explanation=explanation,
            alert_fired=alert_fired,
            alert_result=alert_result,
            exhausted=False,
        )

    def reset(self) -> None:
        """Rewind the source and clear all pipeline state -- for a dashboard 'restart' button."""
        self.source.reset()
        self.buffer.clear()
        self.gate.reset()
        self._last_explained_tier = None
        self._cached_explanation = None
