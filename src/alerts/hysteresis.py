"""
Hysteresis gate -- Stage 5 of the data flow diagram. Decides WHETHER an
alert should fire; alert_bot.py (Day 10) decides HOW to send one once this
says yes. Kept as two separate modules on purpose (see alert_bot.py's
docstring) so gating logic and delivery logic can be tested independently.

CLAUDE.md Section 8 / "What NOT to do": "Don't let a single noisy reading
trigger a Telegram alert -- hysteresis gate is required." The gate has
TWO independent conditions, both from config.py, both must hold:

1. Severity >= HYSTERESIS_ALERT_TIER for HYSTERESIS_CONSECUTIVE_READINGS
   consecutive readings IN A ROW (not just "N times recently" -- a single
   good reading in the middle resets the streak). This is what actually
   prevents a transient motion-artifact spike from firing an alert: real
   sustained illness looks like a run of elevated readings, a sensor
   glitch looks like an isolated one.
2. At least HYSTERESIS_COOLDOWN_SECONDS has elapsed since the last alert
   ACTUALLY FIRED (not since the gate last considered firing). Without
   this, a sustained Severe-AMS episode would fire a new alert on every
   single reading once the consecutive-readings threshold is first met --
   the medical contact would get spammed continuously for as long as the
   condition persists, rather than one alert plus periodic re-notification.
"""

from __future__ import annotations

import time

from src.config import (
    HYSTERESIS_ALERT_TIER,
    HYSTERESIS_CONSECUTIVE_READINGS,
    HYSTERESIS_COOLDOWN_SECONDS,
)


class HysteresisGate:
    """
    Stateful across calls -- must be ONE instance per monitored subject,
    persisted across the live loop's ticks (not recreated each reading),
    since its entire job is remembering recent history. A fresh instance
    per call would always see a consecutive-streak of exactly 1 and never
    fire, defeating the whole point.
    """

    def __init__(
        self,
        alert_tier: int = HYSTERESIS_ALERT_TIER,
        consecutive_required: int = HYSTERESIS_CONSECUTIVE_READINGS,
        cooldown_seconds: float = HYSTERESIS_COOLDOWN_SECONDS,
    ):
        self.alert_tier = alert_tier
        self.consecutive_required = consecutive_required
        self.cooldown_seconds = cooldown_seconds

        self._consecutive_count = 0
        self._last_alert_time: float | None = None

    def evaluate(self, severity_index: int, now: float | None = None) -> bool:
        """
        Feed one new severity classification in. Returns True exactly
        when an alert should fire NOW (both conditions met) -- callers
        (the main loop, Day 11) are expected to call this once per
        reading/tick and act immediately on a True return, since calling
        evaluate() again immediately after a True would correctly return
        False (cooldown just reset), not fire twice for one episode.

        `now` is injectable (defaults to time.monotonic()) specifically so
        tests can control time directly instead of sleeping in real time
        for 15 minutes to exercise the cooldown path -- see
        tests/test_hysteresis.py.
        """
        if now is None:
            now = time.monotonic()

        if severity_index >= self.alert_tier:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0  # any reading below threshold resets the streak

        streak_satisfied = self._consecutive_count >= self.consecutive_required
        cooldown_satisfied = (
            # No alert has ever fired yet -> cooldown trivially satisfied;
            # otherwise only satisfied once enough wall-clock time has
            # passed since the last one actually fired.
            self._last_alert_time is None or (now - self._last_alert_time) >= self.cooldown_seconds
        )

        if streak_satisfied and cooldown_satisfied:
            self._last_alert_time = now
            return True
        return False

    def reset(self) -> None:
        """Clear all state -- used when switching demo scenarios/subjects,
        so a fresh session doesn't inherit a stale consecutive-count or
        cooldown timer from whatever was running before it."""
        self._consecutive_count = 0
        self._last_alert_time = None

    @property
    def consecutive_count(self) -> int:
        """Current streak length -- surfaced for the dashboard, e.g. a
        'building toward alert: 2/3' indicator."""
        return self._consecutive_count

    @property
    def seconds_since_last_alert(self) -> float | None:
        """None if no alert has ever fired; otherwise elapsed time -- for a dashboard cooldown countdown."""
        if self._last_alert_time is None:
            return None
        return time.monotonic() - self._last_alert_time
