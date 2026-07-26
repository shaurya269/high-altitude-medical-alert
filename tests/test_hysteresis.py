"""
Tests for the Day 11 hysteresis gate (src/alerts/hysteresis.py).

`now` is injected explicitly throughout rather than relying on real
wall-clock time -- this is what lets the 15-minute cooldown path be
tested in milliseconds instead of actually sleeping 900 seconds per test.

Run with: python -m pytest tests/test_hysteresis.py -v
"""

from __future__ import annotations

from src.alerts.hysteresis import HysteresisGate
from src.config import SEVERITY_INDEX


def test_single_elevated_reading_does_not_fire():
    """CLAUDE.md: 'a single noisy reading must never trigger an alert.'"""
    gate = HysteresisGate(alert_tier=2, consecutive_required=3, cooldown_seconds=900)
    assert gate.evaluate(severity_index=4, now=0.0) is False


def test_fires_after_consecutive_required_readings():
    gate = HysteresisGate(alert_tier=2, consecutive_required=3, cooldown_seconds=900)
    assert gate.evaluate(4, now=0.0) is False
    assert gate.evaluate(4, now=1.0) is False
    assert gate.evaluate(4, now=2.0) is True  # third consecutive elevated reading


def test_streak_resets_on_a_single_normal_reading():
    """A single reading dipping back below threshold must reset the
    streak -- this is what actually distinguishes sustained illness from
    a noisy spike-then-recovery pattern."""
    gate = HysteresisGate(alert_tier=2, consecutive_required=3, cooldown_seconds=900)
    assert gate.evaluate(4, now=0.0) is False
    assert gate.evaluate(4, now=1.0) is False
    assert gate.evaluate(0, now=2.0) is False  # dips back to Normal -- resets streak
    assert gate.evaluate(4, now=3.0) is False  # streak restarts, only 1 so far
    assert gate.evaluate(4, now=4.0) is False  # 2
    assert gate.evaluate(4, now=5.0) is True  # 3rd consecutive since the reset


def test_below_tier_never_fires_regardless_of_streak_length():
    gate = HysteresisGate(alert_tier=2, consecutive_required=3, cooldown_seconds=900)
    for i in range(20):
        assert gate.evaluate(1, now=float(i)) is False  # Mild AMS, below alert_tier=2


def test_cooldown_suppresses_immediate_refire():
    """Even with the streak condition satisfied again, a second alert
    must not fire until cooldown_seconds has elapsed since the last one."""
    gate = HysteresisGate(alert_tier=2, consecutive_required=1, cooldown_seconds=900)
    assert gate.evaluate(4, now=0.0) is True  # fires immediately (consecutive_required=1)
    assert gate.evaluate(4, now=100.0) is False  # still elevated, but within cooldown
    assert gate.evaluate(4, now=899.9) is False  # just under 900s
    assert gate.evaluate(4, now=900.0) is True  # cooldown has fully elapsed


def test_reset_clears_streak_and_cooldown():
    gate = HysteresisGate(alert_tier=2, consecutive_required=1, cooldown_seconds=900)
    assert gate.evaluate(4, now=0.0) is True
    gate.reset()
    assert gate.evaluate(4, now=1.0) is True  # cooldown cleared, fires again immediately


def test_uses_config_defaults_when_unspecified():
    """Sanity check that the gate actually reads from config.py rather
    than silently using different hardcoded numbers -- a drift here would
    mean the gate no longer matches what config.py documents."""
    from src.config import (
        HYSTERESIS_ALERT_TIER,
        HYSTERESIS_CONSECUTIVE_READINGS,
        HYSTERESIS_COOLDOWN_SECONDS,
    )

    gate = HysteresisGate()
    assert gate.alert_tier == HYSTERESIS_ALERT_TIER
    assert gate.consecutive_required == HYSTERESIS_CONSECUTIVE_READINGS
    assert gate.cooldown_seconds == HYSTERESIS_COOLDOWN_SECONDS
    assert gate.alert_tier == SEVERITY_INDEX["Severe AMS"]


def test_consecutive_count_property_reflects_current_streak():
    gate = HysteresisGate(alert_tier=2, consecutive_required=5, cooldown_seconds=900)
    gate.evaluate(4, now=0.0)
    gate.evaluate(4, now=1.0)
    assert gate.consecutive_count == 2
    gate.evaluate(0, now=2.0)
    assert gate.consecutive_count == 0
