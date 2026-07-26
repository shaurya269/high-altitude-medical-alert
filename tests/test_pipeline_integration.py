"""
Tests for the Day 11 full pipeline integration (src/pipeline.py) -- the
MedicalAlertPipeline class that wires DataSource -> buffer -> feature
engineering -> ML classification -> hysteresis -> LLM -> alerts together.

Runs entirely against the fallback (no GROQ_API_KEY / TELEGRAM_BOT_TOKEN)
paths, same reasoning as test_llm_chat.py / test_alert_bot.py -- fast,
deterministic, no external network dependency. The pipeline's CORRECTNESS
(does it call the right things in the right order, does hysteresis
actually gate alerts) doesn't depend on which LLM/Telegram backend is
active, only on the plumbing between modules being right.

Run with: python -m pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import pytest

from src.datasource import ManualDataSource, ScenarioDataSource
from src.pipeline import MedicalAlertPipeline


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    from src.alerts import alert_bot
    from src.llm import llm_chat

    monkeypatch.setattr(alert_bot, "ALERT_LOG_PATH", tmp_path / "alert_log.jsonl")
    monkeypatch.setattr(llm_chat, "_client", None)
    monkeypatch.setattr(llm_chat, "_client_checked", False)
    yield


def test_tick_returns_no_severity_before_buffer_fills():
    """Below min_readings_for_classification, tick() must not attempt a
    classification at all -- a prediction off 1-2 readings with zero
    trend history would be meaningless (see pipeline.py's docstring)."""
    source = ManualDataSource()
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=60)

    result = pipeline.tick()
    assert result.severity is None
    assert result.explanation is None
    assert result.alert_fired is False
    assert result.exhausted is False


def test_tick_produces_severity_once_buffer_fills():
    source = ManualDataSource()
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=10)

    result = None
    for _ in range(10):
        result = pipeline.tick()
    assert result.severity is not None
    assert result.severity["severity_label"] in [
        "Normal", "Mild AMS", "Severe AMS", "HAPE risk", "HACE risk"
    ]
    assert result.explanation is not None


def test_tick_reports_exhausted_when_source_runs_out():
    source = ScenarioDataSource("Normal", duration_minutes=1, seed=1)
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=10)

    n = len(source)
    exhausted_seen = False
    for _ in range(n + 5):
        result = pipeline.tick()
        if result.exhausted:
            exhausted_seen = True
            break
    assert exhausted_seen


def test_severe_scenario_eventually_fires_an_alert():
    """End-to-end integration check: a sustained HACE-risk trajectory,
    run through the real hysteresis gate with real config thresholds,
    must eventually produce alert_fired=True -- this is the test that
    would catch a wiring bug (e.g. severity_index passed to the wrong
    gate argument) that no individual module's unit tests would catch."""
    source = ScenarioDataSource("HACE risk", duration_minutes=25, seed=11)
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=60)

    alert_fired = False
    while True:
        result = pipeline.tick()
        if result.exhausted:
            break
        if result.alert_fired:
            alert_fired = True
            assert result.alert_result is not None
            assert "sent" in result.alert_result
            break
    assert alert_fired, "a sustained HACE-risk scenario should trigger at least one alert"


def test_alert_never_fires_twice_within_cooldown():
    """Regression test for the exact failure mode CLAUDE.md warns about:
    a sustained elevated episode must produce ONE alert, not one per
    tick, for at least the cooldown window."""
    from src.config import HYSTERESIS_COOLDOWN_SECONDS

    source = ScenarioDataSource("HACE risk", duration_minutes=25, seed=11)
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=60)

    alert_tick_indices = []
    i = 0
    while True:
        result = pipeline.tick()
        if result.exhausted:
            break
        if result.alert_fired:
            alert_tick_indices.append(i)
        i += 1

    # Consecutive alert ticks (1 tick = 1 second at SAMPLE_RATE_HZ) must be
    # spaced at least the cooldown apart.
    for a, b in zip(alert_tick_indices, alert_tick_indices[1:]):
        assert (b - a) >= HYSTERESIS_COOLDOWN_SECONDS


def test_reset_clears_buffer_gate_and_explanation_cache():
    source = ManualDataSource()
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=5)
    for _ in range(10):
        pipeline.tick()
    assert len(pipeline.buffer) > 0

    pipeline.reset()
    assert len(pipeline.buffer) == 0
    assert pipeline.gate.consecutive_count == 0
    assert pipeline._last_explained_tier is None


def test_normal_scenario_never_fires_an_alert():
    """Sanity check in the other direction: a Normal-tier scenario should
    never accumulate enough consecutive elevated readings to fire."""
    source = ScenarioDataSource("Normal", duration_minutes=30, seed=2)
    pipeline = MedicalAlertPipeline(source, min_readings_for_classification=60)

    any_alert = False
    while True:
        result = pipeline.tick()
        if result.exhausted:
            break
        if result.alert_fired:
            any_alert = True
    assert not any_alert
