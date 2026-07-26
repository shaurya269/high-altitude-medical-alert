"""
Tests for the Day 10 Telegram alert dispatch (src/alerts/alert_bot.py).

Like test_llm_chat.py, these exercise the FALLBACK (unconfigured) path,
not a real Telegram send -- an automated test suite should never depend
on live network access to a rate-limited external API, and shouldn't spam
a real chat every time `pytest` runs. Live delivery is verified manually
via `python -m src.alerts.alert_bot` when real credentials are configured.

Run with: python -m pytest tests/test_alert_bot.py -v
"""

from __future__ import annotations

import json

import pytest

from src.alerts import alert_bot


@pytest.fixture(autouse=True)
def no_telegram_credentials(monkeypatch, tmp_path):
    """Unset Telegram credentials AND redirect the alert log to a temp
    file, so these tests never touch the real data/processed/alert_log.jsonl
    or attempt a real send regardless of what's in the machine's .env."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(alert_bot, "ALERT_LOG_PATH", tmp_path / "alert_log.jsonl")
    yield


def test_is_telegram_configured_false_without_credentials():
    assert alert_bot.is_telegram_configured() is False


def test_format_alert_message_includes_required_fields():
    """CLAUDE.md Section 8: alert message must include severity tier,
    vitals snapshot, LLM summary, timestamp."""
    message = alert_bot.format_alert_message(
        "HAPE risk",
        {"spo2": 78, "hr": 125, "temp": 37.3, "altitude": 4500},
        "Test LLM summary text.",
        timestamp="2026-01-01 12:00:00 UTC",
    )
    assert "HAPE risk" in message
    assert "78" in message  # spo2
    assert "125" in message  # hr
    assert "4500" in message  # altitude
    assert "Test LLM summary text." in message
    assert "2026-01-01 12:00:00 UTC" in message


def test_format_alert_message_includes_prototype_disclaimer():
    """CLAUDE.md's project-wide requirement: never let the system present
    itself as a certified medical device -- this must appear on every
    alert, not just in the README."""
    message = alert_bot.format_alert_message(
        "Severe AMS", {"spo2": 80, "hr": 110}, "summary"
    )
    assert "not a certified medical device" in message.lower()


def test_send_alert_falls_back_when_unconfigured():
    result = alert_bot.send_alert(
        "Severe AMS", {"spo2": 80, "hr": 110, "temp": 37.0, "altitude": 4000}, "summary"
    )
    assert result["sent"] is False
    assert "not configured" in result["error"].lower() or "TELEGRAM" in result["error"]


def test_send_alert_logs_locally_when_unconfigured():
    alert_bot.send_alert(
        "Mild AMS", {"spo2": 90, "hr": 90, "temp": 36.8, "altitude": 2500}, "summary"
    )
    records = alert_bot.read_alert_log()
    assert len(records) == 1
    assert records[0]["sent"] is False
    assert "Mild AMS" in records[0]["message"]


def test_read_alert_log_returns_newest_first():
    for severity in ["Normal", "Mild AMS", "Severe AMS"]:
        alert_bot.send_alert(severity, {"spo2": 90, "hr": 90}, "summary")
    records = alert_bot.read_alert_log()
    assert len(records) == 3
    assert "Severe AMS" in records[0]["message"]  # most recent first
    assert "Normal" in records[2]["message"]


def test_read_alert_log_respects_limit():
    for i in range(5):
        alert_bot.send_alert("Normal", {"spo2": 95, "hr": 70}, f"summary {i}")
    records = alert_bot.read_alert_log(limit=2)
    assert len(records) == 2


def test_read_alert_log_empty_when_no_log_file():
    assert alert_bot.read_alert_log() == []
