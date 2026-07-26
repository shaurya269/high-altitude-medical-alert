"""
Tests for the Day 12 Streamlit dashboard (src/app/streamlit_app.py), using
Streamlit's own AppTest harness (streamlit.testing.v1) -- this runs the
actual app script in a simulated Streamlit runtime and lets us click
buttons / read rendered elements, without needing a real browser (that
verification was also done manually via Playwright screenshots during
development; AppTest is what makes it repeatable in the automated suite).

Runs against the fallback (no GROQ_API_KEY/TELEGRAM_BOT_TOKEN) paths for
determinism and speed, same reasoning as the other Day 9-11 test files.

Run with: python -m pytest tests/test_streamlit_app.py -v
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = "src/app/streamlit_app.py"


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


def test_app_loads_without_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_initial_state_shows_empty_classification():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    body_text = " ".join(m.value for m in at.info)
    assert "Start a Demo Mode source" in body_text


def test_starting_a_scenario_creates_buttons():
    """Clicking 'Start scenario' should make Step forward / Auto-play /
    Reset appear -- these only render once st.session_state.pipeline is set."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    start_button = next(b for b in at.sidebar.button if b.label == "Start scenario")
    start_button.click().run(timeout=30)

    button_labels = {b.label for b in at.sidebar.button}
    assert "Step forward" in button_labels
    assert "Reset" in button_labels


def test_stepping_forward_accumulates_history_and_eventually_classifies():
    """The core regression test for the empty-state-vs-not-enough-history
    distinction found during manual browser testing: severity should
    remain unclassified below min_readings_for_classification (60 by
    default) and become classified once enough ticks have accumulated."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    start_button = next(b for b in at.sidebar.button if b.label == "Start scenario")
    start_button.click().run(timeout=30)

    step_button = next(b for b in at.sidebar.button if b.label == "Step forward")

    # Below the classification threshold: still empty state.
    for _ in range(5):
        step_button = next(b for b in at.sidebar.button if b.label == "Step forward")
        step_button.click().run(timeout=30)
    body_text = " ".join(m.value for m in at.info)
    assert "Start a Demo Mode source" in body_text

    # Cross the threshold (60 by default): should now show a real severity tier.
    for _ in range(60):
        step_button = next(b for b in at.sidebar.button if b.label == "Step forward")
        step_button.click().run(timeout=30)

    from src.config import SEVERITY_TIERS

    all_markdown_text = " ".join(m.value for m in at.markdown)
    assert any(tier in all_markdown_text for tier in SEVERITY_TIERS)


def test_reset_clears_history():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    start_button = next(b for b in at.sidebar.button if b.label == "Start scenario")
    start_button.click().run(timeout=30)
    for _ in range(10):
        step_button = next(b for b in at.sidebar.button if b.label == "Step forward")
        step_button.click().run(timeout=30)

    assert len(at.session_state["history"]) == 10

    reset_button = next(b for b in at.sidebar.button if b.label == "Reset")
    reset_button.click().run(timeout=30)
    assert len(at.session_state["history"]) == 0


def test_manual_sliders_mode_starts_without_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    manual_radio = at.sidebar.radio[0]
    manual_radio.set_value("Manual sliders").run(timeout=30)
    assert not at.exception

    start_button = next(b for b in at.sidebar.button if b.label == "Start manual override")
    start_button.click().run(timeout=30)
    assert not at.exception
    assert len(at.sidebar.slider) == 4  # SpO2, HR, temp, altitude
