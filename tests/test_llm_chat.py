"""
Tests for the Day 9 LLM layer (src/llm/llm_chat.py).

Deliberately test the FALLBACK path (no GROQ_API_KEY) rather than mocking
real Groq responses -- the fallback path is fully deterministic, exercises
the same guardrail-relevant code (the wording is hand-written, not model
output), and importantly is what CI / a fresh clone without a .env file
will actually run. A live-API smoke test exists as this module's own
`__main__` block (`python -m src.llm.llm_chat`) for manual verification
when a real key is configured -- not part of the automated suite, since
tests shouldn't depend on a paid/rate-limited external API being reachable.

Run with: python -m pytest tests/test_llm_chat.py -v
"""

from __future__ import annotations

import os

import pytest

from src.llm import llm_chat


@pytest.fixture(autouse=True)
def no_groq_key(monkeypatch):
    """Every test in this file runs with GROQ_API_KEY unset and the
    lazy-client cache cleared, so they exercise the fallback path
    regardless of whether a real .env happens to be present on the
    machine running the tests."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_chat, "_client", None)
    monkeypatch.setattr(llm_chat, "_client_checked", False)
    yield


def test_is_llm_available_false_without_key():
    assert llm_chat.is_llm_available() is False


def test_explain_severity_fallback_mentions_severity_and_confidence():
    result = llm_chat.explain_severity("Severe AMS", 0.87, {"spo2": 82, "hr": 118})
    assert "Severe AMS" in result
    assert "87%" in result


def test_explain_severity_fallback_never_uses_diagnostic_language():
    """Guardrail check on the deterministic fallback text itself -- CLAUDE.md
    Section 7 rule 1: never present as making a diagnosis. The fallback is
    hand-written (not LLM output) so this is a straightforward regression
    test, not a fuzzy check on model behavior."""
    for tier in ["Normal", "Mild AMS", "Severe AMS", "HAPE risk", "HACE risk"]:
        result = llm_chat.explain_severity(tier, 0.9, {"spo2": 90, "hr": 90})
        assert "you have" not in result.lower()
        assert "you are diagnosed" not in result.lower()


def test_explain_severity_fallback_recommends_help_for_elevated_tiers():
    """Guardrail check for rule 4: never discourage seeking real medical
    help when severity is elevated. Every elevated tier's fallback must
    mention seeking medical attention."""
    for tier in ["Severe AMS", "HAPE risk", "HACE risk"]:
        result = llm_chat.explain_severity(tier, 0.9, {"spo2": 80, "hr": 110})
        assert "medical attention" in result.lower()


def test_explain_severity_fallback_normal_is_not_alarmist():
    result = llm_chat.explain_severity("Normal", 0.95, {"spo2": 97, "hr": 72})
    assert "emergency" not in result.lower()
    assert "medical attention" not in result.lower()


def test_chat_fallback_includes_severity_context():
    result = llm_chat.chat(
        "Should I be worried?", "HAPE risk", 0.8, {"spo2": 78, "hr": 120}
    )
    assert "HAPE risk" in result


def test_generate_alert_content_fallback_has_all_required_keys():
    content = llm_chat.generate_alert_content(
        "Severe AMS", 0.87, {"spo2": 82, "hr": 118, "temp": 37.1, "altitude": 4200}
    )
    assert llm_chat.ALERT_JSON_KEYS <= set(content.keys())


def test_generate_alert_content_fallback_never_gives_dosing():
    """Guardrail check for rule 3: no specific treatment/medication dosing.
    The fallback's recommendation field must not contain a dose-shaped
    string like a number followed by 'mg'."""
    import re

    content = llm_chat.generate_alert_content(
        "HACE risk", 0.9, {"spo2": 70, "hr": 130, "temp": 37.5, "altitude": 4800}
    )
    assert not re.search(r"\d+\s*mg", content["recommendation"], re.IGNORECASE)


def test_severity_context_includes_ground_truth_marker():
    """The shared prompt-context formatter must label the ML classification
    as ground truth explicitly -- this is the mechanism that keeps the LLM
    from inventing its own severity assessment (CLAUDE.md: 'the LLM
    explains it, it doesn't invent its own assessment from raw numbers')."""
    context = llm_chat._severity_context("Mild AMS", 0.75, {"spo2": 90, "hr": 95})
    assert "ground truth" in context.lower()
    assert "Mild AMS" in context
