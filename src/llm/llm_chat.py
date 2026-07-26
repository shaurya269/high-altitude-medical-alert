"""
LLM interpretation + chat layer (CLAUDE.md Section 7). Wraps the Groq API
(Llama 3.3 70B, chosen for free-tier speed -- CLAUDE.md Section 6) behind
three functions, all built on the SAME guardrailed system prompt
(prompts/system_prompt.py) so the non-negotiable boundaries can't drift
between call sites:

- explain_severity(): Stage 6A of the data flow diagram -- turns the ML
  model's severity + vitals into a plain-language explanation. Runs
  ALWAYS, regardless of whether the hysteresis gate passes.
- chat(): live conversational Q&A about the current reading, for the
  dashboard's chat panel.
- generate_alert_content(): Stage 6B's structured JSON generation --
  ONLY called when the hysteresis gate passes (src/alerts/), producing a
  machine-parseable summary for alert_bot.py's Telegram message.

Every function ALWAYS receives the ML model's severity classification as
ground-truth context in the prompt (CLAUDE.md: "the LLM explains it, it
doesn't invent its own assessment from raw numbers") -- none of them are
ever given raw vitals alone and asked to infer severity themselves.

Graceful degradation (matching every other optional-dependency piece of
this project -- Redis, Neo4j, Langfuse in the sister CineMind project use
the same pattern): if GROQ_API_KEY isn't set, every function here returns
a clearly-labeled templated fallback instead of raising, so the rest of
the pipeline (ML classification, hysteresis, alerts) keeps working without
an LLM at all.
"""

from __future__ import annotations

import json
import os

from src.config import SEVERITY_TIERS
from src.llm.prompts.system_prompt import SYSTEM_PROMPT

GROQ_MODEL = "llama-3.3-70b-versatile"

_client = None
_client_checked = False


def _get_client():
    """
    Lazily construct the Groq client on first real use, not at import
    time -- importing llm_chat.py must never fail or make a network call
    just because GROQ_API_KEY isn't set yet (e.g. during tests, or before
    the owner has added it to .env). Returns None if no key is configured,
    which every calling function treats as "use the fallback path."
    """
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    import groq

    _client = groq.Groq(api_key=api_key)
    return _client


def _severity_context(severity_label: str, confidence: float, readings: dict) -> str:
    """
    The ground-truth block appended to every prompt -- see module
    docstring's point about the LLM never inferring severity itself.
    Kept as one shared formatter so explain_severity/chat/alert generation
    can't each phrase this slightly differently and drift apart.
    """
    return (
        f"ML MODEL CLASSIFICATION (ground truth -- do not override): {severity_label} "
        f"(confidence: {confidence:.0%})\n"
        f"CURRENT READINGS: SpO2 {readings.get('spo2', '?')}%, "
        f"HR {readings.get('hr', '?')} bpm, "
        f"Temp {readings.get('temp', '?')}°C, "
        f"Altitude {readings.get('altitude', '?')}m"
    )


def _fallback_explanation(severity_label: str, confidence: float) -> str:
    """
    Templated, non-LLM explanation used when no Groq key is configured.
    Deliberately follows the exact same guardrails as the real LLM path
    (no diagnosis language, recommends real help for elevated tiers) --
    the fallback should degrade gracefully in FEATURES, not in SAFETY.
    """
    templates = {
        "Normal": (
            f"The model classifies the current readings as Normal (confidence {confidence:.0%}). "
            "No signs of altitude illness in the current pattern."
        ),
        "Mild AMS": (
            f"The model classifies the current readings as Mild AMS (confidence {confidence:.0%}). "
            "This pattern is consistent with mild acute mountain sickness. Monitor closely and "
            "avoid further ascent until this improves."
        ),
        "Severe AMS": (
            f"The model classifies the current readings as Severe AMS (confidence {confidence:.0%}). "
            "This is an elevated pattern. Real medical attention is recommended, and descent "
            "should be considered if symptoms are present."
        ),
        "HAPE risk": (
            f"The model classifies the current readings as HAPE risk (confidence {confidence:.0%}). "
            "This is a serious elevated pattern. Seek real medical attention immediately and "
            "descend if possible."
        ),
        "HACE risk": (
            f"The model classifies the current readings as HACE risk (confidence {confidence:.0%}). "
            "This is the most serious pattern this system detects. Seek emergency medical "
            "attention immediately and descend without delay."
        ),
    }
    return templates.get(
        severity_label,
        f"The model classifies the current readings as {severity_label} (confidence {confidence:.0%}).",
    ) + "\n\n[LLM interpretation unavailable -- GROQ_API_KEY not configured. This is a templated fallback.]"


def explain_severity(severity_label: str, confidence: float, readings: dict) -> str:
    """
    Stage 6A: turn a severity classification + current readings into a
    plain-language explanation for the dashboard. Always called,
    independent of whether the hysteresis gate passes (unlike alert
    generation, which only runs when it does).
    """
    client = _get_client()
    if client is None:
        return _fallback_explanation(severity_label, confidence)

    context = _severity_context(severity_label, confidence, readings)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                "Explain this classification in plain language for the person wearing the "
                "sensor. 2-4 sentences. Follow all the rules in your system prompt."
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as exc:  # Groq/network errors -- degrade, don't crash the dashboard
        return _fallback_explanation(severity_label, confidence) + f"\n[LLM call failed: {exc}]"


def chat(
    user_message: str,
    severity_label: str,
    confidence: float,
    readings: dict,
    history: list[dict] | None = None,
) -> str:
    """
    Live conversational Q&A about the current reading (dashboard chat
    panel). `history` is a list of {"role": "user"/"assistant", "content":
    str} dicts from earlier turns in the SAME session -- passed through so
    the model has conversational context, but the severity ground-truth
    context is re-injected on EVERY call (not just the first), so a long
    conversation can never drift away from what the ML model actually
    classified.
    """
    client = _get_client()
    if client is None:
        return (
            "[LLM chat unavailable -- GROQ_API_KEY not configured.]\n"
            f"Current model classification: {severity_label} (confidence {confidence:.0%})."
        )

    context = _severity_context(severity_label, confidence, readings)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": context})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.5, max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"[LLM call failed: {exc}] Current model classification: {severity_label}."


# JSON schema every alert content payload must match -- alert_bot.py
# (Day 10) formats a Telegram message directly from these fields, so a
# missing key here would produce a broken alert, not just a display glitch.
ALERT_JSON_KEYS = {"severity", "summary", "key_vitals", "recommendation"}


def _fallback_alert_content(severity_label: str, readings: dict) -> dict:
    return {
        "severity": severity_label,
        "summary": (
            f"[Templated fallback -- LLM unavailable] Sustained {severity_label} classification."
        ),
        "key_vitals": (
            f"SpO2 {readings.get('spo2', '?')}%, HR {readings.get('hr', '?')}bpm, "
            f"Altitude {readings.get('altitude', '?')}m"
        ),
        "recommendation": "Seek real medical attention and consider descent.",
    }


def generate_alert_content(severity_label: str, confidence: float, readings: dict) -> dict:
    """
    Stage 6B: structured JSON generation, ONLY called by the hysteresis
    gate (src/alerts/) once it decides an alert should actually fire.
    CLAUDE.md Section 7's "structured output requirement" -- request JSON
    output from Groq (response_format enforced) so the alert message is
    machine-parseable, not free text that might omit a required field.

    Returns a dict with exactly ALERT_JSON_KEYS -- validated before
    returning, so alert_bot.py can trust the shape without its own
    defensive parsing of possibly-malformed LLM output.
    """
    client = _get_client()
    if client is None:
        return _fallback_alert_content(severity_label, readings)

    context = _severity_context(severity_label, confidence, readings)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                "Generate a structured alert for a medical facility contact, who is receiving "
                "this because the severity has been SUSTAINED at an elevated level (not a "
                "single reading). Respond with ONLY a JSON object with exactly these keys:\n"
                '  "severity": the severity tier name, exactly as given above\n'
                '  "summary": 1-2 sentence plain-language summary of the situation\n'
                '  "key_vitals": the current vitals as a short readable string\n'
                '  "recommendation": one clear, general next-step recommendation '
                "(no dosing, no specific treatment -- follow your system prompt's rules)\n"
                "No other text outside the JSON object."
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = json.loads(response.choices[0].message.content)
        if not ALERT_JSON_KEYS <= set(content.keys()):
            # Model returned valid JSON but missing a required field --
            # treat as a failure rather than shipping a partial alert, per
            # CLAUDE.md's "machine-parseable, not free text that might
            # omit required fields" requirement.
            return _fallback_alert_content(severity_label, readings)
        return content
    except Exception:
        return _fallback_alert_content(severity_label, readings)


def is_llm_available() -> bool:
    """Whether a real Groq call will be made, or the templated fallback used -- surfaced by the dashboard's health badge and __main__ smoke test below."""
    return _get_client() is not None


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # Reset the lazy-client cache so the .env just loaded is actually seen
    # -- module-level state from an earlier import (e.g. if this file was
    # imported before load_dotenv() ran) would otherwise stick.
    _client_checked = False
    _client = None

    print("LLM available:", is_llm_available())
    demo_readings = {"spo2": 82, "hr": 118, "temp": 37.1, "altitude": 4200}

    print("\n--- explain_severity ---")
    print(explain_severity("Severe AMS", 0.87, demo_readings))

    print("\n--- chat ---")
    print(chat("Should I be worried?", "Severe AMS", 0.87, demo_readings))

    print("\n--- generate_alert_content ---")
    print(json.dumps(generate_alert_content("Severe AMS", 0.87, demo_readings), indent=2))
