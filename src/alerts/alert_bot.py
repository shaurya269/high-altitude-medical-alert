"""
Telegram alert dispatch (CLAUDE.md Section 8, Stage 6B of the data flow
diagram). This module is called ONLY after the hysteresis gate
(src/alerts/hysteresis.py) has decided an alert should actually fire --
it has no gating logic of its own and trusts its caller completely. That
separation matters: hysteresis.py's whole job is deciding WHETHER to
alert, this module's whole job is deciding HOW to format and send one --
mixing the two would make it harder to unit-test either in isolation.

Message format is defined ONCE here (format_alert_message()) and reused
by both the real send path and anything that wants to preview/log an
alert without actually sending it (e.g. the Streamlit dashboard's alert
log panel, Day 12) -- CLAUDE.md: "Keep format consistent -- define it once
in alert_bot.py and reuse."

Graceful degradation, same pattern as llm_chat.py: if
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID aren't set, send_alert() logs the
alert locally instead of raising -- CLAUDE.md's README explicitly plans
for "Telegram alerts fall back to local logging" when unconfigured.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from src.config import PROCESSED_DIR

# Local log used both as the offline fallback AND as a permanent record of
# every alert ever sent (even successful ones) -- the Streamlit alert log
# panel (Day 12) reads this file directly rather than needing its own
# separate persistence, and it's useful evidence that the hysteresis gate
# is behaving (not spamming) during Day 13's testing pass.
ALERT_LOG_PATH = PROCESSED_DIR / "alert_log.jsonl"


def format_alert_message(
    severity_label: str,
    readings: dict,
    llm_summary: str,
    timestamp: str | None = None,
) -> str:
    """
    The ONE alert message format, per CLAUDE.md Section 8: severity tier,
    vitals snapshot, LLM summary, timestamp. Plain text (not Telegram
    Markdown/HTML) deliberately -- LLM-generated summary text could
    contain characters that break Markdown parsing (an unescaped `_` or
    `*`), and a malformed-Markdown Telegram API call fails outright rather
    than degrading to plain text. Safer to just never opt into parse_mode.
    """
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        "ALTITUDE MEDICAL ALERT\n"
        f"Severity: {severity_label}\n"
        f"Time: {ts}\n\n"
        f"Vitals: SpO2 {readings.get('spo2', '?')}%, "
        f"HR {readings.get('hr', '?')}bpm, "
        f"Temp {readings.get('temp', '?')}°C, "
        f"Altitude {readings.get('altitude', '?')}m\n\n"
        f"{llm_summary}\n\n"
        "-- Prototype system, not a certified medical device. "
        "This alert was triggered by sustained sensor readings; verify independently."
    )


def _log_alert(message: str, sent: bool, error: str | None = None) -> None:
    """Append one JSON line per alert attempt -- sent or not -- to the local log."""
    import json

    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # create data/processed/ on first alert if it doesn't exist yet
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sent": sent,
        "message": message,
        "error": error,
    }
    with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:  # append, never overwrite -- this is a permanent running log
        f.write(json.dumps(record) + "\n")  # one JSON object per line (JSONL), so the log can be read back line-by-line without parsing a giant array


async def _send_async(message: str) -> None:
    from telegram import Bot

    token = os.environ["TELEGRAM_BOT_TOKEN"]  # raises KeyError if missing -- callers must check is_telegram_configured()/presence first, see send_alert()
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=message)  # plain text send -- no parse_mode, see format_alert_message()'s docstring for why


def send_alert(
    severity_label: str,
    readings: dict,
    llm_summary: str,
) -> dict:
    """
    Format and send one Telegram alert. Returns a small status dict
    ({"sent": bool, "message": str, "error": str|None}) rather than
    raising on failure -- a failed Telegram send (bad token, network
    issue, rate limit) must never crash the live pipeline that called it;
    the caller (hysteresis.py) can inspect the result and decide whether
    to retry, but the classification/dashboard loop keeps running either way.

    python-telegram-bot's Bot.send_message is async (v20+ API) -- wrapped
    in asyncio.run() here so every OTHER module in this synchronous
    pipeline (feature engineering, model inference, Streamlit) doesn't
    need to become async just to call this one function.
    """
    message = format_alert_message(severity_label, readings, llm_summary)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        _log_alert(message, sent=False, error="TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured")
        return {
            "sent": False,
            "message": message,
            "error": "Telegram not configured -- logged locally instead (see data/processed/alert_log.jsonl)",
        }

    try:
        asyncio.run(_send_async(message))  # runs the async Telegram call to completion synchronously, see docstring above
        _log_alert(message, sent=True)
        return {"sent": True, "message": message, "error": None}
    except Exception as exc:  # any Telegram/network failure -- log it and report failure, never raise out of this function
        _log_alert(message, sent=False, error=str(exc))
        return {"sent": False, "message": message, "error": str(exc)}


def is_telegram_configured() -> bool:
    """Whether a real send will be attempted, or the local-log fallback used -- surfaced by the dashboard's health badge, same pattern as llm_chat.is_llm_available()."""
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")) and bool(os.environ.get("TELEGRAM_CHAT_ID"))


def read_alert_log(limit: int = 50) -> list[dict]:
    """Most recent `limit` alert records, newest first -- for the Streamlit alert log panel (Day 12)."""
    import json

    if not ALERT_LOG_PATH.exists():
        return []
    with open(ALERT_LOG_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return list(reversed(records))[:limit]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("Telegram configured:", is_telegram_configured())

    demo_readings = {"spo2": 81, "hr": 122, "temp": 37.2, "altitude": 4300}
    result = send_alert(
        "Severe AMS",
        demo_readings,
        "The model classifies this as Severe AMS, sustained over the last several readings. "
        "Real medical attention is recommended.",
    )
    print("sent:", result["sent"])
    if result["error"]:
        print("error:", result["error"])
    print("\n--- message ---")
    print(result["message"])
