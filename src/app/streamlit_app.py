"""
Streamlit dashboard -- Day 12, the final piece of the Demo Mode UI
(CLAUDE.md Section 2 / 3, "Stage 7" of the data flow diagram). This is a
thin presentation layer: it owns NO pipeline logic of its own, only calls
into src/pipeline.py's MedicalAlertPipeline (built Day 11) and renders
whatever comes back.

Run with: streamlit run src/app/streamlit_app.py

Why st.session_state, and why "step forward" instead of a background
thread: Streamlit reruns the ENTIRE script top-to-bottom on every
interaction (a button click, a slider drag). Without session_state, the
pipeline object itself -- along with the rolling buffer and hysteresis
gate's accumulated streak -- would be recreated from scratch on every
rerun, meaning the "3 consecutive elevated readings" hysteresis logic
could never actually accumulate across ticks. A background thread driving
ticks independently of Streamlit's rerun cycle is possible but adds real
complexity (thread-safety around session_state, orphaned threads on page
navigation) for a demo whose whole point is showing the pipeline's
behavior clearly, not simulating true real-time speed -- a manual/
auto-play "step forward" button, with each step being one full
`pipeline.tick()` inside a normal Streamlit rerun, is simpler and just as
effective for the demo's purpose.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Streamlit Cloud (and `streamlit run src/app/streamlit_app.py` from anywhere
# other than the project root) doesn't put the project root on sys.path the
# way running `python -m` from the root does -- without this, `from src...`
# below fails with "ModuleNotFoundError: No module named 'src'" even though
# every package installed correctly. Insert the project root (two levels up
# from this file: src/app/ -> src/ -> root) before any `src.*` import, the
# same fix the notebooks already apply via `sys.path.insert(0, ...)`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.alerts.alert_bot import is_telegram_configured, read_alert_log
from src.config import SEVERITY_TIERS
from src.data import harespod_loader
from src.datasource import (
    SCENARIOS,
    HarespodReplayDataSource,
    ManualDataSource,
    ScenarioDataSource,
)
from src.llm.llm_chat import chat as llm_chat_fn
from src.llm.llm_chat import is_llm_available
from src.pipeline import MedicalAlertPipeline

st.set_page_config(page_title="High-Altitude Medical Alert System", page_icon="\U0001F3D4", layout="wide")

SEVERITY_COLORS = {
    "Normal": "#2FB8A6",
    "Mild AMS": "#5B8DEF",
    "Severe AMS": "#E8A23D",
    "HAPE risk": "#E8583D",
    "HACE risk": "#8B2E2E",
}

# Immediate-medical-steps copy shown in the ALERT! popup (render_alert_popup(),
# below) -- deliberately general, non-dosing guidance only, matching the exact
# same guardrails src/llm/prompts/system_prompt.py enforces on the LLM (never
# diagnose, never give specific treatment/dosing, always point to real medical
# help for elevated tiers). Kept as static copy here (not LLM-generated) so the
# popup is instant and never depends on a Groq call succeeding.
IMMEDIATE_STEPS = {
    "Severe AMS": [
        "Stop ascending -- do not go any higher until symptoms improve.",
        "Rest and monitor closely; consider descending if symptoms don't improve or worsen.",
        "Stay hydrated; avoid alcohol and sedatives, which can mask worsening symptoms.",
        "Contact a real medical professional or your expedition's medical support.",
    ],
    "HAPE risk": [
        "Descend immediately -- even a few hundred meters can help. This is time-critical.",
        "Minimize exertion; the person should be carried or assisted if possible, not walk under their own power.",
        "Administer supplemental oxygen if available.",
        "Seek emergency medical attention as soon as possible -- this is a medical emergency.",
    ],
    "HACE risk": [
        "Descend immediately -- this takes priority over everything else. Do not delay for any reason.",
        "Do not leave the person alone; watch for worsening confusion, loss of coordination, or unconsciousness.",
        "Administer supplemental oxygen if available.",
        "Seek emergency medical attention immediately -- this is a life-threatening medical emergency.",
    ],
}


# ---------------------------------------------------------------------------
# Session state initialization -- runs once per browser session, not per
# rerun (Streamlit persists st.session_state across reruns automatically).
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "source_kind" not in st.session_state:
        st.session_state.source_kind = None
    if "history" not in st.session_state:
        # Rows of {timestamp, spo2, hr, temp, altitude, severity_index} for
        # the live line charts -- kept separately from the pipeline's own
        # ReadingBuffer, which only holds the trailing ROLLING_BUFFER_MINUTES
        # (Stage 2's rolling window is for feature engineering, not for
        # showing the user the FULL session history on a chart).
        st.session_state.history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "pending_alert" not in st.session_state:
        # Set by _do_tick() the instant a NEW alert fires; render_alert_popup()
        # consumes (and clears) it on the next render. Separate from
        # last_result.alert_fired because that flag is only true for the exact
        # tick the gate passed on -- without a dedicated field, a rerun
        # triggered by something else (e.g. opening the Chat tab) right after
        # an alert would have no way to know "the popup for that alert hasn't
        # been shown yet" vs. "there was never a pending alert at all."
        st.session_state.pending_alert = None


def _new_pipeline(source) -> None:
    st.session_state.pipeline = MedicalAlertPipeline(source)
    st.session_state.history = []
    st.session_state.chat_messages = []
    st.session_state.last_result = None
    st.session_state.pending_alert = None
    st.session_state.auto_play = False


# ---------------------------------------------------------------------------
# Sidebar: Demo Mode panel -- scenario picker / manual sliders / Harespod replay
# ---------------------------------------------------------------------------
def render_demo_mode_panel() -> None:
    st.sidebar.title("Demo Mode")
    st.sidebar.caption(
        "Prototype / learning project -- NOT a certified medical device. "
        "See README for the full disclaimer."
    )

    mode = st.sidebar.radio(
        "Data source",
        ["Scenario player", "Manual sliders", "Harespod replay"],
        key="mode_choice",
    )

    if mode == "Scenario player":
        scenario_name = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()))
        duration = st.sidebar.slider("Duration (minutes)", 10, 90, 30, step=5)
        if st.sidebar.button("Start scenario", type="primary"):
            source = ScenarioDataSource(scenario_name, duration_minutes=duration)
            _new_pipeline(source)
            st.session_state.source_kind = "scenario"

    elif mode == "Manual sliders":
        # Two separate `if` blocks, not if/else: before a pipeline exists,
        # only the "Start" button shows; once started, only the sliders
        # show. Both conditions are re-checked on every rerun (not just
        # once), so the UI naturally flips from one state to the other the
        # moment `st.session_state.pipeline` is set by the button click.
        if st.session_state.source_kind != "manual" or st.session_state.pipeline is None:
            if st.sidebar.button("Start manual override", type="primary"):
                _new_pipeline(ManualDataSource())
                st.session_state.source_kind = "manual"
        if st.session_state.source_kind == "manual" and st.session_state.pipeline is not None:
            source: ManualDataSource = st.session_state.pipeline.source
            st.sidebar.markdown("**Adjust vitals live:**")
            spo2 = st.sidebar.slider("SpO2 (%)", 50.0, 100.0, 96.0, step=0.5)
            hr = st.sidebar.slider("Heart rate (bpm)", 40.0, 180.0, 75.0, step=1.0)
            temp = st.sidebar.slider("Temperature (°C)", 34.0, 40.0, 36.8, step=0.1)
            altitude = st.sidebar.slider("Altitude (m)", 0.0, 8000.0, 2000.0, step=100.0)
            source.set_values(spo2=spo2, hr=hr, temp=temp, altitude=altitude)

    else:  # Harespod replay
        subjects = HarespodReplayDataSource.available_subjects()
        if not subjects:
            st.sidebar.warning(
                "Harespod data not downloaded. See src/data/harespod_loader.py's "
                "docstring for download steps, or use another Demo Mode instead."
            )
        else:
            subject_id = st.sidebar.selectbox("Subject recording", subjects)
            if st.sidebar.button("Start replay", type="primary"):
                source = HarespodReplayDataSource(subject_id)
                _new_pipeline(source)
                st.session_state.source_kind = "replay"

    st.sidebar.divider()

    pipeline = st.session_state.pipeline
    if pipeline is not None:
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Step forward"):
            _do_tick()
        label = "Pause" if st.session_state.auto_play else "Auto-play"
        if col2.button(label):
            st.session_state.auto_play = not st.session_state.auto_play
        if st.sidebar.button("Reset"):
            pipeline.reset()
            st.session_state.history = []
            st.session_state.chat_messages = []
            st.session_state.last_result = None
            st.session_state.pending_alert = None
            st.session_state.auto_play = False

        # Progress bar -- only meaningful for finite-length sources (scenario
        # player / Harespod replay both expose .progress, a [0,1] fraction of
        # the trajectory consumed so far). ManualDataSource has no fixed
        # length (the "next reading" is just whatever the sliders currently
        # say -- see its docstring), so there's nothing to show a fraction
        # of, and it's deliberately skipped rather than showing a fake bar.
        progress = getattr(pipeline.source, "progress", None)
        if progress is not None:
            st.sidebar.progress(
                min(progress, 1.0),
                text=f"Tick {len(st.session_state.history)} / {len(pipeline.source)}",
            )

    st.sidebar.divider()
    st.sidebar.markdown("**System status**")
    st.sidebar.markdown(f"- LLM (Groq): {'🟢 live' if is_llm_available() else '⚪ fallback mode'}")
    st.sidebar.markdown(
        f"- Telegram: {'🟢 configured' if is_telegram_configured() else '⚪ local-log fallback'}"
    )
    st.sidebar.markdown(f"- Harespod data: {'🟢 available' if harespod_loader.has_harespod_data() else '⚪ not downloaded'}")


def _do_tick() -> None:
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return
    result = pipeline.tick()
    st.session_state.last_result = result
    if result.exhausted:
        st.session_state.auto_play = False
        return
    if result.reading is not None:
        row = dict(result.reading)
        if result.severity is not None:
            row["severity_index"] = result.severity["severity_index"]
            row["severity_label"] = result.severity["severity_label"]
        st.session_state.history.append(row)
    if result.alert_fired and result.severity is not None:
        # Record what to show, not just a bool -- the popup needs the tier
        # name and vitals snapshot from THIS specific tick, and by the time
        # render_alert_popup() runs (after auto-play's own rerun/sleep cycle
        # further down in main()) st.session_state.last_result may have moved
        # on to a later tick already.
        st.session_state.pending_alert = {
            "severity_label": result.severity["severity_label"],
            "reading": dict(result.reading) if result.reading else {},
        }


# ---------------------------------------------------------------------------
# Main panel: severity readout, live plots, chat, alert log
# ---------------------------------------------------------------------------
# Number of recent history rows to average over on each side of the trend
# comparison, so a single noisy tick's classification can't flip the arrow
# back and forth -- mirrors the same "sustained, not single-reading" spirit
# as the hysteresis gate itself, just applied to a cosmetic trend indicator
# rather than the alert-firing decision.
_TREND_WINDOW = 5


def _severity_trend() -> tuple[str, str]:
    """
    Compare the average severity_index over the trailing _TREND_WINDOW
    history rows against the _TREND_WINDOW rows before that, using
    st.session_state.history (already recorded per-tick by _do_tick()) --
    deliberately NOT re-deriving trend from raw vitals here, since
    spo2_trend_5min etc. are already computed once by feature_engineering.py
    and used by the model itself; this is a coarser, purely cosmetic
    "is the CLASSIFICATION getting worse" signal for the UI, distinct from
    the model's own input features.

    Returns (arrow, label) -- e.g. ("↑", "worsening") -- or a flat dash
    if there isn't enough history yet to compare two windows.
    """
    history = st.session_state.history
    if len(history) < _TREND_WINDOW * 2:
        return "→", "not enough history yet"

    indices = [row.get("severity_index") for row in history if "severity_index" in row]
    if len(indices) < _TREND_WINDOW * 2:
        return "→", "not enough history yet"

    recent = sum(indices[-_TREND_WINDOW:]) / _TREND_WINDOW
    earlier = sum(indices[-_TREND_WINDOW * 2 : -_TREND_WINDOW]) / _TREND_WINDOW
    delta = recent - earlier

    if delta > 0.2:
        return "↑", "worsening"
    if delta < -0.2:
        return "↓", "improving"
    return "→", "stable"


def _metric_card(label: str, value: str) -> str:
    """Small styled-div card, matching the severity readout's card look
    (col1 in render_severity_readout()) and the System Info tab's tier
    cards -- one visual language for "a labeled fact in a box" everywhere
    in the dashboard, rather than plain st.metric widgets that look
    visually disconnected from the rest of the page."""
    return (
        "<div style='padding:12px 14px;margin-bottom:8px;background:rgba(128,128,128,0.06);"
        "border:1px solid rgba(128,128,128,0.15);border-radius:6px;'>"
        f"<div style='font-size:12px;color:#8FA3B8;text-transform:uppercase;letter-spacing:0.04em;'>{label}</div>"
        f"<div style='font-size:18px;font-weight:700;'>{value}</div>"
        "</div>"
    )


def render_severity_readout() -> None:
    result = st.session_state.last_result
    st.subheader("Current classification")

    if result is None or result.severity is None:
        st.info(
            "Start a Demo Mode source in the sidebar, then step forward or "
            "auto-play to begin classification. The model needs a short buffer "
            "of readings before its first classification."
        )
        return

    severity = result.severity
    color = SEVERITY_COLORS.get(severity["severity_label"], "#8FA3B8")
    trend_arrow, trend_label = _severity_trend()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(
            f"<div style='padding:16px;border-left:6px solid {color};background:rgba(128,128,128,0.08);"
            f"border-radius:6px;'><div style='font-size:13px;color:#8FA3B8;'>SEVERITY</div>"
            f"<div style='font-size:28px;font-weight:700;'>{severity['severity_label']} "
            f"<span style='font-size:18px;' title='{trend_label}'>{trend_arrow}</span></div>"
            f"<div style='font-size:13px;color:#8FA3B8;'>confidence {severity['confidence']:.0%} "
            f"&middot; model: {severity['model_used']} &middot; {trend_label}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        gate = st.session_state.pipeline.gate
        cooldown = gate.seconds_since_last_alert
        cooldown_text = f"{cooldown:.0f}s ago" if cooldown is not None else "never fired"
        st.markdown(_metric_card("Consecutive elevated readings", str(gate.consecutive_count)), unsafe_allow_html=True)
        st.markdown(_metric_card("Last alert", cooldown_text), unsafe_allow_html=True)
    with col3:
        if result.reading:
            r = result.reading
            st.markdown(
                _metric_card(
                    "Current vitals",
                    f"SpO2 {r['spo2']:.1f}% &middot; HR {r['hr']:.0f}bpm<br>"
                    f"Temp {r['temp']:.1f}°C &middot; Altitude {r['altitude']:.0f}m",
                ),
                unsafe_allow_html=True,
            )

    if result.alert_fired:
        # A brief full-width red flash under the cards, distinct from (and in
        # addition to) the modal ALERT! popup -- the popup is a one-shot
        # event consumed on the next rerun (see render_alert_popup()), but
        # this banner keeps showing for as long as result stays the tick
        # that fired, so a user who dismisses the popup still sees that an
        # alert happened without needing to check the Alert log tab.
        st.markdown(
            "<div style='padding:14px 18px;background:#8B2E2E;color:white;border-radius:6px;"
            "font-weight:700;text-align:center;margin:8px 0;animation:cinemind-flash 1s ease-in-out 2;'>"
            f"🚨 ALERT DISPATCHED -- hysteresis gate passed. "
            f"Telegram send: {'succeeded' if result.alert_result['sent'] else 'failed/fallback (see alert log)'}"
            "</div>"
            "<style>@keyframes cinemind-flash {0%,100%{opacity:1;} 50%{opacity:0.4;}}</style>",
            unsafe_allow_html=True,
        )

    st.markdown("**LLM interpretation:**")
    st.markdown(result.explanation or "_(no explanation yet)_")


def render_live_plots() -> None:
    st.subheader("Live vitals")
    history = st.session_state.history
    if not history:
        st.caption("No data yet.")
        return

    df = pd.DataFrame(history)
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(df.set_index("timestamp")[["spo2"]], height=200)
        st.caption("SpO2 (%)")
    with col2:
        st.line_chart(df.set_index("timestamp")[["hr"]], height=200)
        st.caption("Heart rate (bpm)")

    col3, col4 = st.columns(2)
    with col3:
        st.line_chart(df.set_index("timestamp")[["altitude"]], height=200)
        st.caption("Altitude (m)")
    with col4:
        if "severity_index" in df.columns:
            st.line_chart(df.set_index("timestamp")[["severity_index"]], height=200)
            st.caption(f"Severity index (0={SEVERITY_TIERS[0]} .. 4={SEVERITY_TIERS[-1]})")


def render_chat_panel() -> None:
    st.subheader("Ask about this reading")
    result = st.session_state.last_result
    if result is None or result.severity is None:
        st.caption("Start classification first to enable chat.")
        return

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the current reading...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # History passed to the LLM excludes the ground-truth severity
        # context (re-injected fresh every call, see llm_chat.chat's
        # docstring) -- only the user/assistant turns themselves.
        reply = llm_chat_fn(
            user_input,
            result.severity["severity_label"],
            result.severity["confidence"],
            result.reading,
            history=st.session_state.chat_messages[:-1],
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def render_alert_log() -> None:
    st.subheader("Alert log")
    records = read_alert_log(limit=20)
    if not records:
        st.caption("No alerts fired yet this session (or ever, on this machine).")
        return
    for record in records:
        icon = "✅" if record["sent"] else "⚪"
        with st.expander(f"{icon} {record['timestamp']}"):
            st.text(record["message"])
            if record["error"]:
                st.caption(f"Note: {record['error']}")


# ---------------------------------------------------------------------------
# System Info tab -- static reference content (architecture, thresholds,
# severity tiers, disease background). Doesn't touch st.session_state or the
# pipeline at all -- pure documentation, safe to render regardless of whether
# a Demo Mode source has been started yet.
# ---------------------------------------------------------------------------
def render_system_info() -> None:
    st.subheader("System architecture")
    st.markdown(
        "```\n"
        "Demo Mode UI (scenario picker / manual sliders / Harespod replay)\n"
        "        -> DataSource interface\n"
        "        -> Feature Engineering (rolling buffer -> trend features)\n"
        "        -> ML Severity Classifier (XGBoost, ordinal)\n"
        "        -> Hysteresis Gate (persistence + cooldown check)\n"
        "        -> [LLM Interpreter (Groq)]  +  [Telegram Alert (if gate passes)]\n"
        "        -> Streamlit Dashboard (live plots, chat, alert log)\n"
        "```"
    )
    st.caption(
        "Every reading passes through all 7 stages above once per `.tick()` call. "
        "See `Architecture_Diagrams/01_system_architecture.html` for the full interactive diagram."
    )

    st.divider()
    st.subheader("Severity tiers and disease background")
    st.caption(
        "Mapped from the Lake Louise Score (a symptom self-report questionnaire) to a "
        "literature-informed sensor pattern -- see `docs/lls_mapping.md` for full sourcing. "
        "**Not a validated clinical instrument.**"
    )
    tier_info = [
        (
            "Normal", SEVERITY_COLORS["Normal"],
            "No meaningful symptoms (LLS 0-2).",
            "SpO2 within ~3% of altitude-expected baseline, HR at/near resting range, "
            "normal temp, no adverse trend.",
        ),
        (
            "Mild AMS", SEVERITY_COLORS["Mild AMS"],
            "Acute Mountain Sickness -- headache plus at least one mild symptom "
            "(nausea, fatigue, dizziness, sleep disturbance). The most common altitude "
            "illness; usually resolves with rest and no further ascent.",
            "SpO2 ~4-8% below altitude-expected, HR +10-20bpm over resting, symptoms "
            "plateau rather than worsen.",
        ),
        (
            "Severe AMS", SEVERITY_COLORS["Severe AMS"],
            "Headache plus multiple or severe symptoms with functional impairment -- "
            "still AMS, but bad enough to interfere with normal activity.",
            "SpO2 ~8-14% below expected, HR persistently +20-35bpm, mild temp elevation "
            "more common, and a negative trend (worsening, not just low).",
        ),
        (
            "HAPE risk", SEVERITY_COLORS["HAPE risk"],
            "High-Altitude Pulmonary Edema -- fluid builds up in the lungs. Signs include "
            "breathlessness at rest, persistent cough, and chest tightness layered on AMS. "
            "Can progress fast and is life-threatening without descent/oxygen/treatment.",
            "SpO2 sharply below expected (>14% delta) and/or a steep negative 5-minute "
            "trend even before the absolute value looks extreme -- early HAPE can "
            "desaturate quickly, so the trend matters as much as the number.",
        ),
        (
            "HACE risk", SEVERITY_COLORS["HACE risk"],
            "High-Altitude Cerebral Edema -- swelling in the brain. Signs include ataxia "
            "(loss of coordination/balance) and altered consciousness layered on AMS. "
            "The most dangerous and rapidly progressing altitude illness; a medical "
            "emergency requiring immediate descent.",
            "Severe SpO2 depression similar to or worse than HAPE risk, but the defining "
            "signal is a sustained trajectory that doesn't plateau or recover, combined "
            "with the highest heart-rate elevation band -- modeling a late, decompensating state.",
        ),
    ]
    for name, color, clinical, sensor in tier_info:
        st.markdown(
            f"<div style='padding:12px 16px;margin-bottom:8px;border-left:4px solid {color};"
            f"background:rgba(128,128,128,0.06);border-radius:6px;'>"
            f"<b>{name}</b><br>"
            f"<span style='font-size:13px;'>{clinical}</span><br>"
            f"<span style='font-size:12px;color:#8FA3B8;'>Sensor pattern: {sensor}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Thresholds actually used by the model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Clinical reference constants** (`src/config.py`)")
        st.markdown(
            "- SpO2 sea-level baseline: **98.0%**\n"
            "- Expected SpO2 drop: **~3% per 1000m** altitude gained\n"
            "- Normal resting HR range: **50-100 bpm**\n"
            "- Normal temperature range: **36.1-37.2°C**\n"
            "- AMS risk becomes relevant above: **2500m** altitude"
        )
    with col2:
        st.markdown("**Hysteresis alert gate** (`src/alerts/hysteresis.py`)")
        st.markdown(
            "- Alert-eligible at tier >= **Severe AMS**\n"
            "- Must sustain for **3 consecutive readings** in a row (any dip resets the streak)\n"
            "- Cooldown between repeat alerts: **15 minutes**\n"
            "- Both conditions must hold -- a single noisy reading never fires an alert"
        )
    st.caption(
        "These aren't diagnostic cutoffs -- they're literature-informed engineering "
        "approximations tuned so the demo behaves sensibly. See the **Limitations** "
        "section of the README and `docs/lls_mapping.md`'s \"Known limitation\" writeup "
        "for the honest under-triage-vs-false-alarm tradeoff this gate makes."
    )


# ---------------------------------------------------------------------------
# Roadmap tab -- future improvements. Kept in sync with the README's own
# "Future improvements" section; update both together if this list changes.
# ---------------------------------------------------------------------------
def render_roadmap() -> None:
    st.subheader("Future improvements")
    st.caption(
        "This is a Demo Mode prototype (see CLAUDE.md). None of the items below are "
        "built yet -- listed here as the honest next steps, not claims."
    )

    roadmap_items = [
        (
            "📡 GPS tracking",
            "Attach real GPS coordinates (and altitude derived from GPS/barometric fusion, "
            "not just a manual slider or chamber marker) to every reading, so the dashboard "
            "and alerts can show *where* a subject is, not just their vitals. Would let the "
            "Telegram alert include a live location link for a real rescue response.",
        ),
        (
            "💬 Telegram query bot (ask the system your own vitals)",
            "`src/alerts/alert_bot.py` today is **send-only** -- it dispatches alerts but "
            "never listens for incoming messages. A real improvement: add an inbound "
            "listener (`python-telegram-bot`'s `Application`/`CommandHandler`, e.g. "
            "`/status` or `/vitals`) so a user (or their medical contact) can message the "
            "bot directly and get the current severity, vitals, and trend back on demand, "
            "instead of only receiving alerts the hysteresis gate decides to push.",
        ),
        (
            "🔌 Hardware integration",
            "Everything currently runs against synthetic/replayed data behind the "
            "`DataSource` interface (`src/datasource/base.py`). The interface was built "
            "specifically so a future `arduino_reader.py` reading real MAX30102 "
            "(SpO2/HR) and barometric altitude sensors over serial is a drop-in swap, "
            "not a rewrite -- see CLAUDE.md Section 2A.",
        ),
        (
            "🧠 Fixing the under-triage gap",
            "~24% of test-set predictions underestimate true severity (45% for the "
            "most dangerous HAPE/HACE tiers specifically). Two threshold-tuning fixes "
            "were tried and reverted because they caused false alarms on genuinely "
            "normal readings (see `docs/lls_mapping.md`). A real fix likely needs either "
            "a differently-trained model or a second, independent confirming signal "
            "(e.g. requiring the LLM's interpretation to agree before alerting).",
        ),
        (
            "👤 Per-subject personal baselines",
            "The rule baseline and current features compare against a population-normal "
            "HR/SpO2 band, not an individual's own resting values. Learning (or letting a "
            "user enter) a personal baseline during a calm period could sharpen "
            "sensitivity for people whose normal readings sit near the edges of the "
            "population range.",
        ),
        (
            "🩺 Wearable-grade sensor fusion",
            "Add respiratory rate (already loaded but unused from the pilot-altitude "
            "dataset, see `src/data/pilot_altitude_loader.py`) and blood pressure as "
            "additional model features -- both are clinically relevant to AMS/HAPE/HACE "
            "and could improve separation between adjacent tiers.",
        ),
        (
            "🗺️ Multi-user / expedition mode",
            "Currently one `MedicalAlertPipeline` instance monitors one subject. A group "
            "expedition use case would need multiple concurrent subjects, a roster view, "
            "and alerts that identify *which* team member triggered them.",
        ),
        (
            "☁️ Cloud persistence + auth",
            "Alert logs and history currently live in a local JSONL file / browser "
            "session. A real deployment would want a proper database, user accounts, and "
            "historical trend views across multiple expeditions/sessions.",
        ),
        (
            "📴 Offline-first operation",
            "High-altitude expeditions often have no connectivity. A real hardware phase "
            "would need the ML classification and hysteresis gate to keep working fully "
            "offline, queuing Telegram alerts and LLM calls for whenever connectivity "
            "returns (both already degrade gracefully to a local fallback today -- this "
            "would extend that to an explicit offline queue/retry mechanism).",
        ),
    ]
    for title, desc in roadmap_items:
        with st.expander(title):
            st.markdown(desc)


# ---------------------------------------------------------------------------
# ALERT! popup -- st.dialog renders as a modal overlay on top of the whole
# page, regardless of which tab is active, which is the point: a hysteresis
# alert is exactly the kind of event that shouldn't be missable just because
# the user happened to be on the Chat tab when it fired.
# ---------------------------------------------------------------------------
@st.dialog("🚨 ALERT!")
def _show_alert_dialog(severity_label: str, reading: dict) -> None:
    color = SEVERITY_COLORS.get(severity_label, "#8FA3B8")
    st.markdown(
        f"<div style='padding:14px 18px;border-left:6px solid {color};"
        f"background:rgba(128,128,128,0.08);border-radius:6px;margin-bottom:14px;"
        f"animation:cinemind-border-pulse 0.8s ease-in-out 3;'>"
        f"<div style='font-size:13px;color:#8FA3B8;'>CONDITION DETECTED</div>"
        f"<div style='font-size:26px;font-weight:800;'>{severity_label}</div>"
        f"</div>"
        f"<style>@keyframes cinemind-border-pulse "
        f"{{0%,100%{{border-left-color:{color};}} 50%{{border-left-color:#ffffff;}}}}</style>",
        unsafe_allow_html=True,
    )
    # Synthesize a short two-tone beep with the Web Audio API rather than
    # embedding a base64 audio file -- a few lines of JS instead of a large
    # opaque data blob in the source. Browsers commonly block unmuted
    # autoplay audio with no prior user gesture on the page; this degrades
    # silently (caught and swallowed) if so, same as st.dialog itself never
    # crashing the app over a cosmetic feature -- the visual pulse above and
    # the popup itself are the primary, always-working cues regardless.
    st.components.v1.html(
        """
        <script>
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            [880, 660].forEach((freq, i) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.frequency.value = freq;
                osc.type = "sine";
                gain.gain.setValueAtTime(0.15, ctx.currentTime + i * 0.18);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.18 + 0.15);
                osc.connect(gain).connect(ctx.destination);
                osc.start(ctx.currentTime + i * 0.18);
                osc.stop(ctx.currentTime + i * 0.18 + 0.15);
            });
        } catch (e) { /* autoplay blocked or no audio support -- fine, visual cues still show */ }
        </script>
        """,
        height=0,
    )
    if reading:
        st.markdown(
            f"**SpO2:** {reading.get('spo2', 0):.1f}% &nbsp;&nbsp; "
            f"**HR:** {reading.get('hr', 0):.0f}bpm &nbsp;&nbsp; "
            f"**Temp:** {reading.get('temp', 0):.1f}°C &nbsp;&nbsp; "
            f"**Altitude:** {reading.get('altitude', 0):.0f}m"
        )
    st.markdown("**Immediate steps to take:**")
    for step in IMMEDIATE_STEPS.get(severity_label, []):
        st.markdown(f"- {step}")
    st.caption(
        "Prototype / learning project -- NOT a certified medical device. This is "
        "general guidance, not a diagnosis or treatment plan. Always follow real "
        "medical advice and your own judgment on the ground."
    )
    if st.button("Acknowledge", type="primary", use_container_width=True):
        st.rerun()


def render_alert_popup() -> None:
    """
    Called once per script run, near the top of main(). Consumes (clears)
    st.session_state.pending_alert if set, so the same alert never pops up
    twice -- e.g. if the user switches tabs after acknowledging it, there's
    nothing left in pending_alert to re-trigger the dialog on that next rerun.
    """
    pending = st.session_state.get("pending_alert")
    if pending is not None:
        st.session_state.pending_alert = None
        _show_alert_dialog(pending["severity_label"], pending["reading"])


def main() -> None:
    _init_session_state()
    render_alert_popup()  # must run before the tabs render, so the modal overlays whichever tab is active

    st.title("High-Altitude Medical Alert System")
    st.caption(
        "Prototype / learning project. Not a certified medical device. "
        "Severity labels are derived from a literature-based mapping, not clinical diagnoses -- see docs/lls_mapping.md."
    )

    render_demo_mode_panel()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Dashboard", "Chat", "Alert log", "System info", "Roadmap"]
    )
    with tab1:
        render_severity_readout()
        st.divider()
        render_live_plots()
    with tab2:
        render_chat_panel()
    with tab3:
        render_alert_log()
    with tab4:
        render_system_info()
    with tab5:
        render_roadmap()

    if st.session_state.auto_play and st.session_state.pipeline is not None:
        _do_tick()
        time.sleep(0.3)  # brief pause so the UI is watchable, not a 1:1 real-time simulation
        st.rerun()  # forces Streamlit to immediately re-run this whole script again, creating the "auto-play" illusion out of repeated single ticks


if __name__ == "__main__":
    main()
