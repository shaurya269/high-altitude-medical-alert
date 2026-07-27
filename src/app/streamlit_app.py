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


def _new_pipeline(source) -> None:
    st.session_state.pipeline = MedicalAlertPipeline(source)
    st.session_state.history = []
    st.session_state.chat_messages = []
    st.session_state.last_result = None
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
            st.session_state.auto_play = False

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


# ---------------------------------------------------------------------------
# Main panel: severity readout, live plots, chat, alert log
# ---------------------------------------------------------------------------
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

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(
            f"<div style='padding:16px;border-left:6px solid {color};background:rgba(128,128,128,0.08);"
            f"border-radius:6px;'><div style='font-size:13px;color:#8FA3B8;'>SEVERITY</div>"
            f"<div style='font-size:28px;font-weight:700;'>{severity['severity_label']}</div>"
            f"<div style='font-size:13px;color:#8FA3B8;'>confidence {severity['confidence']:.0%} "
            f"&middot; model: {severity['model_used']}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        gate = st.session_state.pipeline.gate
        st.metric("Consecutive elevated readings", f"{gate.consecutive_count}")
        cooldown = gate.seconds_since_last_alert
        st.metric("Since last alert", f"{cooldown:.0f}s" if cooldown is not None else "never fired")
    with col3:
        if result.reading:
            r = result.reading
            st.markdown(
                f"**SpO2:** {r['spo2']:.1f}% &nbsp;&nbsp; **HR:** {r['hr']:.0f}bpm  \n"
                f"**Temp:** {r['temp']:.1f}°C &nbsp;&nbsp; **Altitude:** {r['altitude']:.0f}m"
            )

    if result.alert_fired:
        st.error(
            f"🚨 ALERT DISPATCHED -- hysteresis gate passed. "
            f"Telegram send: {'succeeded' if result.alert_result['sent'] else 'failed/fallback (see alert log)'}"
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


def main() -> None:
    _init_session_state()

    st.title("High-Altitude Medical Alert System")
    st.caption(
        "Prototype / learning project. Not a certified medical device. "
        "Severity labels are derived from a literature-based mapping, not clinical diagnoses -- see docs/lls_mapping.md."
    )

    render_demo_mode_panel()

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Chat", "Alert log"])
    with tab1:
        render_severity_readout()
        st.divider()
        render_live_plots()
    with tab2:
        render_chat_panel()
    with tab3:
        render_alert_log()

    if st.session_state.auto_play and st.session_state.pipeline is not None:
        _do_tick()
        time.sleep(0.3)  # brief pause so the UI is watchable, not a 1:1 real-time simulation
        st.rerun()  # forces Streamlit to immediately re-run this whole script again, creating the "auto-play" illusion out of repeated single ticks


if __name__ == "__main__":
    main()
