# CLAUDE.md — High-Altitude Medical Alert System

This file is the primary reference for Claude Code (or any agent) working on this project.
Read this fully before writing code. Keep it updated as the project evolves — it is the
source of truth for scope, decisions, and architecture.

---

## 1. Project Summary

A sensor-based early-warning system for high-altitude illness (AMS, HAPE, HACE). An
Arduino Uno collects SpO₂, heart rate, body temperature, and altitude (via pressure).
A Python gateway feeds this into an ML severity classifier, an LLM interprets the result
for a live chat interface, and a Telegram bot alerts a medical facility contact in severe
cases.

**This is a prototype / learning project, not a certified medical device.** Say so in the
UI and README. The person building this is learning ML/DL/LLM concepts alongside the
build — favor clear, well-commented code over cleverness, and prefer explaining *why*
in comments over silent one-liners for anything non-trivial.

**Current build scope is software-only.** Physical hardware (Arduino + sensors) is a
separate, later phase — see Section 2A. For now, the entire pipeline runs in **Demo
Mode**, driven by synthetic and replayed data injected through the web UI itself, not
by a serial connection. Nothing in Phase 1–2 should assume a physical device is
present or block on one.

**Owner's context:** Learning AI/ML. Wants things explained, not just delivered. When
making a nontrivial design choice while coding, add a short comment explaining the
concept (e.g. why a temporal split instead of random split) rather than assuming
familiarity.

---

## 2. Current Phase: Software-Only, Demo Mode

**No physical hardware exists yet and none is required for Phase 1–2.** The system must
be fully demonstrable end-to-end using synthetic and replayed data injected directly
through the web UI. This is the primary deliverable of the current 10-day build.

**Demo Mode requirements:**
- The Streamlit UI includes a **Demo Mode panel** where the user can:
  - Select a pre-built scenario (Normal / Mild AMS / Severe AMS / HAPE onset / HACE risk)
    and watch it play forward through the pipeline in real time (simulated ticks)
  - Manually adjust SpO₂ / HR / temp / altitude with sliders and see the ML
    classification + LLM interpretation update live, for exploring edge cases
  - Replay a real Harespod recording end to end
- Data still flows through the *exact same* pipeline a real sensor stream would use —
  feature engineering, ML classification, hysteresis gate, LLM interpretation, Telegram
  alerts. Demo Mode only replaces the data **source** (Stage 1 in the data flow
  diagram), nothing downstream.
- Build the data source as a swappable interface (e.g. a simple `DataSource` class/
  generator with `.next_reading()`) so that a real serial-based source can be dropped in
  later without touching Stages 2–7.

This means: **do not build `arduino_reader.py` / pyserial integration in the current
phase.** Build `demo_data_source.py` instead (scenario player + manual override +
Harespod replay), and design its interface so Phase 3 (hardware) can implement the same
interface against a real Arduino.

---

## 2A. Future Phase — Hardware Build (not in current scope)

Hardware design is already complete in schematik.io. This phase begins **after** the
software system (Phase 1–2) is working end to end in Demo Mode and the owner explicitly
asks to proceed. Do not start this work proactively.

| Component | Role | I2C Addr |
|---|---|---|
| Arduino Uno | MCU | — |
| MAX30102 | SpO₂ + Heart Rate | 0x57 |
| MLX90614 | Contactless body temp | 0x5A |
| BMP280 | Pressure → altitude | 0x76 |
| SSD1306 OLED | Local 3-page display | 0x3C |
| PCA9306 | I2C level shifter (5V Uno ↔ 3.3V sensors) | — |

All sensors on shared I2C bus (A4=SDA, A5=SCL). Do **not** remove the PCA9306 — MAX30102
and MLX90614 are not 5V tolerant. No GPS, no local storage, no battery — explicitly out
of scope per project owner.

When this phase starts: build `arduino_reader.py` (pyserial, 115200 baud) implementing
the same `DataSource` interface as `demo_data_source.py`, so it drops in as a straight
swap with no changes to Stages 2–7 of the pipeline.

---

## 3. Software Architecture (see `/Architecture_Diagrams/01_system_architecture.html`)

```
Demo Mode UI (scenario picker / manual sliders / Harespod replay)
        → DataSource interface (demo_data_source.py)
        → Feature Engineering (rolling buffer → trend features)
        → ML Severity Classifier (XGBoost, ordinal)
        → Hysteresis Gate (persistence + cooldown check)
        → [LLM Interpreter (Groq)]  +  [Telegram Alert (if gate passes)]
        → Streamlit Dashboard (live plots, chat, alert log)
```

Future hardware phase only changes the first arrow:
`Sensors → Arduino → Serial → arduino_reader.py → DataSource interface → (same as above)`

**Deployment phasing (explicit decision — do not skip ahead):**
- **Phase 1 (Days 1–7):** Everything local on Windows, software-only, no cloud, no
  hardware. Build data pipeline + ML models against Harespod + synthetic data.
- **Phase 2 (Days 8–14):** Still local, still software-only. Full pipeline integrated
  end to end + Streamlit UI with Demo Mode + Telegram alerts. **This is the deliverable
  for the current 10-day timeline.**
- **Phase 3 (future, out of current scope):** Physical hardware build — Arduino +
  sensors, implementing the same `DataSource` interface. Begins only when explicitly
  requested after Phase 2 is complete and working.
- **Phase 4 (further future, out of current scope):** Extract ML + LLM + alert logic
  into a FastAPI cloud backend (Render/Railway free tier). Laptop or Raspberry Pi
  becomes a thin gateway. Do not build this until asked.

---

## 4. ML Target — Ordinal Classification

**5 classes, ordered:** `Normal < Mild AMS < Severe AMS < HAPE risk < HACE risk`

This is **ordinal classification**, not plain multi-class — the ordering matters because
misclassifying HAPE as Normal is far worse than misclassifying it as Severe AMS. See the
in-chat explanation delivered to the owner for the full rationale; the short version:
use either (a) a cascade of binary "at least this severe?" classifiers, or (b) a
regression model with severity thresholds binned after the fact. **Default to approach
(b) with XGBoost regression + threshold binning** — simpler to implement and tune, and
it's the primary model per the decision log below.

**Labels are derived, not ground-truth clinical diagnoses.** They come from a
literature-derived mapping of the Lake Louise Score (LLS) to sensor patterns (see
`docs/lls_mapping.md` once written in Phase 2/3). Document this limitation in the
README — it matters for how confidently the system's output should be presented.

---

## 5. Dataset Plan

**Primary source (as originally planned):** synthetic generation (`src/data/synth_data.py`)
— built first per the "synthetic-first, adapter ready" decision, and still the majority
of the training data by row count. **Real data (added after initial build, once
downloaded):** Harespod (15 subjects, hypobaric chamber, ambient air, SpO₂/HR at 100Hz
downsampled to 1Hz — `src/data/harespod_loader.py`), merged into the training split only.
Downloaded from Figshare (DOI: 10.6084/m9.figshare.c.6623344.v1) + companion code at
https://github.com/oca-john/Harespod.

**A second real dataset was investigated and deliberately NOT used for labeling:** the
"High-Altitude Pilot Physiological Monitoring Dataset" (Figshare DOI:
10.6084/m9.figshare.29947679, `src/data/pilot_altitude_loader.py`, 20 subjects, 4500–7500m,
real absolute units, no rescaling needed). It's loaded and available for exploration, but
excluded from severity labeling/training: its subjects breathed oxygen-enriched air
(~40–45% O₂) throughout, a fundamentally different physiological scenario than the
ambient-air hypoxia this project models — confirmed via the dataset's own O2/N2/CO2
columns. Harespod, by contrast, is confirmed ambient-air per its paper's Methods (ambient
pressure reduction only; real oxygen used solely as an emergency intervention).

**Gaps in Harespod, and how each is filled:**
| Gap | Fill |
|---|---|
| No body temperature | Synthesized from literature-based distributions conditioned on inferred severity (`synth_data.synthesize_temperature`, reusing the same `TIER_PROFILES` bands as pure-synthetic generation) |
| No continuous pressure/altitude | Interpolated from Harespod's `key_timestamp.txt` altitude-change markers, with a detected-and-corrected 8-hour GMT/local-time offset (affects 6 of 15 subjects) and two per-subject format exceptions (314c's differently-worded first marker, 328a's truncated 5-stage protocol) — see `harespod_loader.py`'s docstrings |
| Values normalized to [0,1] per subject, original min/max not recoverable | Rescaled onto a physiologically-plausible range (SpO₂ 70–100%, HR/pulse 40–160bpm) rather than the device's raw 0–100/0–250 collection range (that rescale was tried first and rejected — it produced clinically implausible numbers, e.g. mean HR of 174bpm) — an approximation, not a recovery of true values, documented as such |
| No HAPE/HACE cases (ethically can't induce in subjects) | Fully synthetic trajectories built from clinical literature ranges |
| No severity labels | LLS-derived rule mapping applied post-hoc (`src/data/label_real_data.py`, reusing `rule_baseline.classify_row()`) |

**Pipeline (as actually built):**
1. Load + clean + downsample Harespod to 1Hz (matches realistic Arduino sample rate)
2. Interpolate continuous altitude from markers (with the GMT-offset + per-subject exception handling above)
3. Rescale normalized signals to physiological units; synthesize a temperature column
4. Generate fully synthetic HAPE/HACE trajectories (gradual onset, not step-function)
5. Apply LLS-derived severity labeling rules to both synthetic (generation-time) and real (post-hoc) data
6. Feature engineering (trend features, ascent rate, expected-vs-actual SpO₂ delta)
7. **Temporal train/val/test split — never random-shuffle split on time-series data.**
   Split by time blocks so validation/test represent "the future" relative to train.
8. **Merge real (rule-labeled) data into the TRAIN split only** — val/test stay 100%
   synthetic, specifically so evaluating the rule-based baseline against them is never
   circular (real labels were generated by that same rule logic; see
   `feature_engineering.run_pipeline()`'s docstring for the full reasoning). A documented
   alternative — splitting real data across all three sets and scoring the rule baseline
   on synthetic-only rows — was considered and intentionally not built.

**Class imbalance is expected and real** (Normal will dominate). Use class weighting in
XGBoost (`scale_pos_weight` / per-class weights) and/or oversampling for rare classes —
do not just accept a model that always predicts "Normal."

---

## 6. Models — What We're Actually Using

| Purpose | Model | Why |
|---|---|---|
| Severity classification (primary) | **XGBoost** (regression + threshold binning, or ordinal-aware boosting) | Best fit for small tabular data, fast to train/tune, interpretable feature importances, easy to serialize and deploy |
| Severity classification (comparison) | **LSTM** (PyTorch, small, 1-2 layers) | Built as a comparison model on 60-second rolling windows to demonstrate DL approach; keep whichever performs better in practice — XGBoost is favored default for small datasets like this |
| Baseline (build first, always) | Rule-based / clinical threshold classifier | Sanity floor — if ML doesn't clearly beat this, something is wrong with features or labels |
| Conversational + alert interpretation | **gpt-oss-120b via Groq API** (originally Llama 3.3 70B) | Switched from 70B after genuinely hitting its free-tier daily limit during testing; gpt-oss-120b's free tier is far more generous. Being a reasoning model, it spends a variable share of `max_tokens` on internal reasoning before visible output (70B did not) — `max_tokens` was raised from 200/300 to 600 across `src/llm/llm_chat.py`'s three functions as part of the switch, after 4/5 calls truncated mid-sentence at the old values. The adversarial guardrail test (refuse dosing, refuse overriding the ML classification) was re-verified and still passes. |

Do not add additional models beyond this set without discussing with the project owner
— the goal is a complete, well-understood pipeline, not a model zoo.

---

## 7. LLM Integration — Guardrails (non-negotiable)

The LLM's job is to **interpret** the ML model's severity output and sensor readings in
plain language, and hold a live chat. It must **never**:
- Present itself as making a medical diagnosis
- Override or contradict the ML severity classification
- Give specific treatment/medication dosing advice
- Discourage the user from seeking real medical help when severity is elevated

System prompt must explicitly state these boundaries (full primer given to the owner in
chat — implement it as written there). Always pass the ML model's severity + confidence
into the prompt as ground truth context; the LLM explains it, it doesn't invent its own
assessment from raw numbers.

**Structured output requirement:** For alert generation, request JSON output from Groq
(response format enforced) so the alert message is machine-parseable, not free text that
might omit required fields (severity, key vitals, timestamp).

---

## 8. Alerts — Telegram Bot

- Free, no API cost, created via @BotFather.
- Alert fires only when the **hysteresis gate** passes (see data flow diagram):
  severity sustained above threshold for N consecutive readings + cooldown elapsed since
  last alert. A single noisy reading must never trigger an alert.
- Alert message includes: severity tier, current vitals snapshot, LLM-generated summary,
  timestamp. Keep format consistent — define it once in `alert_bot.py` and reuse.

---

## 9. Tech Stack

Python 3.11, pandas, numpy, scikit-learn, XGBoost, PyTorch, pyserial, groq (official
SDK), python-telegram-bot, Streamlit, matplotlib. Windows development environment.

---

## 10. Repo Structure (target)

```
high-altitude-medical-alert/
├── CLAUDE.md                          # this file
├── README.md                          # written in Phase 2 (Day 14)
├── Architecture_Diagrams/
│   ├── 01_system_architecture.html
│   ├── 02_project_phases.html
│   └── 03_data_flow_diagram.html
├── data/
│   ├── raw/harespod/                  # downloaded, untouched
│   ├── synthetic/                     # generated
│   └── processed/                     # merged, feature-engineered, split
├── notebooks/                         # exploration (Days 1-3)
├── src/
│   ├── data/                          # loaders, synth generator, feature engineering
│   ├── models/                        # training scripts, saved models
│   ├── datasource/                    # demo_data_source.py (DataSource interface)
│   │                                   #   — arduino_reader.py added here in future
│   │                                   #     hardware phase, same interface
│   ├── llm/                           # llm_chat.py, prompts/
│   ├── alerts/                        # alert_bot.py, hysteresis logic
│   └── app/                           # streamlit_app.py (includes Demo Mode panel)
├── firmware/                          # Arduino .ino from schematik.io — FUTURE PHASE,
│                                       #   not touched until hardware phase begins
└── tests/
```

---

## 11. Architecture Diagrams — Maintenance Instructions

**This is a standing instruction, not a one-time task.** The three files in
`Architecture_Diagrams/` are living documents:

- `01_system_architecture.html` — update whenever a component is added, removed, or a
  layer's responsibility changes (e.g. if the ML model changes, if a new alert channel
  is added).
- `02_project_phases.html` — update the `status` class/text on each `.day` block
  (`pending` → `progress` → `done`) as work completes. Update the footer's revision
  note. This should reflect reality at all times, not just at project end.
- `03_data_flow_diagram.html` — update if the pipeline order changes, if hysteresis
  thresholds change materially, or if a new stage is added (e.g. cloud deployment in
  Phase 3).

When in doubt, prefer editing these files over leaving them stale. If a new diagram
would help (e.g. a dataset schema diagram, a model comparison chart), add it to
`Architecture_Diagrams/` with a new numbered filename and list it here. and keep them updated to .

---

## 12. Decision Log

| Decision | Choice | Status |
|---|---|---|
| Illness scope | AMS + HAPE + HACE, ordinal 5-tier | Confirmed |
| Architecture | Local-first, cloud later | Confirmed |
| LLM provider | Groq (originally Llama 3.3 70B, switched to gpt-oss-120b after hitting 70B's free-tier daily limit) | Confirmed, API key obtained and verified working; switch re-verified against the same adversarial guardrail test |
| Dataset | Synthetic (primary) + Harespod (real, train-only, rescaled+rule-labeled) | Confirmed, both downloaded and integrated |
| Pilot altitude dataset (2nd real source) | Investigated, loaded, NOT used for labeling — oxygen-enriched protocol, not ambient air | Confirmed exclusion, kept for exploration only |
| Alert channel | Telegram bot | Confirmed, bot token + chat_id obtained and verified with a live test message |
| Primary ML model | XGBoost (ordinal via regression+thresholds) | Confirmed |
| Comparison model | LSTM (PyTorch) | Confirmed, for learning comparison |
| Timeline | 10 days, ~3hrs/day, flexible pace | Confirmed |
| GPS / local storage / battery | Explicitly excluded | Confirmed |
| Hardware build (Arduino + sensors) | Deferred to future Phase 3 — software/demo only for now | Confirmed |
| Data source for current phase | Demo Mode via UI (scenarios, sliders, Harespod replay) — no serial/Arduino | Confirmed |

Update this table if the owner changes a decision mid-build — don't silently diverge
from what's documented here.

---

## 13. What NOT to do

- **Don't build the physical hardware integration (Phase 3, arduino_reader.py, pyserial)
  unprompted — current scope is software/Demo Mode only.**
- Don't build the cloud deployment (Phase 4) unprompted.
- Don't add hardware components not in the schematik.io design (no GPS/storage/battery).
- Don't let the LLM make diagnostic claims — it interprets, the ML model classifies.
- Don't do a random shuffle train/test split on time-series data.
- Don't skip the rule-based baseline model — it's the sanity check for everything after.
- Don't let a single noisy reading trigger a Telegram alert — hysteresis gate is required.
- Don't hardcode the data source — always go through the `DataSource` interface so the
  future hardware phase is a drop-in swap, not a rewrite.
