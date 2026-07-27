# High-Altitude Medical Alert System

A sensor-based early-warning prototype for high-altitude illness (AMS, HAPE, HACE). An ML
severity classifier reads simulated (or, in a future phase, real Arduino) vitals, an LLM
interprets the result in plain language and holds a live chat, and a Telegram bot alerts a
medical contact when severity is sustained at a dangerous level.

**This is a prototype / learning project, not a certified medical device.** Severity labels
come from a literature-derived mapping (see [`docs/lls_mapping.md`](docs/lls_mapping.md)),
not clinical diagnoses. See [Limitations](#limitations) below before trusting any output.

---

## What's built

Everything in this repo runs in **Demo Mode** — a fully working sensor-to-alert pipeline
driven by synthetic scenarios, manual sliders, or replayed real recordings, with no physical
hardware required (see [`CLAUDE.md`](CLAUDE.md) for the full phased plan and why hardware is
deferred).

| Layer | What it does | Where |
|---|---|---|
| Data | Synthetic vitals generator (400 subjects, 5 severity tiers) + real Harespod hypobaric-chamber recordings (15 subjects, rule-labeled, merged into training only) | `src/data/` |
| Models | Rule-based baseline, XGBoost ordinal regressor (**selected**), LSTM comparison model — all compared on identical held-out data | `src/models/` |
| Demo data sources | Scenario player, manual sliders, Harespod replay — all behind one swappable `DataSource` interface | `src/datasource/` |
| LLM | Groq (gpt-oss-120b, originally Llama 3.3 70B): plain-language severity interpretation, live chat, structured-JSON alert content — with hard guardrails | `src/llm/` |
| Alerts | Hysteresis gate (sustained-severity + cooldown) → Telegram dispatch, with a local log fallback | `src/alerts/` |
| Pipeline | Wires all of the above into one `.tick()` call per reading | `src/pipeline.py` |
| Dashboard | Streamlit app: scenario picker, live vitals/severity plots, chat, alert log | `src/app/streamlit_app.py` |
| Docs | 10 teaching notebooks walking through every `src/data/` and `src/models/` module, an architecture diagram set, and this README | `notebooks/`, `Architecture_Diagrams/` |

Run `python -m pytest tests/ -v` for the full automated suite (67 tests as of this writing —
covering the ML pipeline, real-data loaders, the DataSource layer, the LLM's guardrails, the
Telegram bot, the hysteresis gate, full pipeline integration, and the Streamlit app itself via
`streamlit.testing.v1.AppTest`).

---

## Architecture

See `Architecture_Diagrams/01_system_architecture.html` for the full diagram (open it in any
browser). Short version:

```
Demo Mode UI (scenario picker / manual sliders / Harespod replay)
        → DataSource interface
        → Feature Engineering (rolling buffer → trend features)
        → ML Severity Classifier (XGBoost, ordinal)
        → Hysteresis Gate (persistence + cooldown check)
        → [LLM Interpreter (Groq)]  +  [Telegram Alert (if gate passes)]
        → Streamlit Dashboard (live plots, chat, alert log)
```

`Architecture_Diagrams/02_project_phases.html` tracks day-by-day build status;
`Architecture_Diagrams/03_data_flow_diagram.html` traces one reading through all 7 stages.

---

## Setup

**Requirements:** Python 3.12, Windows (developed and tested on Windows; should work
cross-platform since nothing here is Windows-specific, but only Windows has been verified).

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in what you have — **everything is optional**, the
system degrades gracefully without any of it:

```bash
# Groq API (free tier) — get a key at https://console.groq.com/keys
GROQ_API_KEY=

# Telegram bot — create via @BotFather, see .env.example for the full steps
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Without `GROQ_API_KEY`: the LLM layer falls back to templated (still guardrail-compliant)
explanations instead of live interpretation/chat.
Without `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`: alerts are logged locally to
`data/processed/alert_log.jsonl` instead of sent, and the Streamlit alert log tab reads from
the same file either way.

### Real dataset (optional, for real-data-informed training)

The synthetic generator is the primary dataset and works with zero setup. If you want to
include real sensor data in training:

1. Download Harespod from Figshare (DOI `10.6084/m9.figshare.c.6623344.v1`) and unzip it into
   `data/raw/harespod/`.
2. Re-run the data + training pipeline (below) — `src/data/harespod_loader.py` will pick it up
   automatically and merge it into the training split (never validation/test — see
   [`docs/lls_mapping.md`](docs/lls_mapping.md) for why).

A second dataset (`src/data/pilot_altitude_loader.py`, a pilot physiological monitoring
dataset) is also supported for loading/exploration, but is **deliberately excluded** from
severity labeling — it uses an oxygen-enriched protocol that doesn't match this project's
ambient-air hypoxia model. See that module's docstring and `docs/lls_mapping.md`.

---

## Running things

**Regenerate the full data + model pipeline from scratch:**
```bash
python -m src.data.synth_data              # generate synthetic dataset
python -m src.data.feature_engineering     # engineer features, temporal split, merge real data
python -m src.models.predict_severity      # train/compare baseline+XGBoost+LSTM, select winner
```

**Run the dashboard:**
```bash
streamlit run src/app/streamlit_app.py
```
Pick a Demo Mode source in the sidebar (scenario, manual sliders, or Harespod replay), then
step forward or auto-play. The severity readout, live charts, chat, and alert log all update
live.

**Run the test suite:**
```bash
python -m pytest tests/ -v
```

**Explore the teaching notebooks:**
```bash
jupyter notebook notebooks/
```
Start with `02_config_walkthrough.ipynb` — each notebook imports and explains one `src/`
module, with real outputs and the reasoning behind every non-obvious design choice.

---

## Verified benchmark results

On the held-out synthetic test set (never touched by real-data merging — see
`docs/lls_mapping.md`):

| Model | Precision (macro) | Recall (macro) | F1 (macro) | Mean Abs Tier Error | Under-triage rate |
|---|---|---|---|---|---|
| Rule-based baseline | 0.349 | 0.287 | 0.285 | 0.819 | 0.309 |
| **XGBoost Ordinal (selected)** | 0.386 | 0.381 | 0.354 | **0.604** | 0.244 |
| LSTM | 0.403 | 0.408 | 0.397 | 0.701 | 0.203 |

XGBoost wins on Mean Absolute Tier Error (the ordinal-aware metric that best reflects
clinical cost — see `src/models/metrics.py` / the `07_metrics_walkthrough.ipynb` notebook)
and clearly beats the rule-based sanity floor. See `src/models/artifacts/model_comparison.json`
for the machine-readable record.

All 15 real Harespod recordings replay through the complete pipeline with zero exceptions and
zero rejected readings (Day 13 testing — see `docs/lls_mapping.md`'s changelog-style notes).

---

## Limitations

Read this before trusting anything this system says.

- **Not a certified medical device.** This is a learning/portfolio project. Severity tiers
  are a literature-informed engineering approximation (see `docs/lls_mapping.md`), not a
  validated clinical instrument or diagnosis.
- **Known under-triage gap.** On the held-out test set, ~24% of predictions underestimate
  true severity, and specifically 45% of true HAPE-risk/HACE-risk readings predict below the
  Telegram alert's trigger tier. Two fixes were tried and reverted after they caused false
  alerts on genuinely normal readings — see `docs/lls_mapping.md`'s "Known limitation"
  section for the full investigation and what a real fix would require.
- **No real body-temperature sensor exists in any dataset used.** Temperature values for real
  (Harespod) data are synthesized from the same literature-based distributions used for
  synthetic data, conditioned on the (also rule-derived) severity label — never a genuine
  sensor reading.
- **Harespod's absolute vitals are an approximation, not ground truth.** The dataset's own
  values are min-max normalized per subject with the original scale unrecoverable; this
  project rescales onto a physiologically-plausible range rather than the device's literal
  collection range (which produced implausible numbers like a 174bpm average resting heart
  rate) — see `src/data/harespod_loader.py`'s docstring.
- **LLM guardrails are prompt-based, not architecturally enforced.** `src/llm/llm_chat.py`'s
  system prompt instructs the model never to diagnose, override the ML classification, give
  dosing advice, or discourage seeking real help — verified resistant to direct user pressure
  in manual testing, but this is model behavior (gpt-oss-120b, previously Llama 3.3 70B — the
  adversarial test was re-run and passed again after switching), not a hard guarantee.
- **No physical hardware exists.** Everything runs in Demo Mode against synthetic/replayed
  data. See `CLAUDE.md` Section 2A for the (not-yet-started) hardware phase design.

---

## Future improvements

None of the items below are built — listed here as honest next steps, not claims. The
Streamlit dashboard's **Roadmap** tab mirrors this list; keep both in sync if it changes.

- **GPS tracking** — attach real GPS coordinates (and altitude derived from GPS/barometric
  fusion, not just a manual slider or chamber marker) to every reading, so the dashboard and
  alerts can show *where* a subject is, not just their vitals. Would let the Telegram alert
  include a live location link for a real rescue response.
- **Telegram query bot (ask the system your own vitals)** — `src/alerts/alert_bot.py` today is
  **send-only**: it dispatches alerts but never listens for incoming messages. A real
  improvement is an inbound listener (`python-telegram-bot`'s `Application`/`CommandHandler`,
  e.g. `/status` or `/vitals`) so a user or their medical contact can message the bot directly
  and get the current severity, vitals, and trend back on demand, instead of only receiving
  alerts the hysteresis gate decides to push.
- **Hardware integration** — everything currently runs against synthetic/replayed data behind
  the `DataSource` interface (`src/datasource/base.py`), built specifically so a future
  `arduino_reader.py` reading real MAX30102 (SpO2/HR) and barometric altitude sensors over
  serial is a drop-in swap, not a rewrite. See `CLAUDE.md` Section 2A.
- **Fixing the under-triage gap** — ~24% of test-set predictions underestimate true severity
  (45% for the most dangerous HAPE/HACE tiers specifically). Two threshold-tuning fixes were
  tried and reverted because they caused false alarms on genuinely normal readings (see
  `docs/lls_mapping.md`). A real fix likely needs either a differently-trained model or a
  second, independent confirming signal (e.g. requiring the LLM's interpretation to agree
  before alerting).
- **Per-subject personal baselines** — the rule baseline and current features compare against
  a population-normal HR/SpO2 band, not an individual's own resting values. Learning (or
  letting a user enter) a personal baseline during a calm period could sharpen sensitivity for
  people whose normal readings sit near the edges of the population range.
- **Wearable-grade sensor fusion** — add respiratory rate (already loaded but unused from the
  pilot-altitude dataset, see `src/data/pilot_altitude_loader.py`) and blood pressure as
  additional model features. Both are clinically relevant to AMS/HAPE/HACE and could improve
  separation between adjacent tiers.
- **Multi-user / expedition mode** — currently one `MedicalAlertPipeline` instance monitors one
  subject. A group expedition use case would need multiple concurrent subjects, a roster view,
  and alerts that identify *which* team member triggered them.
- **Cloud persistence + auth** — alert logs and history currently live in a local JSONL file /
  browser session. A real deployment would want a proper database, user accounts, and
  historical trend views across multiple expeditions/sessions.
- **Offline-first operation** — high-altitude expeditions often have no connectivity. A real
  hardware phase would need the ML classification and hysteresis gate to keep working fully
  offline, queuing Telegram alerts and LLM calls for whenever connectivity returns (both
  already degrade gracefully to a local fallback today — this would extend that to an explicit
  offline queue/retry mechanism).

---

## Project structure

```
high-altitude-medical-alert/
├── CLAUDE.md                    # full project context, decisions, and build plan
├── README.md                    # this file
├── Architecture_Diagrams/       # system architecture, phase tracker, data flow (HTML)
├── docs/lls_mapping.md          # clinical rationale + known limitations, written out
├── data/                        # gitignored — raw/synthetic/processed datasets
├── notebooks/                   # 10 teaching walkthroughs, one per src/ module
├── scripts/                     # one-off notebook-generation scripts (not runtime code)
├── src/
│   ├── config.py                # shared constants: severity tiers, thresholds, paths
│   ├── data/                     # loaders, synthetic generator, feature engineering
│   ├── models/                   # baseline/XGBoost/LSTM training + comparison + inference
│   ├── datasource/                # DataSource interface + scenario/manual/replay sources
│   ├── llm/                       # Groq interpretation, chat, guardrailed system prompt
│   ├── alerts/                    # hysteresis gate + Telegram dispatch
│   ├── app/                       # Streamlit dashboard
│   └── pipeline.py                 # wires everything into one live pipeline
└── tests/                        # 67 tests across every layer above
```
