# Lake Louise Score (LLS) → Sensor-Pattern Severity Mapping

This document is the **rule book** that `src/data/synth_data.py` codes into
synthetic trajectories, and that `src/models/rule_baseline.py` codes into
if/else thresholds. It exists so both are *traceable back to a single written
rationale* instead of scattering "why 85%?" decisions across code comments.

**Important limitation (also in CLAUDE.md Section 4 and the README):** the
Lake Louise Score is a *symptom self-report questionnaire* (headache, GI
symptoms, fatigue, dizziness, sleep disturbance) — it is not itself a sensor
reading. There is no dataset that pairs raw SpO2/HR/temp/altitude sensor
streams with real clinician-assigned LLS scores at the scale we need. So the
mapping below is a **literature-informed engineering approximation**: for
each severity tier, what sensor pattern would plausibly co-occur with that
LLS-implied clinical picture, based on general high-altitude physiology
literature. It is *not* a validated clinical instrument, and the system must
never be presented as diagnosing LLS or any clinical condition — see
CLAUDE.md Section 7 (LLM guardrails) and the README's prototype disclaimer.

## Tier definitions used by this project

| Tier | LLS-equivalent picture | Sensor pattern (approximate, literature-informed) |
|---|---|---|
| **Normal** | LLS 0-2, no meaningful symptoms | SpO2 within ~3% of altitude-expected baseline; HR at/near individual resting range; temp normal; no adverse trend |
| **Mild AMS** | LLS 3-5, headache + 1+ mild symptom | SpO2 modestly below altitude-expected (~4-8% delta); HR mildly elevated (+10-20 bpm over resting); temp normal-to-slightly-elevated; symptoms plateau, don't worsen over the window |
| **Severe AMS** | LLS 6+, headache + multiple/severe symptoms, functional impairment | SpO2 clearly below altitude-expected (~8-14% delta); HR persistently elevated (+20-35 bpm); mild temp elevation more common; negative trend over the trailing window (worsening, not just low) |
| **HAPE risk** | Respiratory distress signs (Lake Louise "HAPE" criteria: dyspnea at rest, cough, chest tightness) layered on AMS | SpO2 sharply below altitude-expected (>14% delta) AND/OR a steep negative SpO2 trend over 5 min even if the absolute value hasn't bottomed out yet (early HAPE can desaturate fast); HR strongly elevated; the *trend* matters as much as the absolute value here |
| **HACE risk** | Neurological signs (Lake Louise "HACE" criteria: ataxia, altered consciousness) layered on AMS | Severe SpO2 depression similar to or worse than HAPE risk, but the defining synthetic signal is a *sustained, non-recovering* trajectory (no plateau/recovery even as other interventions would normally help) combined with the highest HR elevation band -- modeling the idea that HACE is a late, decompensating state |

## Why thresholds are expressed as deltas-from-expected, not raw SpO2

A raw SpO2 of 88% means something very different at sea level (alarming)
than at 5,500m (potentially unremarkable for an acclimatized person). That's
why `src/config.py` defines `SPO2_DROP_PER_1000M` and feature engineering
computes an `expected_spo2_at_altitude` delta — the ML model and the rule
baseline both reason about *deviation from what's expected for that
altitude*, not an absolute number. This mirrors how a clinician would
actually reason about the reading.

## How this maps into code

- `src/data/synth_data.py`: `simulate_trajectory()` generates gradual-onset
  (not step-function) trajectories per tier, sampling actual values from
  ranges implied by the table above, plus per-subject individual variation
  (baseline resting HR, personal altitude sensitivity) so the model doesn't
  overfit to one "template" patient.
- `src/models/rule_baseline.py`: encodes the same deltas/thresholds as
  explicit if/else rules — the sanity-check baseline any ML model must beat.
- `src/data/label_real_data.py`: applies the SAME `rule_baseline.classify_row()`
  thresholds to real, unlabeled sensor data (Harespod) post-hoc, since
  neither real dataset available to this project includes a severity
  assessment. This is a deliberate reuse, not a re-derivation — there is
  exactly one place in the codebase encoding "what spo2_delta/hr_elevation/
  trend combination means Severe AMS." See `feature_engineering.py`'s
  `run_pipeline()` docstring for why rule-labeled real data is merged into
  the TRAINING split only, never validation/test (using it in
  val/test would let the rule-based baseline trivially "predict" labels
  it generated itself).

## A real-data caveat this mapping does NOT resolve: oxygen-enriched exposure

This entire mapping assumes **ambient-air hypoxia** — SpO2 drops because
the air itself has less oxygen at altitude (`config.SPO2_DROP_PER_1000M`).
While integrating real datasets, a second candidate (the "High-Altitude
Pilot Physiological Monitoring Dataset," `src/data/pilot_altitude_loader.py`)
turned out to use an **oxygen-enriched** chamber protocol (~40-45% O2
throughout, confirmed via its own O2/N2/CO2 columns) — a fundamentally
different physiological scenario where SpO2 stays high even at extreme
altitude because supplemental oxygen compensates for the reduced pressure.
Applying this document's ambient-air-derived thresholds to that dataset
would actively mislabel it (a healthy 98% SpO2 reading at 7500m would be
flagged as severely abnormal, since ambient air at 7500m would expect
~75%). That dataset is loaded but deliberately excluded from severity
labeling for this reason — see `CLAUDE.md` Section 5's Decision Log.
Harespod, by contrast, is confirmed ambient-air (per its own paper's
Methods section) and is the dataset this mapping is actually applied to.

## Known limitation: under-triage vs. false-alarm tradeoff (Day 13 finding)

Testing found that on the held-out synthetic test set, **45% of true
HAPE-risk/HACE-risk readings get an XGBoost prediction below the
hysteresis gate's alert-eligible tier (Severe AMS)** — meaning the
Telegram alert (`src/alerts/hysteresis.py` + `alert_bot.py`) is never even
*attempted* for nearly half the most dangerous cases, regardless of how
well the consecutive-readings/cooldown logic itself behaves. This is a
genuine gap in the current model's calibration, not a hysteresis bug —
the gate is working exactly as designed, but the severity classification
feeding into it is what's under-confident on the rarest, most dangerous
tiers.

**Two fixes were tried and both reverted** after concrete testing showed
they traded this problem for a worse one — a real Telegram alert firing
on a genuinely Normal demo scenario:

1. **Bias `xgb_ordinal.py`'s threshold tuning to penalize under-triage
   more than over-triage** (`UNDER_TRIAGE_PENALTY_WEIGHT` in that file).
   At weight 3.0, this reduced the 45% figure to 26.8% — but caused 10/10
   random-seed Normal scenarios to fire a false alert, because biasing the
   *same* thresholds that separate Normal from Severe AMS pushes ordinary
   sensor noise across that line often enough to satisfy the
   3-consecutive-reading requirement. A gentler weight (1.5) reduced the
   45% figure to 33.3% and still reliably caused false alerts (confirmed
   across the first several random seeds before stopping). Reverted to
   weight 1.0 (plain, symmetric mean-absolute-tier-error), matching the
   original Days 4-7 model exactly.

2. **Lower `config.HYSTERESIS_ALERT_TIER` from Severe AMS to Mild AMS**
   instead, leaving the model's own classification thresholds untouched.
   This improved HAPE/HACE coverage substantially (only 9.7% now fall
   below the new, lower gate) — but failed for a related reason: 38% of
   true-*Normal* readings already predict at/above Mild AMS at the
   single-reading level (this model was never trained/tuned to separate
   Normal from Mild AMS as tightly as it separates Normal from Severe
   AMS), which is high enough to reliably sustain a 3-in-a-row streak too.
   Confirmed via a live scenario test before reverting back to Severe AMS.

**Both attempts hit the same underlying wall:** this specific XGBoost
model's calibration doesn't currently support catching more true danger
without also catching more false alarms, at any tier boundary tried so
far. The system currently ships with the **original, well-tested
configuration** (Severe AMS gate, symmetric threshold tuning) — meaning
it is honestly biased toward fewer false alarms at the cost of missing
some true elevated cases, which is a defensible but not ideal tradeoff
point for a medical alert system to sit at by default.

**Possible real fixes, not implemented here:**
- Train a model (or a dedicated binary classifier) specifically optimized
  to separate "Normal" from "everything else" as sharply as possible,
  rather than relying on one multi-tier ordinal regression to do that
  job well at every boundary simultaneously.
- Require a second, independent signal to agree before alerting (e.g. the
  LLM's own plain-language read of the vitals, or a longer/adaptive
  consecutive-reading requirement specifically for lower-severity gates)
  rather than gating on the ML classification alone.
- Collect more real (non-synthetic) training data, particularly at the
  Normal/Mild-AMS boundary specifically, since that's where both fix
  attempts' false alarms concentrated.

## Sources (general references, not a single clinical trial)

General high-altitude medicine literature on Acute Mountain Sickness, HAPE,
and HACE (Lake Louise Consensus criteria for AMS scoring; standard
physiology references on SpO2-vs-altitude relationships). This project does
not cite a single peer-reviewed source for the exact numeric thresholds
because none exists at this sensor-pattern granularity — that's the
limitation this document exists to make explicit and traceable.
