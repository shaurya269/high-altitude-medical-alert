"""
Central configuration for the High-Altitude Medical Alert System.

Keeping severity tiers, clinical thresholds, and paths in one module means
every stage of the pipeline (labeling, features, models, hysteresis gate,
LLM prompts, UI) reads from the same source of truth instead of each file
re-declaring its own copy of "what counts as Severe AMS" and slowly drifting
apart. If you retune a threshold, change it here once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Path(__file__).resolve() -> .../high-altitude-medical-alert/src/config.py
# .parent twice -> project root. Using this instead of a hardcoded absolute
# path means the project still works if the folder gets moved or renamed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HARESPOD_DIR = RAW_DIR / "harespod"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Severity tiers (the ordinal target)
# ---------------------------------------------------------------------------
# This ordering IS the point of "ordinal" classification: index 0 is
# healthiest, index 4 is most dangerous, and the distance between indices
# is meaningful (mistaking tier 4 for tier 0 is a much worse error than
# mistaking tier 4 for tier 3). Plain multi-class classification treats all
# wrong answers as equally wrong, which throws this information away.
SEVERITY_TIERS = [
    "Normal",
    "Mild AMS",
    "Severe AMS",
    "HAPE risk",
    "HACE risk",
]
SEVERITY_INDEX = {name: i for i, name in enumerate(SEVERITY_TIERS)}
N_TIERS = len(SEVERITY_TIERS)

# ---------------------------------------------------------------------------
# Clinical reference ranges (used by the rule-based baseline AND to sanity
# check / bound the synthetic data generator so it produces physiologically
# plausible trajectories rather than arbitrary numbers).
#
# Sources are general high-altitude medicine literature (Lake Louise Score
# criteria + expected SpO2-vs-altitude relationships), not a single paper --
# treat these as reasonable engineering defaults for a learning project, NOT
# clinically validated constants. See CLAUDE.md Section 4.
# ---------------------------------------------------------------------------

# Expected SpO2 (%) at sea level for a healthy adult at rest.
SPO2_SEA_LEVEL_BASELINE = 98.0

# Rough expected SpO2 drop per 1000m of altitude gain (rule-of-thumb used to
# compute "expected_spo2_at_altitude" so we can flag readings that are LOWER
# than expected-for-altitude, not just low in absolute terms -- 88% SpO2 is
# very different at sea level vs at 5000m).
SPO2_DROP_PER_1000M = 3.0

# Normal resting heart rate range (bpm) -- used by the rule baseline as a
# sanity band; individual variation is real, so this is intentionally wide.
HR_NORMAL_RANGE = (50, 100)

# Normal core/contactless-forehead body temp range (Celsius). HAPE/HACE can
# present with mild fever-like elevation in some literature-derived patterns.
TEMP_NORMAL_RANGE = (36.1, 37.2)

# Altitude bands (meters) roughly matching where AMS/HAPE/HACE risk rises in
# the literature. Not a hard cutoff -- just informs synthetic generation and
# the rule baseline's "at risk altitude" flag.
ALTITUDE_RISK_ONSET_M = 2500  # AMS risk becomes clinically relevant above this

# ---------------------------------------------------------------------------
# Feature engineering windows
# ---------------------------------------------------------------------------
ROLLING_BUFFER_MINUTES = 10   # Stage 2 in the data flow diagram
TREND_WINDOW_MINUTES = 5      # spo2_trend_5min / hr_trend_5min
SAMPLE_RATE_HZ = 1            # matches a realistic Arduino polling rate (see CLAUDE.md)
LSTM_WINDOW_SECONDS = 60      # comparison model's rolling window size

# ---------------------------------------------------------------------------
# Hysteresis gate (Stage 5) -- prevents a single noisy reading from firing
# a Telegram alert. Tune these here, not inline in alerts/hysteresis logic.
#
# KNOWN LIMITATION (Day 13 finding, two fix attempts tried and both
# reverted -- see docs/lls_mapping.md "Under-triage vs. false-alarm
# tradeoff" for the full writeup): 45% of true HAPE-risk/HACE-risk
# readings get an XGBoost prediction BELOW Severe AMS on the held-out
# synthetic test set, meaning the hysteresis gate never even attempts an
# alert for nearly half the most dangerous cases.
#
#   Attempt 1 -- bias xgb_ordinal.py's threshold tuning to penalize
#   under-triage more than over-triage. Reduced the 45% figure (to 26.8%
#   at penalty weight 3.0), but reliably caused a genuinely Normal demo
#   scenario to trigger a real false alert (10/10 random seeds). Reverted.
#
#   Attempt 2 -- lower HYSTERESIS_ALERT_TIER to Mild AMS instead (this
#   constant), leaving the model's own thresholds untouched. Failed for a
#   related reason: 38% of true-Normal readings already predict >= Mild
#   AMS at the single-reading level (this model was never tuned to
#   separate Normal from Mild AMS as tightly as Normal from Severe AMS),
#   which is high enough to reliably sustain the 3-consecutive-reading
#   streak too. Reverted back to Severe AMS.
#
# Both attempts hit the same underlying wall: this specific model's
# calibration doesn't cleanly support catching more true danger without
# also catching more false alarms, at ANY tier boundary tried so far.
# Fixing this properly would need either a differently-trained model
# (e.g. one explicitly optimized to separate Normal from everything else
# more sharply) or a second, independent signal beyond the single ML
# classification (e.g. requiring the LLM's interpretation to agree, or a
# longer consecutive-reading requirement specifically for the Mild-AMS
# gate) -- both are real future work, not implemented here.
HYSTERESIS_ALERT_TIER = SEVERITY_INDEX["Severe AMS"]  # alert-eligible at/above this tier
HYSTERESIS_CONSECUTIVE_READINGS = 3    # must sustain for N readings in a row
HYSTERESIS_COOLDOWN_SECONDS = 15 * 60  # minimum gap between repeat alerts

# ---------------------------------------------------------------------------
# Train/val/test temporal split (CLAUDE.md: never random-shuffle time series)
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15  # kept explicit (not derived) so the three always sum to 1.0 visibly

RANDOM_SEED = 42
