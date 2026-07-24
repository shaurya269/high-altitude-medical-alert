import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import make_notebook

cells = [
    ("markdown", """# 02 — `src/config.py`: The Single Source of Truth

Every other module in this project (`synth_data.py`, `feature_engineering.py`, all three models, and later the hysteresis gate + LLM prompts) imports constants from here instead of re-declaring its own copy of "what counts as Severe AMS" or "how many minutes is the trend window."

**Why centralize this at all?** Imagine `synth_data.py` hardcoded `SEVERITY_TIERS = [...]` and `rule_baseline.py` had its own separate copy. Six months from now you decide to rename "HAPE risk" to "HAPE onset" — you'd have to remember every file that has its own copy, and if you miss one, the two lists silently drift apart and comparisons between them break in confusing ways. With one shared `config.py`, you change it in exactly one place.

This notebook imports the real module and walks through each section."""),

    ("code", """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # so `import src...` works from notebooks/

from src import config"""),

    ("markdown", """## Paths

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

`__file__` is the path to `config.py` itself. `.resolve()` turns it into an absolute path. `.parent` once gets you to `src/`, `.parent` again gets you to the project root. Everything else (`data/`, `src/models/artifacts/`) is defined *relative to* `PROJECT_ROOT`.

**Why not just hardcode `C:/Users/shaur/Desktop/.../Medical Alert System`?** Because that only works on your machine, in this exact folder. If you ever move the project, rename the folder, or someone else clones the repo to a different path, a hardcoded path breaks immediately. Deriving it from `__file__` means the code always finds its own project root no matter where it's sitting."""),

    ("code", """print("PROJECT_ROOT:  ", config.PROJECT_ROOT)
print("DATA_DIR:      ", config.DATA_DIR)
print("HARESPOD_DIR:  ", config.HARESPOD_DIR)
print("SYNTHETIC_DIR: ", config.SYNTHETIC_DIR)
print("PROCESSED_DIR: ", config.PROCESSED_DIR)
print("MODELS_DIR:    ", config.MODELS_DIR)"""),

    ("markdown", """## Severity tiers — the ordinal target

```python
SEVERITY_TIERS = ["Normal", "Mild AMS", "Severe AMS", "HAPE risk", "HACE risk"]
```

This list is ordered on purpose, and that order is the single most important design decision in the whole ML pipeline.

**Ordinal vs. plain multi-class — why it matters:** if you told a model "these are 5 unrelated categories" (plain multi-class classification), predicting "Normal" when the truth was "HACE risk" (a life-threatening miss) would be scored *exactly as wrong* as predicting "Mild AMS" when the truth was "Severe AMS" (a much smaller, less dangerous miss). Both are just "wrong" to a plain classifier's loss function. But clinically those two mistakes are nowhere near equally bad.

Because `SEVERITY_TIERS` is an ordered list, we can index it (0=Normal, 1=Mild AMS, ... 4=HACE risk) and use that number directly in a loss function — see the `xgb_ordinal.py` notebook, where XGBoost is trained to *regress* this index as a continuous number, which is what actually teaches the model that tier 4 is "far" from tier 0 but "close" to tier 3."""),

    ("code", """print("Tiers:", config.SEVERITY_TIERS)
print("Index lookup:", config.SEVERITY_INDEX)
print("N_TIERS:", config.N_TIERS)

# This is exactly what lets us treat severity as a number, not just a label:
for tier, idx in config.SEVERITY_INDEX.items():
    print(f"  {idx} -> {tier}")"""),

    ("markdown", """## Clinical reference ranges

These come from general high-altitude medicine literature (Lake Louise Score criteria, typical SpO2-vs-altitude relationships) — **not** one specific clinical trial. The comments in `config.py` are explicit about this being an engineering approximation for a learning project, not a validated medical instrument (see `docs/lls_mapping.md` for the full reasoning and `CLAUDE.md` Section 4).

| Constant | Value | What it's for |
|---|---|---|
| `SPO2_SEA_LEVEL_BASELINE` | 98.0% | Reference point: what a healthy person's SpO2 looks like at sea level |
| `SPO2_DROP_PER_1000M` | 3.0 | Rule-of-thumb: SpO2 drops ~3% for every 1000m you climb |
| `HR_NORMAL_RANGE` | (50, 100) bpm | Resting heart rate band used as a sanity check |
| `TEMP_NORMAL_RANGE` | (36.1, 37.2)°C | Normal contactless-forehead temp band |
| `ALTITUDE_RISK_ONSET_M` | 2500m | Altitude above which AMS risk becomes clinically relevant |

**Why deltas, not raw numbers?** An SpO2 reading of 88% means something very different at sea level (alarming — call a doctor) than at 5000m altitude (potentially unremarkable for an acclimatized climber). `SPO2_DROP_PER_1000M` exists so the rest of the pipeline can compute "how far is this reading from what's *expected* at this altitude" instead of judging raw numbers in a vacuum. You'll see this used directly in `synth_data.expected_spo2()`."""),

    ("code", """# expected_spo2 lives in synth_data.py but USES these config constants --
# quick demo of the relationship:
from src.data.synth_data import expected_spo2

for altitude in [0, 2000, 3500, 5000, 6000]:
    print(f"altitude={altitude:>5}m  ->  expected SpO2 = {expected_spo2(altitude):.1f}%")"""),

    ("markdown", """## Feature engineering windows

```python
ROLLING_BUFFER_MINUTES = 10   # how much history we keep in the live buffer
TREND_WINDOW_MINUTES   = 5    # how far back we look to compute a "trend"
SAMPLE_RATE_HZ         = 1    # one reading per second
LSTM_WINDOW_SECONDS    = 60   # the LSTM's fixed input window length
```

`SAMPLE_RATE_HZ = 1` is worth pausing on: Harespod's raw data is 100Hz (100 readings/second), but a real Arduino polling I2C sensors realistically can't sustain that — 1Hz (once per second) is what CLAUDE.md specifies as the realistic target rate. Every module that touches timing (feature windows, LSTM sequence length) is written in terms of `SAMPLE_RATE_HZ` rather than a hardcoded "60" or "300", so if this assumption ever changes, one edit here propagates everywhere."""),

    ("code", """print("Rolling buffer:", config.ROLLING_BUFFER_MINUTES, "minutes")
print("Trend window:  ", config.TREND_WINDOW_MINUTES, "minutes")
print("Sample rate:   ", config.SAMPLE_RATE_HZ, "Hz")
print("LSTM window:   ", config.LSTM_WINDOW_SECONDS, "seconds",
      f"= {config.LSTM_WINDOW_SECONDS * config.SAMPLE_RATE_HZ} samples at {config.SAMPLE_RATE_HZ}Hz")"""),

    ("markdown", """## Hysteresis gate settings

This is Stage 5 of the data flow diagram — the logic that decides whether an elevated severity reading actually fires a Telegram alert.

```python
HYSTERESIS_ALERT_TIER = SEVERITY_INDEX["Severe AMS"]  # = 2
HYSTERESIS_CONSECUTIVE_READINGS = 3
HYSTERESIS_COOLDOWN_SECONDS = 15 * 60
```

**Why is this needed at all?** Sensor readings are noisy. A single bad reading — a loose contact, a motion artifact on the MAX30102, a brief SpO2 dip while someone coughs — could spuriously classify as "Severe AMS" for one second and then go back to Normal. If we alerted on every single reading that crosses the threshold, the medical contact would get spammed with false alarms and (just as bad) start ignoring real ones. So the gate requires the severity to be `>= HYSTERESIS_ALERT_TIER` for `HYSTERESIS_CONSECUTIVE_READINGS` readings *in a row* — 3 consecutive seconds of sustained severity, not a single spike — AND that at least `HYSTERESIS_COOLDOWN_SECONDS` (15 minutes) has passed since the last alert, so it doesn't re-fire every second while the situation persists. This logic isn't built yet (it's Day 11 in the plan) but the numbers already live here."""),

    ("code", """print("Alert-eligible at tier >=", config.HYSTERESIS_ALERT_TIER,
      f"({config.SEVERITY_TIERS[config.HYSTERESIS_ALERT_TIER]})")
print("Must sustain for", config.HYSTERESIS_CONSECUTIVE_READINGS, "consecutive readings")
print("Cooldown between alerts:", config.HYSTERESIS_COOLDOWN_SECONDS, "seconds",
      f"= {config.HYSTERESIS_COOLDOWN_SECONDS / 60:.0f} minutes")"""),

    ("markdown", """## Train/val/test split fractions

```python
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15
RANDOM_SEED    = 42
```

Nothing exotic here, but two things are worth explaining:

1. **All three fractions are written out explicitly and must sum to 1.0** (there's an `assert` for this in `feature_engineering.py`) rather than computing `TEST_FRACTION = 1 - TRAIN_FRACTION - VAL_FRACTION`. This is a readability choice — you can see all three numbers at a glance and confirm they're sane, instead of having to do mental math on one of them.
2. **`RANDOM_SEED = 42`** is used everywhere randomness happens (synthetic data generation, XGBoost training, LSTM weight initialization) so that re-running the pipeline produces the *same* dataset and *same* trained models every time. This matters for debugging — if a bug appears, you want to be able to reproduce it exactly, not have it come and go with a different random seed each run."""),

    ("code", """print("Train / Val / Test:", config.TRAIN_FRACTION, "/", config.VAL_FRACTION, "/", config.TEST_FRACTION)
print("Sum:", config.TRAIN_FRACTION + config.VAL_FRACTION + config.TEST_FRACTION)
print("Random seed:", config.RANDOM_SEED)"""),

    ("markdown", """## Summary

`config.py` doesn't *do* anything by itself — it has no functions, just constants and derived paths. Its entire value is that every other file in this project imports from it instead of inventing its own numbers. When you read the other notebooks and see `from src.config import SEVERITY_TIERS` or `from src.config import HR_NORMAL_RANGE`, that's this file at work.

**Next notebook:** `03_synth_data.ipynb` — how the synthetic vitals dataset is actually generated, tier by tier."""),
]

make_notebook(cells, "c:/Users/shaur/Desktop/My codes shaurya/Medical Alert System/notebooks/02_config_walkthrough.ipynb")
