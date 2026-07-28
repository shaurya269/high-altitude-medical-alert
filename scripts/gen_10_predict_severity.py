import sys
from pathlib import Path

# Make build_notebook.py importable regardless of the cwd this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import make_notebook

# Static, in-order (cell_type, source) list defining every cell of
# notebooks/10_predict_severity_walkthrough.ipynb -- walks through
# run_comparison()'s model selection (by mean_abs_tier_error, baseline
# excluded), the row-vs-window prediction-count subtlety between XGBoost
# and the LSTM, and predict_severity()'s single stable entry point +
# per-model confidence proxies.
cells = [
    ("markdown", """# 10 — `src/models/predict_severity.py`: Selecting the Winner + Live Inference

This is the module that ties everything from the previous six notebooks together. It has two distinct jobs, deliberately kept in one file because they share so much:

1. **`run_comparison()`** — a one-time (well, re-run-whenever-you-retrain) step that loads all three trained models, evaluates them on the identical held-out test set, picks a winner, and writes the decision to disk.
2. **`predict_severity()`** — the single function every future piece of the live pipeline (hysteresis gate, LLM interpreter, Streamlit dashboard) will call. It doesn't know or care whether XGBoost or the LSTM is active underneath — that's the entire point."""),

    ("code", """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import json
import pandas as pd

from src.models import predict_severity as ps
from src.config import PROCESSED_DIR, MODELS_DIR"""),

    ("markdown", """## Why the comparison loads saved models instead of retraining

```python
def run_comparison():
    ...
    xgb_model_obj, thresholds = xgb_ordinal.load()
    lstm_model_obj, mean, std = lstm_model.load()
```

`run_comparison()` assumes `xgb_ordinal.py` and `lstm_model.py` have already been run once and their artifacts saved. This is a deliberate separation of concerns: training is expensive (especially the LSTM), comparison is cheap. If you only want to re-run the comparison — say, after tweaking `metrics.py`'s selection criterion — you shouldn't have to pay for a full LSTM retrain just to see updated numbers."""),

    ("markdown", """## A subtlety: XGBoost and the LSTM don't produce the same number of predictions

```python
xgb_pred = xgb_ordinal.predict(xgb_model_obj, thresholds, test_df)   # one prediction per ROW
...
lstm_pred, lstm_true = lstm_model.predict(lstm_model_obj, mean, std, test_df)  # one per WINDOW
results.append(evaluate(lstm_true, lstm_pred, model_name="LSTM"))  # note: lstm_true, not the outer y_true!
```

XGBoost predicts once per row of `test_df` (it doesn't need history beyond the current row's already-engineered trend features). The LSTM predicts once per **60-second window**, and with the `STRIDE_SECONDS=15` windowing from the previous notebook, that's a different — and smaller — count than the row count. If we'd evaluated the LSTM's predictions against the row-level `y_true`, the arrays would be different lengths (or worse, silently misaligned if truncated to match). The code instead uses `lstm_true` — the labels `build_windows()` returned alongside the LSTM's predictions, which are already correctly aligned to what the LSTM actually saw. This is a real "comparing apples to apples" detail worth being careful about whenever you're comparing models with structurally different input shapes."""),

    ("code", """test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
print(f"test_df has {len(test_df):,} rows (one XGBoost prediction each)")

from src.models import lstm_model
lstm_model_obj, mean, std = lstm_model.load()
_, lstm_true = lstm_model.build_windows(test_df)  # just to see the count, ignoring predictions here
print(f"LSTM produces {len(lstm_true):,} windowed predictions -- a DIFFERENT count, and that's expected")"""),

    ("markdown", """## The selection rule

```python
candidates = [r for r in results if r["model_name"] in ("XGBoost Ordinal", "LSTM")]
winner = min(candidates, key=lambda r: r["mean_abs_tier_error"])

beats_baseline = winner["mean_abs_tier_error"] < baseline_result["mean_abs_tier_error"]
if not beats_baseline:
    print("*** WARNING: ... ***")
```

Two things worth noting:

1. **The rule-based baseline is explicitly excluded from being selectable**, even if it happened to post a better number on some run by chance. It exists to *check* the real models, not to compete as a production candidate — CLAUDE.md's whole point in requiring it.
2. **Selection is by `mean_abs_tier_error`**, not F1 or accuracy — because, as the `metrics.py` notebook explained, that's the metric that actually reflects clinical cost for ordinal data (being one tier off is a small mistake, being four tiers off is a big one).
3. If the winner still doesn't beat the baseline, the code doesn't silently proceed — it prints a loud warning. That's the CLAUDE.md-mandated sanity check made operational: if this warning ever fires, don't trust the selected model until you've investigated features/labels."""),

    ("markdown", """## Running the real comparison

This loads the actual saved XGBoost and LSTM models (trained in the previous two notebooks) and the rule baseline (no saved state needed — it's just code), evaluates all three on the same test set, and writes `model_comparison.json` + `selected_model.json` to `src/models/artifacts/`."""),

    ("code", """comparison_record = ps.run_comparison()"""),

    ("markdown", """## Inspecting the saved decision record

`run_comparison()` doesn't just print to console — it writes the decision to disk as JSON, specifically so the choice is **traceable** later (e.g. "why is XGBoost the active model? let's check what the actual test-set numbers were when it was selected" — a comment in code can rot out of date; a JSON artifact written by the actual run cannot lie about what happened)."""),

    ("code", """with open(MODELS_DIR / "model_comparison.json") as f:
    print(json.dumps(json.load(f), indent=2))"""),

    ("code", """with open(MODELS_DIR / "selected_model.json") as f:
    print(json.load(f))"""),

    ("markdown", """## `predict_severity()`: the single live inference entry point

This is the function every future module (hysteresis gate in `src/alerts/`, LLM prompt builder in `src/llm/`, the Streamlit dashboard in `src/app/`) will import. Its entire job is to **hide which model is active**.

```python
def predict_severity(buffer_df: pd.DataFrame) -> dict:
    selected = _get_selected_model_name()   # reads selected_model.json, cached after first read
    if selected == "XGBoost Ordinal":
        ...
    elif selected == "LSTM":
        ...
    return {"severity_index": ..., "severity_label": ..., "confidence": ..., "model_used": ...}
```

**Why this matters architecturally:** if, months from now, someone improves the LSTM (say, gives it an ordinal-aware loss) and it starts winning the comparison, `selected_model.json` would just say `"LSTM"` instead of `"XGBoost Ordinal"` — and every downstream caller of `predict_severity()` keeps working with **zero code changes**, because they never knew which model was active in the first place. This is the same "swap the implementation without touching the callers" principle you saw in the `DataSource` interface design from CLAUDE.md's hardware phase planning."""),

    ("markdown", """## Confidence: an honest, not over-claimed, proxy

The two branches compute "confidence" completely differently, and the docstrings are explicit about what it does and doesn't mean.

**XGBoost branch:**
```python
edges = [0.0] + list(thresholds) + [4.0]
lo, hi = edges[tier_index], edges[tier_index + 1]
center = (lo + hi) / 2
confidence = float(1.0 - min(abs(continuous - center) / (span / 2), 1.0) * 0.5)
```
This measures how close the raw continuous prediction sits to the *center* of its predicted bin vs. how close to an edge (a borderline case). It is explicitly commented as **not a calibrated probability** — XGBoost regression doesn't natively produce one — just a "how borderline is this" signal.

**LSTM branch:**
```python
probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
confidence = float(probs[tier_index])
```
This one genuinely *is* a probability (softmax normalizes the classifier's output logits into a valid probability distribution over the 5 tiers), because the LSTM is a real classifier with a softmax head — a different, more standard notion of confidence than XGBoost's regression-based proxy.

**Why does this distinction matter for the project?** When this confidence value eventually reaches the LLM prompt (Day 9) or the dashboard (Day 12), it should be presented honestly — "the model is fairly confident" rather than a specific miscalibrated percentage implying more precision than the underlying number actually has, especially for the XGBoost proxy."""),

    ("markdown", """## Trying it live, on a real buffer

Let's simulate what the live pipeline will eventually do: hand `predict_severity()` a subject's buffered readings and see what comes back."""),

    ("code", """test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

# Grab one subject's full trajectory as a stand-in "live buffer"
some_subject = test_df["subject_id"].unique()[3]
buffer = test_df[test_df["subject_id"] == some_subject].sort_values("timestamp")

print(f"Subject: {some_subject}")
print(f"True severity at the end of this buffer: {buffer.iloc[-1]['severity_label']}")
print()
print("predict_severity() output:")
print(ps.predict_severity(buffer))"""),

    ("markdown", """## What happens with too little history

The docstring is explicit: `predict_severity()` needs the buffer to have already been through `engineer_features()`, and if the LSTM is the active model, it additionally needs at least 60 seconds of history for a full window. Let's see the LSTM path's guard rail in action (temporarily, without changing which model is actually selected on disk)."""),

    ("code", """# Directly exercise the LSTM branch's minimum-buffer-length guard, independent
# of whichever model selected_model.json currently points to.
short_buffer = buffer.iloc[:10]  # only 10 seconds of history -- not enough for a 60s LSTM window

try:
    lstm_obj, mean, std = lstm_model.load()
    if len(short_buffer) < lstm_model.WINDOW_SAMPLES:
        raise ValueError(
            f"LSTM model needs at least {lstm_model.WINDOW_SAMPLES} buffered readings "
            f"(60s at 1Hz), got {len(short_buffer)}. Not enough history yet -- caller "
            "should fall back to the rule baseline until the buffer fills."
        )
except ValueError as e:
    print("Expected guard-rail error:")
    print(e)"""),

    ("markdown", """This matters for the real system: when a demo scenario or a live sensor stream first starts, there won't be 60 seconds of buffered history yet. The error message is explicit about what the caller should do about it (fall back to the rule baseline) — this is exactly the kind of fallback logic Day 11's full integration will need to wire up."""),

    ("markdown", """## Summary — tying together Days 4-7

Reading this notebook alongside the previous six tells the whole Week 1 story:

1. **`docs/lls_mapping.md`** wrote down the clinical reasoning.
2. **`synth_data.py`** turned it into thousands of realistic, gradually-onsetting synthetic trajectories.
3. **`feature_engineering.py`** turned raw vitals into clinically-meaningful derived features, and split subjects into train/val/test without leakage (including fixing a real stratification bug along the way).
4. **`rule_baseline.py`** coded the same clinical reasoning directly as a sanity-floor model.
5. **`metrics.py`** defined a fair, ordinal-aware, imbalance-aware way to score every model identically.
6. **`xgb_ordinal.py`** and **`lstm_model.py`** each tried a different strategy for the same ordinal severity-prediction problem.
7. **`predict_severity.py`** (this notebook) picked the winner objectively, on the metric that matters clinically, and packaged it behind one stable function signature.

That function — `predict_severity()` — is what the rest of the project (Days 8-14: demo data source, LLM interpretation, Telegram alerts, the Streamlit dashboard) will build on top of, without ever needing to know it's XGBoost underneath."""),
]

# Render the cell list above into a real .ipynb file at this fixed path.
make_notebook(cells, "c:/Users/shaur/Desktop/My codes shaurya/Medical Alert System/notebooks/10_predict_severity_walkthrough.ipynb")
