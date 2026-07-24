import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import make_notebook

cells = [
    ("markdown", """# 07 — `src/models/metrics.py`: Scoring Every Model the Same Way

Small module, but an important one: it exists so the rule baseline, XGBoost, and LSTM are all scored with **literally the same code**. If each model script computed its own precision/recall independently, a subtle difference in averaging method (macro vs weighted, say) could make a "the XGBoost F1 is higher" comparison meaningless without anyone noticing — you'd be comparing numbers computed two different ways and calling it a fair fight.

This notebook explains *why* each metric was chosen, which matters more here than most ML projects because of the ordinal + imbalanced + safety-critical combination this system has."""),

    ("code", """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.metrics import evaluate, print_report, compare_models
from src.config import SEVERITY_TIERS"""),

    ("markdown", """## Why macro averaging, not micro/weighted

```python
precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
```

scikit-learn offers a few ways to average a per-class metric into one number:
- **micro**: pool all predictions together, compute one global score — dominated by whichever class has the most examples.
- **weighted**: average per-class scores, weighted by how common each class is — still lets the dominant class dominate the final number.
- **macro**: average per-class scores giving **every class equal weight**, regardless of how many examples it has.

Given `CLAUDE.md`'s explicit point that "Normal will dominate" the dataset, micro/weighted averaging would let a model that's great at Normal and terrible at HACE risk still post an impressive-looking overall score — because Normal is most of the data, it dominates the average. That's exactly backwards for a system whose entire point is catching the rare, dangerous cases. Macro averaging forces HACE risk's score to count just as much as Normal's, even though it's ~4% of the data."""),

    ("code", """# Concrete illustration: a model that's PERFECT on the dominant class and
# TERRIBLE on the rare class -- watch how differently each averaging method scores it.
from sklearn.metrics import f1_score

# 1000 "Normal" examples, model gets them all right.
# 40 "HACE risk" examples, model gets NONE right (always predicts Normal instead).
y_true = np.array([0]*1000 + [4]*40)
y_pred = np.array([0]*1000 + [0]*40)   # always predicts Normal

print("micro F1:   ", round(f1_score(y_true, y_pred, average='micro', labels=[0,4]), 3),
      " <- looks great! dominated by the 1000 correct 'Normal' predictions")
print("weighted F1:", round(f1_score(y_true, y_pred, average='weighted', labels=[0,4]), 3),
      " <- still looks great, same problem")
print("macro F1:   ", round(f1_score(y_true, y_pred, average='macro', labels=[0,4]), 3),
      " <- correctly exposes that this model NEVER catches HACE risk")"""),

    ("markdown", """That's the exact failure mode macro-averaging is chosen to expose: a model that's silently useless on the rare, dangerous class can hide behind a good micro/weighted score, but macro-averaging drags the overall number down to reflect that failure."""),

    ("markdown", """## Mean Absolute Tier Error (MATE) — the ordinal-specific metric

```python
mate = float(np.mean(np.abs(y_pred - y_true)))
```

None of precision/recall/F1 know anything about the tiers being *ordered*. To them, "predicted Normal (0), actual HACE risk (4)" and "predicted Severe AMS (2), actual HAPE risk (3)" are just two separate wrong classifications — same weight. But clinically, being off by 4 tiers is catastrophically worse than being off by 1.

MATE captures this directly: it's the average absolute difference between predicted and true tier index. A MATE of 0 is perfect. A MATE of 1 means predictions are typically one tier off. This is the metric `predict_severity.py` actually uses to pick the winning model, specifically because it's the one that reflects "how clinically costly are this model's mistakes on average," not just "how often is it exactly right.\""""),

    ("code", """# Two hypothetical models with the SAME accuracy but very different MATE
y_true_demo = np.array([0, 1, 2, 3, 4])

model_a_pred = np.array([1, 0, 1, 4, 3])   # always off by exactly 1 tier
model_b_pred = np.array([4, 4, 0, 0, 0])   # sometimes exactly right, sometimes wildly off

acc_a = np.mean(model_a_pred == y_true_demo)
acc_b = np.mean(model_b_pred == y_true_demo)
mate_a = np.mean(np.abs(model_a_pred - y_true_demo))
mate_b = np.mean(np.abs(model_b_pred - y_true_demo))

print(f"Model A: accuracy={acc_a:.2f}  MATE={mate_a:.2f}  (consistently close, never exactly right)")
print(f"Model B: accuracy={acc_b:.2f}  MATE={mate_b:.2f}  (right sometimes, catastrophically wrong other times)")
print()
print("Same 0% accuracy for model A here isn't the point -- the point is MATE tells")
print("them apart even when a plain 'exact match' metric like accuracy can't.")"""),

    ("markdown", """## Under-triage rate — the direction of error that actually matters

```python
under_triage_rate = float(np.mean(y_pred < y_true))
```

Not every wrong prediction is equally dangerous, and MATE alone doesn't capture *which direction* the error goes. Consider two mistakes with the identical |error| = 2:
- **Over-triage**: true tier is Normal (0), model predicts Severe AMS (2). Annoying (a false alarm, extra chat reassurance needed) but not dangerous.
- **Under-triage**: true tier is HAPE risk (3), model predicts Mild AMS (1). This is the failure mode that matters — the hysteresis gate and Telegram alert only fire based on the model's *predicted* severity, so an under-triaged HAPE case might never trigger an alert at all.

`under_triage_rate` is the fraction of predictions where the model said "less severe than reality" — tracked explicitly so it can't hide inside an aggregate number. This is exactly the failure mode the whole alert system exists to avoid, so it gets its own dedicated metric rather than being folded into MATE."""),

    ("code", """# Same |error|=2 in both directions -- MATE treats them identically, but clinically
# they are NOT equivalent. under_triage_rate is what tells them apart.
over_triage_case = dict(y_true=np.array([0]), y_pred=np.array([2]))   # predicted TOO SEVERE
under_triage_case = dict(y_true=np.array([3]), y_pred=np.array([1]))  # predicted TOO MILD

for name, case in [("Over-triage (false alarm)", over_triage_case), ("Under-triage (missed alert)", under_triage_case)]:
    mate = np.mean(np.abs(case["y_pred"] - case["y_true"]))
    under = np.mean(case["y_pred"] < case["y_true"])
    print(f"{name:32s}  MATE={mate:.1f}  under_triage_rate={under:.1f}")"""),

    ("markdown", """## `evaluate()`: putting it all together

```python
def evaluate(y_true, y_pred, model_name="model") -> dict:
    ...
    return {
        "model_name": model_name,
        "precision_macro": ..., "recall_macro": ..., "f1_macro": ...,
        "mean_abs_tier_error": mate,
        "under_triage_rate": under_triage_rate,
        "confusion_matrix": cm,
        "classification_report": report,
    }
```

One function, called identically by `rule_baseline.py`, `xgb_ordinal.py`, `lstm_model.py`, and `predict_severity.py`. It takes plain integer arrays (`y_true`/`y_pred` as severity indices 0-4) rather than string labels or model-specific output formats — which is what makes it usable regardless of whether the underlying model natively predicts strings (none do here), integers (rule baseline, LSTM via argmax), or a continuous score that's already been threshold-binned by the caller (XGBoost). Every model funnels down to the same plain integer-array contract before hitting this function."""),

    ("code", """# A worked example with fake predictions, showing the full dict this returns
demo_true = np.array([0,0,0,1,1,2,2,3,4,4])
demo_pred = np.array([0,0,1,1,2,2,2,2,3,4])

result = evaluate(demo_true, demo_pred, model_name="toy example")
print_report(result)"""),

    ("markdown", """## `compare_models()`: the side-by-side table

```python
def compare_models(results: list[dict]) -> pd.DataFrame:
    ...
```

Takes a list of `evaluate()` output dicts and produces one tidy comparison table — this is what actually gets printed when `predict_severity.py` runs the full baseline-vs-XGBoost-vs-LSTM comparison (see that notebook)."""),

    ("code", """# Simulate two more toy models to show the comparison table shape
result_b = evaluate(demo_true, np.array([0,0,0,0,1,1,2,3,3,4]), model_name="toy model B")
result_c = evaluate(demo_true, demo_true, model_name="toy perfect model")  # for contrast

comparison = compare_models([result, result_b, result_c])
comparison"""),

    ("markdown", """## Summary

Three deliberate choices in this module, each addressing a specific risk in this project:

1. **Macro averaging** — so the dominant "Normal" class can't hide poor performance on rare, dangerous tiers.
2. **Mean Absolute Tier Error** — so being "close but wrong" is rewarded over being wildly wrong, which plain accuracy/F1 can't express for ordinal data.
3. **Under-triage rate** — so predicting "too mild" (the dangerous direction, the one that could suppress a real alert) is tracked explicitly rather than blended into an aggregate.

Every model comparison you see elsewhere in this project traces back to these three numbers, computed the same way every time.

**Next notebook:** `08_xgb_ordinal.ipynb` — the primary model, and how it turns regression + tuned thresholds into an ordinal-aware classifier."""),
]

make_notebook(cells, "c:/Users/shaur/Desktop/My codes shaurya/Medical Alert System/notebooks/07_metrics_walkthrough.ipynb")
