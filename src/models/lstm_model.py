"""
LSTM severity classifier -- the deep-learning COMPARISON model (CLAUDE.md
Section 6). XGBoost is the favored default for a dataset this size/shape
(small tabular-ish data), but we build this alongside it to demonstrate the
DL approach and let the owner compare the two directly, which is itself a
learning goal (CLAUDE.md: "owner is learning ML/DL/LLM concepts").

Why an LSTM at all, given XGBoost already sees engineered trend features:
the LSTM instead consumes a 60-SECOND ROLLING WINDOW of raw vitals directly
(not the hand-engineered spo2_trend_5min etc.) and learns its own temporal
representation. This is the classic tradeoff being demonstrated: XGBoost
needs a human to engineer "trend" as a feature; an LSTM can in principle
learn temporal patterns (rising/falling/oscillating SpO2) from the raw
sequence itself. Small, 1-2 layers per CLAUDE.md -- this is a learning
comparison, not a bid to squeeze out maximum performance from a deep model
on a modestly-sized dataset.

Windowing strategy: rather than one window per timestep (which at 1Hz over
2M+ rows would produce ~2M highly-overlapping, near-duplicate windows and
make training extremely slow for no real benefit -- consecutive windows
one second apart are >98% identical), we STRIDE across each subject's
trajectory, taking a window every STRIDE_SECONDS. This keeps window
diversity high per unit of training time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_WINDOW_SECONDS,
    MODELS_DIR,
    N_TIERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE_HZ,
)

WINDOW_SAMPLES = LSTM_WINDOW_SECONDS * SAMPLE_RATE_HZ
STRIDE_SECONDS = 15  # take a window every 15s of a trajectory, not every 1s -- see module docstring

# Raw per-timestep features fed to the LSTM. Deliberately NOT the same as
# XGBoost's FEATURE_COLUMNS (which include hand-engineered trend/delta
# features) -- the whole point of this comparison model is seeing whether
# an LSTM can learn temporal structure from raw vitals + altitude alone.
LSTM_RAW_COLUMNS = ["spo2", "hr", "temp", "altitude"]
MODEL_PATH = MODELS_DIR / "lstm_severity.pt"


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide fixed-length windows over each subject's trajectory (strided, see
    module docstring), labeling each window with the severity_index of its
    LAST timestep -- i.e. "given the last 60 seconds of vitals, what's the
    severity RIGHT NOW" -- which mirrors exactly how the live pipeline would
    use this model (Stage 4 of the data flow diagram, fed the trailing
    buffer at the current moment).
    """
    X_windows = []
    y_windows = []

    for _subject_id, group in df.groupby("subject_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[LSTM_RAW_COLUMNS].to_numpy(dtype=np.float32)
        labels = group["severity_index"].to_numpy()

        n = len(group)
        if n < WINDOW_SAMPLES:
            continue  # trajectory too short for even one full window

        # range(0, n - WINDOW_SAMPLES + 1, stride): every possible window
        # start position, stepping STRIDE_SECONDS at a time instead of 1 --
        # "+1" on the stop bound because range()'s upper bound is exclusive
        # and the LAST valid start position must still leave room for a
        # full WINDOW_SAMPLES-length slice before the trajectory ends.
        for start in range(0, n - WINDOW_SAMPLES + 1, STRIDE_SECONDS * SAMPLE_RATE_HZ):
            end = start + WINDOW_SAMPLES
            X_windows.append(values[start:end])
            y_windows.append(labels[end - 1])  # label the window by its LAST timestep's severity, see docstring

    return np.stack(X_windows), np.array(y_windows)


class VitalsWindowDataset(Dataset):
    """
    Thin torch Dataset wrapper. Normalization stats (mean/std) are computed
    ONCE on the training set and passed in here, never recomputed per split
    -- fitting normalization on val/test would leak their distribution into
    what the model implicitly "knows," the same leakage principle as the
    temporal split itself.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.X = (X - mean) / std
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


class SeverityLSTM(nn.Module):
    """
    Small 2-layer LSTM -> final hidden state -> linear classifier head.

    Deliberately small (hidden_size=32-64, 2 layers) per CLAUDE.md's "keep
    it a comparison model, not a model zoo" instruction, and because a
    dataset this size doesn't warrant (and risks overfitting with) a large
    model. We classify directly into N_TIERS classes here with cross-
    entropy (unlike XGBoost's regression+threshold approach) -- comparing
    the two different ordinal-handling STRATEGIES is itself informative:
    XGBoost enforces ordinality via a regression target; the LSTM instead
    gets class weighting in its loss (see train()) to handle imbalance
    without explicitly enforcing tier ordering. Both are compared fairly
    using the SAME metrics.py, including mean_abs_tier_error, which is
    where any lack of ordinal-awareness would actually show up in practice.
    """

    def __init__(self, n_features: int, hidden_size: int = 48, n_layers: int = 2):
        # hidden_size=48: width of the LSTM's internal hidden/cell state --
        # controls how much temporal information it can carry forward;
        # deliberately small (see class docstring) since this is a
        # comparison model, not a bid for max performance.
        # n_layers=2: stacked LSTM layers (each layer's hidden sequence feeds
        # the next) -- 2 is enough to let the model compose simple temporal
        # patterns without the overfitting risk of going deeper.
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,  # width of each timestep's input vector (len(LSTM_RAW_COLUMNS) == 4)
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,  # expect input shaped (batch, seq_len, features) instead of PyTorch's default (seq_len, batch, features)
            dropout=0.2 if n_layers > 1 else 0.0,  # dropout BETWEEN stacked LSTM layers, regularizing the inter-layer connections -- PyTorch ignores/warns if set with only 1 layer, hence the guard
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # dropout in the classifier head -- separate regularization from the LSTM's own inter-layer dropout above
            nn.Linear(32, N_TIERS),  # final layer outputs one raw logit per severity tier, consumed as class scores by CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's final hidden state
        return self.head(last_layer_hidden)  # (batch, N_TIERS) logits


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 12,  # max training passes over the full train set -- capped low since early stopping (patience below) usually cuts this short anyway
    batch_size: int = 256,  # number of windows per gradient update -- large enough for stable gradients, small enough to fit comfortably in memory/CPU
    lr: float = 1e-3,  # Adam's learning rate -- standard default step size for this optimizer, rarely needs tuning for a model this small
) -> tuple[SeverityLSTM, np.ndarray, np.ndarray]:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = build_windows(train_df)
    X_val, y_val = build_windows(val_df)
    print(f"LSTM windows -- train: {len(y_train):,}, val: {len(y_val):,}")

    # X_train has shape (n_windows, WINDOW_SAMPLES, n_features); reshape
    # flattens the window/timestep dims together so mean/std are computed
    # per-feature ACROSS every timestep of every window (one mean/std per
    # of the 4 LSTM_RAW_COLUMNS), not per-window or per-timestep.
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6  # avoid div-by-zero

    train_ds = VitalsWindowDataset(X_train, y_train, mean, std)
    val_ds = VitalsWindowDataset(X_val, y_val, mean, std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SeverityLSTM(n_features=len(LSTM_RAW_COLUMNS)).to(device)

    # Class weights for the loss, same "balanced" logic as XGBoost's sample
    # weights -- computed from TRAIN labels only (see xgb_ordinal.py's
    # identical reasoning about not accepting an always-predict-Normal
    # model given the expected class imbalance).
    class_counts = np.bincount(y_train, minlength=N_TIERS).astype(np.float32)
    class_weights = (class_counts.sum() / (N_TIERS * np.clip(class_counts, 1, None))).astype(
        np.float32
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))  # per-class weight scales each sample's loss contribution -- this is what makes rare tiers "count more" during backprop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam: adaptive per-parameter learning rates -- standard, low-maintenance choice for a small model like this over plain SGD

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 3, 0  # patience=3: stop training if val loss hasn't improved for 3 consecutive epochs (see the early-stopping check below)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)  # de-average the batch's mean loss back to a sum, weighted by batch size (last batch may be smaller) -- so dividing by len(train_ds) below gives a true per-sample average
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)  # same de-averaging as the train loop above
        val_loss /= len(val_ds)

        print(f"  epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Early stopping on val loss -- with only a 12-epoch budget this is
        # mostly a safety net, but it's the same principle as XGBoost's
        # early_stopping_rounds: don't keep training past the point val
        # performance stops improving, and keep the BEST checkpoint, not
        # just whatever the last epoch happened to produce.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, mean, std


def save(model: SeverityLSTM, mean: np.ndarray, std: np.ndarray) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "raw_columns": LSTM_RAW_COLUMNS,
            "window_samples": WINDOW_SAMPLES,
        },
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")


def load() -> tuple[SeverityLSTM, np.ndarray, np.ndarray]:
    checkpoint = torch.load(MODEL_PATH, weights_only=False)  # weights_only=False needed since the checkpoint bundles plain numpy arrays (mean/std) alongside the state_dict, not just tensor weights
    model = SeverityLSTM(n_features=len(checkpoint["raw_columns"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]


def predict(model: SeverityLSTM, mean: np.ndarray, std: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, y_true) arrays built from sliding windows over df."""
    X, y_true = build_windows(df)
    X_norm = (X - mean) / std
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():  # inference only -- skip gradient tracking to save memory/time
        for i in range(0, len(X_norm), 512):  # manual batching (fixed chunk size of 512) instead of a DataLoader -- avoids the loader's shuffling/worker overhead for a one-off prediction pass
            batch = torch.from_numpy(X_norm[i : i + 512]).float().to(device)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())  # argmax over class logits -> the predicted tier index; .cpu() first in case device is GPU
    return np.concatenate(preds), y_true


if __name__ == "__main__":
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run `python -m src.data.feature_engineering` first."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("Training LSTM comparison model...")
    model, mean, std = train(train_df, val_df)
    save(model, mean, std)

    from src.models.metrics import evaluate, print_report

    y_pred, y_true = predict(model, mean, std, test_df)
    result = evaluate(y_true, y_pred, model_name="LSTM (test set)")
    print_report(result)
