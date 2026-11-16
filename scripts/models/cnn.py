"""
NN-2: CNN on the cached log-mel spectrograms.

Uses the same train/val/test split as every other model (track ids come from
load_split(return_ids=True)). Settings are tuned on the validation set and every
run is saved to results/cnn_tuning.csv; the test set is only used once, for the
best setting. Training time and hardware are printed and saved with the results.

Run from the repo root (F21DL-main/):
    python scripts/models/cnn.py            # full grid
    python scripts/models/cnn.py --quick    # 1 setting, 2 epochs, to check it runs
"""

import argparse
import copy
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

# Make "scripts" importable no matter where this file is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402

SEED = 42
SPEC_DIR = os.path.join(ROOT, "data", "processed", "spectrograms")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 6  # stop if val macro-F1 hasn't improved for this many epochs

# Small grid, tuned on the validation set
PARAM_GRID = [
    {"lr": lr, "dropout": d, "channels": (16, 32, 64, 128)}
    for lr in (1e-3, 3e-4)
    for d in (0.3, 0.5)
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_description(device):
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(0)} (CUDA)"
    return f"{platform.processor() or platform.machine()} {platform.platform()} ({device.type})"


def load_spectrograms(track_ids):
    """Stack the cached (128, 1292) float16 spectrograms for these track ids into (N, 1, 128, 1292)."""
    arrays = [np.load(os.path.join(SPEC_DIR, f"{int(t)}.npy")) for t in track_ids]
    return torch.from_numpy(np.stack(arrays))[:, None]


def train_stats(x_train):
    """Mean/std of the TRAIN spectrograms only, computed in chunks to keep memory low."""
    total, total_sq, n = 0.0, 0.0, 0
    for i in range(0, len(x_train), 256):
        chunk = x_train[i:i + 256].double()
        total += chunk.sum().item()
        total_sq += (chunk ** 2).sum().item()
        n += chunk.numel()
    mean = total / n
    return mean, (total_sq / n - mean ** 2) ** 0.5


class SpectrogramCNN(nn.Module):
    """4 x (conv -> batch norm -> ReLU -> max-pool), global average pool, dropout, linear."""

    def __init__(self, channels=(16, 32, 64, 128), dropout=0.3, n_classes=8):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def predict(model, x, mean, std, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            batch = (x[i:i + 64].to(device).float() - mean) / std
            preds.append(model(batch).argmax(1).cpu())
    return torch.cat(preds).numpy()


def score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


# AI Generated Function
# Prompt: Write a training loop for a spectrogram CNN in PyTorch that trains with AdamW and cross-entropy, evaluates val accuracy and macro-F1 every epoch, keeps the best epoch's weights by val macro-F1, stops early after PATIENCE epochs without improvement, and returns the model, per-epoch history and training time
def train(params, data, device, max_epochs=MAX_EPOCHS):
    x_train, y_train, x_val, y_val, mean, std = data
    torch.manual_seed(SEED)
    model = SpectrogramCNN(params["channels"], params["dropout"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    y_train_t = torch.as_tensor(y_train.to_numpy())

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        running = 0.0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb = (x_train[idx].to(device).float() - mean) / std
            yb = y_train_t[idx].to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(idx)

        val_acc, val_f1 = score(y_val, predict(model, x_val, mean, std, device))
        history["train_loss"].append(running / len(order))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"    epoch {epoch:2d}  loss={history['train_loss'][-1]:.4f}  val acc={val_acc:.4f}  val macro-F1={val_f1:.4f}", flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch, time.time() - start


# AI Generated Code
# Prompt: Write a CNN script (PyTorch) for the FMA log-mel spectrograms in data/processed/spectrograms using the same split as load_split(return_ids=True). Tune a small grid of learning rate and dropout on the validation set by macro-F1, save every run to results/cnn_tuning.csv, evaluate the best setting once on test (accuracy, macro-F1, classification report, confusion matrix), record training time and hardware, and save training curves to figures/cnn_training_curve.png. Follow the structure of scripts/models/mlp.py.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 setting, 2 epochs, just to check the script runs")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = get_device()
    hardware = hardware_description(device)
    print(f"Device: {device}   Hardware: {hardware}")

    # Labels and track ids come from the shared split; features are unused (the CNN reads spectrograms)
    (_, _, _, y_train, y_val, y_test, ids_train, ids_val, ids_test) = load_split(scale=False, return_ids=True)
    class_names = get_label_names()

    print("Loading spectrograms...")
    x_train, x_val, x_test = (load_spectrograms(ids) for ids in (ids_train, ids_val, ids_test))
    mean, std = train_stats(x_train)
    print(f"train={tuple(x_train.shape)} val={tuple(x_val.shape)} test={tuple(x_test.shape)}  mean={mean:.2f} std={std:.2f}")
    data = (x_train, y_train, x_val, y_val, mean, std)

    grid = PARAM_GRID[:1] if args.quick else PARAM_GRID
    epochs = 2 if args.quick else MAX_EPOCHS

    # 1. Tune on validation, record every run
    rows, best = [], None
    for params in grid:
        print(f"\nTraining {params}")
        model, history, best_epoch, secs = train(params, data, device, epochs)
        val_acc, val_f1 = history["val_acc"][best_epoch - 1], history["val_f1"][best_epoch - 1]
        print(f"  -> best epoch {best_epoch}  val acc={val_acc:.4f}  val macro-F1={val_f1:.4f}  {secs / 60:.1f} min")
        rows.append({**params, "channels": str(params["channels"]), "best_epoch": best_epoch,
                     "val_acc": val_acc, "val_macro_f1": val_f1, "train_minutes": secs / 60, "hardware": hardware})
        if best is None or val_f1 > best["val_f1"]:
            best = {"params": params, "model": model, "history": history, "best_epoch": best_epoch,
                    "val_acc": val_acc, "val_f1": val_f1, "secs": secs}

    pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "cnn_tuning.csv"), index=False
    )

    # 2. Evaluate the best run once on test
    test_pred = predict(best["model"], x_test, mean, std, device)
    test_acc, test_f1 = score(y_test, test_pred)
    print(f"\nBest params: {best['params']}  (epoch {best['best_epoch']}, {best['secs'] / 60:.1f} min on {hardware})")
    print(f"Validation: accuracy={best['val_acc']:.4f}  macro-F1={best['val_f1']:.4f}")
    print(f"Test:       accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print("Target to beat (SVM on MFCCs, FMA paper): 46.4% accuracy")

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=class_names, digits=3))

    print("Test confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_test, test_pred)
    short = [n[:6] for n in class_names]
    print(" " * 14 + " ".join(f"{s:>6}" for s in short))
    for name, row in zip(class_names, cm):
        print(f"{name:>13} " + " ".join(f"{v:>6}" for v in row))

    # 3. Training curves for the best run
    h = best["history"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"])
    ax1.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    ax2.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label="val accuracy")
    ax2.plot(range(1, len(h["val_f1"]) + 1), h["val_f1"], label="val macro-F1")
    ax2.axvline(best["best_epoch"], color="grey", linestyle="--", label="chosen epoch")
    ax2.set(title="Validation", xlabel="Epoch", ylabel="Score")
    ax2.legend()
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "cnn_training_curve.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
