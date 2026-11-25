"""
NN-3: CNN data augmentation on the spectrograms.

Takes the best CNN setting from NN-2 (lr 0.001, dropout 0.5) and trains it with a
few augmentations, compared against the same CNN with no augmentation. Everything
else is identical (same split, same seed, same epoch limit), so the only thing
that changes between runs is the augmentation. Augmentation is only applied to
training batches.

Settings are compared on validation macro-F1 and saved to results/cnn_augmentation.csv.
The test set is only used for the best augmentation and for the plain control.

Run from the repo root (F21DL-main/):
    python scripts/models/cnn_augment.py            # all 4 runs
    python scripts/models/cnn_augment.py --quick    # 1 run, 2 epochs, to check it runs
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

# Make "scripts" importable no matter where this file is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402
from scripts.models.cnn import (  # noqa: E402
    FIGURES_DIR, RESULTS_DIR, get_device, hardware_description,
    load_spectrograms, predict, score, train, train_stats,
)

# Best setting found in NN-2
BEST_PARAMS = {"lr": 1e-3, "dropout": 0.5, "channels": (16, 32, 64, 128)}

# The NN-2 best epochs were 28 of 30, so give every run (including the control) more room
MAX_EPOCHS = 40


# AI Generated Function
# Prompt: Write a batched SpecAugment-style augmentation for a normalised spectrogram batch of shape (B, 1, n_mels, n_frames) on any torch device. Zero out n_freq random frequency bands of up to max_freq mel bins and n_time random time bands of up to max_time frames, with a different random position for every sample in the batch
def spec_mask(x, n_freq=2, max_freq=20, n_time=2, max_time=100):
    b, _, n_mels, n_frames = x.shape
    f_idx = torch.arange(n_mels, device=x.device).view(1, 1, n_mels, 1)
    t_idx = torch.arange(n_frames, device=x.device).view(1, 1, 1, n_frames)
    for _ in range(n_freq):
        width = torch.randint(0, max_freq + 1, (b, 1, 1, 1), device=x.device)
        start = (torch.rand(b, 1, 1, 1, device=x.device) * (n_mels - width)).long()
        x = x.masked_fill((f_idx >= start) & (f_idx < start + width), 0.0)  # 0 = the train mean
    for _ in range(n_time):
        width = torch.randint(0, max_time + 1, (b, 1, 1, 1), device=x.device)
        start = (torch.rand(b, 1, 1, 1, device=x.device) * (n_frames - width)).long()
        x = x.masked_fill((t_idx >= start) & (t_idx < start + width), 0.0)
    return x


# AI Generated Function
# Prompt: Write a batched random circular time shift for a spectrogram batch of shape (B, 1, n_mels, n_frames), shifting each sample by a different random number of frames up to max_shift in either direction
def time_shift(x, max_shift=200):
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],)).tolist()
    return torch.stack([torch.roll(xi, s, dims=-1) for xi, s in zip(x, shifts)])


# AI Generated Function
# Prompt: Write a batched augmentation for a normalised spectrogram batch that adds small gaussian noise and a random per-sample gain (a constant offset in dB space)
def noise_and_gain(x, noise_std=0.1, gain_std=0.2):
    gain = torch.randn(x.shape[0], 1, 1, 1, device=x.device) * gain_std
    return x + torch.randn_like(x) * noise_std + gain


AUGMENTATIONS = {
    "none (control)": None,
    "spec_mask": spec_mask,
    "time_shift": time_shift,
    "noise_and_gain": noise_and_gain,
}


# AI Generated Code
# Prompt: Write a script that retrains the best NN-2 CNN setting with several spectrogram augmentations (SpecAugment-style masking, time shift, noise and gain) plus a no-augmentation control using the same epoch limit, compares them on validation macro-F1, saves all runs to results/cnn_augmentation.csv, evaluates the best augmentation and the control once on test, and saves validation curves to figures/cnn_augmentation_curves.png. Reuse train() from scripts/models/cnn.py.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 run, 2 epochs, just to check the script runs")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = get_device()
    hardware = hardware_description(device)
    print(f"Device: {device}   Hardware: {hardware}")

    (_, _, _, y_train, y_val, y_test, ids_train, ids_val, ids_test) = load_split(scale=False, return_ids=True)
    class_names = get_label_names()

    print("Loading spectrograms...")
    x_train, x_val, x_test = (load_spectrograms(ids) for ids in (ids_train, ids_val, ids_test))
    mean, std = train_stats(x_train)
    data = (x_train, y_train, x_val, y_val, mean, std)

    runs = list(AUGMENTATIONS.items())
    if args.quick:
        runs = runs[1:2]  # just spec_mask
    epochs = 2 if args.quick else MAX_EPOCHS

    results = {}
    for name, fn in runs:
        print(f"\nTraining with augmentation: {name}")
        model, history, best_epoch, secs = train(BEST_PARAMS, data, device, epochs, augment=fn)
        results[name] = {
            "model": model, "history": history, "best_epoch": best_epoch, "secs": secs,
            "val_acc": history["val_acc"][best_epoch - 1], "val_f1": history["val_f1"][best_epoch - 1],
        }
        r = results[name]
        print(f"  -> best epoch {best_epoch}  val acc={r['val_acc']:.4f}  val macro-F1={r['val_f1']:.4f}  {secs / 60:.1f} min")

    # Test: only the best augmentation and the plain control
    best_aug = max((n for n in results if n != "none (control)"), key=lambda n: results[n]["val_f1"])
    to_test = [n for n in ("none (control)", best_aug) if n in results]
    for name in to_test:
        pred = predict(results[name]["model"], x_test, mean, std, device)
        results[name]["test_acc"], results[name]["test_f1"] = score(y_test, pred)
        results[name]["test_pred"] = pred

    rows = [{
        "augmentation": n, "lr": BEST_PARAMS["lr"], "dropout": BEST_PARAMS["dropout"],
        "max_epochs": epochs, "best_epoch": r["best_epoch"], "val_acc": r["val_acc"],
        "val_macro_f1": r["val_f1"], "test_acc": r.get("test_acc"), "test_macro_f1": r.get("test_f1"),
        "train_minutes": r["secs"] / 60, "hardware": hardware,
    } for n, r in results.items()]
    pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "cnn_augmentation.csv"), index=False
    )

    print("\n=== Summary (same CNN, same seed, same epoch limit) ===")
    print(f"{'augmentation':16s} {'val acc':>8s} {'val F1':>8s} {'test acc':>9s} {'test F1':>8s}")
    for n, r in results.items():
        t_acc = f"{r['test_acc']:.4f}" if "test_acc" in r else "-"
        t_f1 = f"{r['test_f1']:.4f}" if "test_f1" in r else "-"
        print(f"{n:16s} {r['val_acc']:8.4f} {r['val_f1']:8.4f} {t_acc:>9s} {t_f1:>8s}")
    print(f"\nBest augmentation on validation: {best_aug}")

    print(f"\nTest classification report ({best_aug}):")
    print(classification_report(y_test, results[best_aug]["test_pred"], target_names=class_names, digits=3))
    print("Test confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_test, results[best_aug]["test_pred"])
    short = [n[:6] for n in class_names]
    print(" " * 14 + " ".join(f"{s:>6}" for s in short))
    for name, row in zip(class_names, cm):
        print(f"{name:>13} " + " ".join(f"{v:>6}" for v in row))

    # Validation macro-F1 per epoch for every run
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, r in results.items():
        ax.plot(range(1, len(r["history"]["val_f1"]) + 1), r["history"]["val_f1"], label=n)
    ax.set(title="Validation macro-F1 by augmentation", xlabel="Epoch", ylabel="Macro-F1")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "cnn_augmentation_curves.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
