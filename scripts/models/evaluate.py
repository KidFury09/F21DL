"""
BASE-1: Shared evaluation for every model (baselines, MLP, CNN).

Every model script should score itself with these functions so all results are
computed the same way and land in one table:

    results/results_summary.csv   one row per model (re-running a model replaces its row)

Usage in a model script (scripts/models/<model>.py):

    from scripts.models.evaluate import evaluate, print_test_report, plot_confusion, save_result

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)
    print_test_report(y_test, test_pred, class_names)
    plot_confusion(y_test, test_pred, class_names, "Random Forest", "rf_confusion_matrix.png")
    save_result("Random Forest", val_acc, val_f1, y_test, test_pred, class_names,
                train_secs, notes="518 features, 200 trees")

"""

# AI Assisted Code (Claude Code, model Opus 5.5)
# Prompt: make an evaluate file that makes the scripts use it which gives a shared module with evaluate(), 
# a printed test report, a confusion matrix figure, and save_result() 
# that adds one row per model in results/results_summary.csv including per-genre test F1


import os

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")

FMA_PAPER_BASELINE = 0.464  # SVM on MFCCs, FMA paper (Defferrard et al., 2017)


def evaluate(model, X, y):
    """Predict on X and return (accuracy, macro-F1, predictions)."""
    pred = model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


def print_test_report(y_true, pred, class_names):
    """Print the per-genre classification report and a text confusion matrix."""
    print("\nTest classification report:")
    print(classification_report(y_true, pred, target_names=class_names, digits=3))

    print("Test confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, pred)
    short = [n[:6] for n in class_names]
    print(" " * 14 + " ".join(f"{s:>6}" for s in short))
    for name, row in zip(class_names, cm):
        print(f"{name:>13} " + " ".join(f"{v:>6}" for v in row))


def plot_confusion(y_true, pred, class_names, model_name, filename):
    """Save the test confusion matrix as figures/<filename>. Returns the path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=class_names, cmap="Blues", colorbar=False,
        xticks_rotation=45, ax=ax,
    )
    ax.set_title(f"{model_name} - test confusion matrix (acc {acc:.1%}, macro-F1 {f1:.3f})")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def save_result(model_name, val_acc, val_f1, y_test, test_pred, class_names, train_secs=None, notes=""):
    """
    Add (or replace) this model's row in results/results_summary.csv.

    The row holds validation and test accuracy / macro-F1, the test F1 of every
    genre (f1_<genre>), training time and a free-text note (e.g. the chosen setting).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_genre = f1_score(y_test, test_pred, average=None, labels=range(len(class_names)))

    row = {
        "model": model_name,
        "val_acc": round(float(val_acc), 4),
        "val_macro_f1": round(float(val_f1), 4),
        "test_acc": round(accuracy_score(y_test, test_pred), 4),
        "test_macro_f1": round(f1_score(y_test, test_pred, average="macro"), 4),
        **{f"f1_{name}": round(float(f), 4) for name, f in zip(class_names, per_genre)},
        "train_secs": None if train_secs is None else round(float(train_secs), 2),
        "notes": notes,
    }

    if os.path.exists(SUMMARY_CSV):
        table = pd.read_csv(SUMMARY_CSV)
        table = table[table["model"] != model_name]  # replace, don't duplicate
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])

    table = table.sort_values("test_macro_f1", ascending=False)
    table.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {model_name} row to {SUMMARY_CSV}")
    return row
