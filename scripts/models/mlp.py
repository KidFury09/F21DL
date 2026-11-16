"""
NN-1: MLP on the 518 FMA audio features.

Uses the shared split from load_split() so results are comparable with the
other models. Hyperparameters are tuned on the validation set (every setting
tried is saved to results/mlp_tuning.csv); the test set is only used once at
the end.

"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

# Make "scripts" importable no matter where this file is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402

RANDOM_STATE = 42
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

# Small grid, tuned on the validation set.
# alpha = L2 regularisation
PARAM_GRID = [
    {"hidden_layer_sizes": h, "alpha": a, "learning_rate_init": lr}
    for h in ((256,), (512, 256), (256, 128, 64))
    for a in (1e-4, 1e-2)
    for lr in (1e-3, 3e-4)
]


def make_model(params):
    # early_stopping holds out 10% of TRAIN internally, so the real validation
    # set is never used for anything except choosing between settings.
    return MLPClassifier(
        activation="relu",
        solver="adam",
        batch_size=128,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        **params,
    )



#function to evaluate the model on a given dataset
def evaluate(model, X, y):
    pred = model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


#make sure the results and figures directories exist, load the data, tune hyperparameters, evaluate the best model on the test set, and save training curves
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # MLPs need scaled inputs (scaler is fitted on train only inside load_split)
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(scale=True)
    class_names = get_label_names()
    print(f"Features: {X_train.shape[1]}  train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # 1. Tune on validation, record everything that was tried
    rows, best_params, best_f1 = [], None, -1.0
    print("\nTuning on validation set:")
    for params in PARAM_GRID:
        start = time.time()
        model = make_model(params).fit(X_train, y_train)
        secs = time.time() - start
        acc, f1, _ = evaluate(model, X_val, y_val)
        print(f"  {params}  val acc={acc:.4f}  val macro-F1={f1:.4f}  epochs={model.n_iter_}  {secs:.1f}s")
        rows.append({**params, "val_acc": acc, "val_macro_f1": f1, "epochs": model.n_iter_, "train_secs": secs})
        if f1 > best_f1:
            best_params, best_f1 = params, f1

    pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "mlp_tuning.csv"), index=False
    )
    print(f"\nBest params: {best_params}")

    # 2. Refit with the best params and evaluate once on test
    start = time.time()
    model = make_model(best_params).fit(X_train, y_train)
    train_secs = time.time() - start

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)

    print(f"\nTraining time: {train_secs:.1f}s ({model.n_iter_} epochs)")
    print(f"Validation: accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
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

    # 3. Training curves (loss on train, accuracy on the internal early-stopping hold-out)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(model.loss_curve_)
    ax1.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    ax2.plot(model.validation_scores_)
    ax2.set(title="Hold-out accuracy (10% of train)", xlabel="Epoch", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mlp_training_curve.png"), dpi=150)
    print(f"\nSaved {os.path.join(FIGURES_DIR, 'mlp_training_curve.png')}")
