"""
R3 baseline: Random Forest on the 518 FMA audio features.

Uses the shared split from load_split() so results are comparable with the
other models. Hyperparameters are picked on the validation set; the test set
is only used once at the end.

Run from the repo root (F21DL-main/):
    python scripts/models/random_forest.py
"""

import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Make "scripts" importable no matter where this file is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402

RANDOM_STATE = 42

# Small grid, tuned on the validation set
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "max_features": "sqrt"}
    for n in (200, 500)
    for d in (None, 20)
]


# AI Generated Function
# Prompt: Predict on X using the given model and return accuracy, macro-F1 score, and predictions
def evaluate(model, X, y):
    pred = model.predict(X)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


# AI Generated Code
# Prompt: Tune a RandomForestClassifier on the validation set using PARAM_GRID, refit with best macro-F1 params, and print test accuracy, macro-F1, classification report, confusion matrix, and top 10 feature importances
if __name__ == "__main__":
    # Tree models don't need scaling
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(scale=False)
    class_names = get_label_names()

    # 1. Tune on validation
    best_params, best_f1 = None, -1.0
    print("Tuning on validation set:")
    for params in PARAM_GRID:
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        acc, f1, _ = evaluate(model, X_val, y_val)
        print(f"  {params}  val acc={acc:.4f}  val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_params, best_f1 = params, f1

    print(f"\nBest params: {best_params}")

    # 2. Refit with the best params and evaluate once on test
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    model.fit(X_train, y_train)

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)

    print(f"\nValidation: accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
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

    # Top 10 most important features
    importances = sorted(zip(model.feature_importances_, X_train.columns), reverse=True)[:10]
    print("\nTop 10 features by importance:")
    for imp, col in importances:
        print(f"  {col:30s} {imp:.4f}")
