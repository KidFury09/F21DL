"""
R3 baseline: Random Forest on the FMA audio features.

Uses the shared split from load_split() so results are comparable with the
other models. Hyperparameters (including how many of the EDA-2 ranked features
to use) are picked on the validation set; the test set is only used once at
the end.

Outputs:
    results/rf_tuning.csv               every setting tried, sorted by val macro-F1
    figures/rf_confusion_matrix.png     test confusion matrix
    figures/rf_feature_importance.png   20 most important features of the final model

Run from the repo root (F21DL-main/):
    python scripts/models/random_forest.py
"""

# AI Generated Code (Claude Code, model Opus 5.5)
# Original prompt: Tune a RandomForestClassifier on the validation set using PARAM_GRID, refit with
# best macro-F1 params, and print test accuracy, macro-F1, classification report, confusion matrix,
# and top 10 feature importances
# Follow-up prompt: update random forest to give same outputs as other two
# (context given to the model: match decision_tree.py / naive_bayes.py, i.e. tune the number of
#  features too, save a tuning CSV, a confusion matrix figure and a feature importance figure)

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

# Make "scripts" importable no matter where this file is launched from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402
from scripts.models.evaluate import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    evaluate,
    plot_confusion,
    print_test_report,
    save_result,
)

RANDOM_STATE = 42

# Number of top ANOVA features (ranked on X_train only inside load_split). None = all 518.
# EDA-2 found RF plateaus around 150-200 features, so compare 150 against all 518.
N_FEATURES_GRID = (150, None)

# Small grid, tuned on the validation set
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "max_features": "sqrt"}
    for n in (200, 500)
    for d in (None, 20)
    for leaf in (1, 3)
]


def make_model(params):
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    class_names = get_label_names()

    # 1. Tune on validation (trees don't need scaling)
    rows, best, best_f1 = [], None, -1.0
    print("Tuning on validation set:")
    for n_features in N_FEATURES_GRID:
        X_train, X_val, _, y_train, y_val, _ = load_split(scale=False, n_features=n_features)
        k = X_train.shape[1]
        for params in PARAM_GRID:
            start = time.time()
            model = make_model(params).fit(X_train, y_train)
            secs = time.time() - start
            acc, f1, _ = evaluate(model, X_val, y_val)
            print(f"  k={k:3d} {params}  val acc={acc:.4f}  val macro-F1={f1:.4f}  {secs:.1f}s")
            rows.append({"n_features": k, **params, "val_acc": acc, "val_macro_f1": f1,
                         "train_secs": secs})
            if f1 > best_f1:
                best, best_f1 = (n_features, params), f1

    tuning = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    tuning.to_csv(os.path.join(RESULTS_DIR, "rf_tuning.csv"), index=False)

    best_n_features, best_params = best
    print(f"\nBest: n_features={best_n_features or 518}  {best_params}")

    # 2. Refit with the best setting and evaluate once on test
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(scale=False, n_features=best_n_features)
    start = time.time()
    model = make_model(best_params).fit(X_train, y_train)
    train_secs = time.time() - start

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)

    print(f"\nTraining time: {train_secs:.1f}s")
    print(f"Validation: accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
    print(f"Test:       accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print("Target to beat (SVM on MFCCs, FMA paper): 46.4% accuracy")

    print_test_report(y_test, test_pred, class_names)

    # 3. Most important features
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 features by importance:")
    for col, imp in importances.head(10).items():
        print(f"  {col:30s} {imp:.4f}")

    # 4. Figures and shared results table
    print()
    plot_confusion(y_test, test_pred, class_names, "Random Forest", "rf_confusion_matrix.png")

    top =importances.head(20)[::-1]  # reversed so the most important is at the top
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top.values)
    ax.set(xlabel="Mean decrease in impurity", title="Random Forest - 20 most important features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    imp_path = os.path.join(FIGURES_DIR, "rf_feature_importance.png")
    fig.savefig(imp_path, dpi=150)
    plt.close(fig)

    print(f"Saved {imp_path}")
    print(f"Saved {os.path.join(RESULTS_DIR, 'rf_tuning.csv')}")

    save_result(
        "Random Forest", val_acc, val_f1, y_test, test_pred, class_names, train_secs,
        notes=f"{X_train.shape[1]} features, {best_params['n_estimators']} trees, "
              f"max_depth={best_params['max_depth']}, min_samples_leaf={best_params['min_samples_leaf']}",
    )
