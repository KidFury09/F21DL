"""
R3 baseline: Decision Tree on the FMA audio features.

Uses the shared split from load_split() so results are comparable with the
other models. Hyperparameters (including how many of the EDA-2 ranked features
to use) are picked on the validation set; the test set is only used once at
the end.

Outputs:
    results/dt_tuning.csv            every setting tried, sorted by val macro-F1
    figures/dt_confusion_matrix.png  test confusion matrix
    figures/dt_tree_top_levels.png   top 2 levels of the final tree (interpretability)

Run from the repo root (F21DL-main/):
    python scripts/models/decision_tree.py
"""

# AI Generated Code (Claude Code, model Opus 5.5)
# Prompt: please make the code for decision trees and naive bayes
# (context given to the model: follow the structure of scripts/models/random_forest.py,
#  use load_split(), tune on validation macro-F1, evaluate on test once, save the
#  tuning results, a confusion matrix and a plot of the top tree levels)

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

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

# EDA-2 found the tree does better with fewer features (peak ~k=25-100, worse at 518),
# so the number of top ANOVA features is tuned along with the tree settings.
# None = all 518 features.
N_FEATURES_GRID = (50, 150, None)

PARAM_GRID = [
    {"criterion": c, "max_depth": d, "min_samples_leaf": leaf}
    for c in ("gini", "entropy")
    for d in (5, 10, 15, None)
    for leaf in (1, 5, 20)
]


def make_model(params):
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)


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
            print(f"  k={k:3d} {params}  val acc={acc:.4f}  val macro-F1={f1:.4f}  depth={model.get_depth()}")
            rows.append({"n_features": k, **params, "val_acc": acc, "val_macro_f1": f1,
                         "tree_depth": model.get_depth(), "n_leaves": model.get_n_leaves(),
                         "train_secs": secs})
            if f1 > best_f1:
                best, best_f1 = (n_features, params), f1

    tuning = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    tuning.to_csv(os.path.join(RESULTS_DIR, "dt_tuning.csv"), index=False)

    best_n_features, best_params = best
    print(f"\nBest: n_features={best_n_features or 518}  {best_params}")

    # 2. Refit with the best setting and evaluate once on test
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(scale=False, n_features=best_n_features)
    start = time.time()
    model = make_model(best_params).fit(X_train, y_train)
    train_secs = time.time() - start

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)

    print(f"\nTraining time: {train_secs:.2f}s  depth={model.get_depth()}  leaves={model.get_n_leaves()}")
    print(f"Validation: accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
    print(f"Test:       accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print("Target to beat (SVM on MFCCs, FMA paper): 46.4% accuracy")

    print_test_report(y_test, test_pred, class_names)

    # 3. Interpretability: the rules at the top of the tree and the most used features
    print("\nTop 3 levels of the tree (the first splits it makes):")
    print(export_text(model, feature_names=list(X_train.columns), class_names=class_names, max_depth=3))

    importances = sorted(zip(model.feature_importances_, X_train.columns), reverse=True)[:10]
    print("Top 10 features by importance:")
    for imp, col in importances:
        print(f"  {col:30s} {imp:.4f}")

    # 4. Figures and shared results table
    print()
    plot_confusion(y_test, test_pred, class_names, "Decision Tree", "dt_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, max_depth=2, precision=2, feature_names=list(X_train.columns), class_names=class_names,
              filled=True, impurity=False, proportion=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Decision Tree - top 2 levels (value = share of each genre at that node)")
    fig.tight_layout()
    tree_path = os.path.join(FIGURES_DIR, "dt_tree_top_levels.png")
    fig.savefig(tree_path, dpi=150)
    plt.close(fig)

    print(f"Saved {tree_path}")
    print(f"Saved {os.path.join(RESULTS_DIR, 'dt_tuning.csv')}")

    save_result(
        "Decision Tree", val_acc, val_f1, y_test, test_pred, class_names, train_secs,
        notes=f"{X_train.shape[1]} features, {best_params['criterion']}, max_depth={best_params['max_depth']}, "
              f"min_samples_leaf={best_params['min_samples_leaf']}",
    )
