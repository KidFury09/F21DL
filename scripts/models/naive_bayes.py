"""
R3 baseline: Gaussian Naive Bayes on the FMA audio features.

Uses the shared split from load_split() so results are comparable with the
other models. Naive Bayes assumes features are independent given the genre,
which the 518 FMA features break badly (many are correlated, e.g. the three
chroma sets and mean/median/min/max of the same signal). EDA-2 showed it does
much better with fewer features, so the number of top ANOVA features is the
main thing tuned here, along with var_smoothing and whether the features are scaled. Everything is picked on the
validation set; the test set is only used once at the end.

Outputs:
    results/nb_tuning.csv            every setting tried, sorted by val macro-F1
    figures/nb_feature_count.png     validation macro-F1 vs. number of features (scaled vs. unscaled)
    figures/nb_confusion_matrix.png  test confusion matrix

Run from the repo root (F21DL-main/):
    python scripts/models/naive_bayes.py
"""

# AI Generated Code (Claude Code, model Opus 5.5)
# Prompt: please make the code for decision trees and naive bayes
# (context given to the model: follow the structure of scripts/models/random_forest.py,
#  use load_split(), tune the number of features and var_smoothing on validation
#  macro-F1, evaluate on test once, save the tuning results and plots)

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402

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

# Number of top ANOVA features (ranked on X_train only inside load_split). None = all 518.
N_FEATURES_GRID = (10, 25, 50, 100, 150, 200, 300, None)
# var_smoothing adds this fraction of the largest feature variance to every variance,
# which stops near-zero variances from dominating the likelihood.
VAR_SMOOTHING_GRID = (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    class_names = get_label_names()

    # 1. Tune on validation. Scaling is tuned too: GaussianNB's means/variances
    # don't care about units, BUT var_smoothing is a fraction of the LARGEST
    # feature variance. Unscaled, that variance is huge (e.g. spectral centroid),
    # so every feature gets a lot of extra variance = strong smoothing. Scaled,
    # all variances are ~1 and the smoothing is tiny. So the two behave differently.
    rows, best, best_f1 = [], None, -1.0
    print("Tuning on validation set:")
    for scale in (False, True):
        for n_features in N_FEATURES_GRID:
            X_train, X_val, _, y_train, y_val, _ = load_split(scale=scale, n_features=n_features)
            k = X_train.shape[1]
            for vs in VAR_SMOOTHING_GRID:
                start = time.time()
                model = GaussianNB(var_smoothing=vs).fit(X_train, y_train)
                secs = time.time() - start
                acc, f1, _ = evaluate(model, X_val, y_val)
                print(f"  scaled={scale!s:5}  k={k:3d}  var_smoothing={vs:.0e}  "
                      f"val acc={acc:.4f}  val macro-F1={f1:.4f}")
                rows.append({"scaled": scale, "n_features": k, "var_smoothing": vs, "val_acc": acc,
                             "val_macro_f1": f1, "train_secs": secs})
                if f1 > best_f1:
                    best, best_f1 = (scale, n_features, vs), f1

    tuning = pd.DataFrame(rows)
    tuning.sort_values("val_macro_f1", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "nb_tuning.csv"), index=False
    )

    best_scale, best_n_features, best_vs = best
    print(f"\nBest: scaled={best_scale}  n_features={best_n_features or 518}  var_smoothing={best_vs:.0e}")

    # 2. Refit with the best setting and evaluate once on test
    X_train, X_val, X_test, y_train, y_val, y_test = load_split(scale=best_scale, n_features=best_n_features)
    start = time.time()
    model = GaussianNB(var_smoothing=best_vs).fit(X_train, y_train)
    train_secs = time.time() - start

    val_acc, val_f1, _ = evaluate(model, X_val, y_val)
    test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)

    print(f"\nTraining time: {train_secs:.3f}s")
    print(f"Validation: accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
    print(f"Test:       accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print("Target to beat (SVM on MFCCs, FMA paper): 46.4% accuracy")

    print_test_report(y_test, test_pred, class_names)

    # 3. Figures
    # Validation macro-F1 vs. number of features (best var_smoothing at each k),
    # one line for unscaled and one for scaled features
    fig, ax = plt.subplots(figsize=(7, 4))
    for scale, group in tuning.groupby("scaled"):
        per_k = group.groupby("n_features")["val_macro_f1"].max()
        ax.plot(per_k.index, per_k.values, marker="o", label="scaled" if scale else "unscaled")
    ax.axvline(X_train.shape[1], color="grey", linestyle="--", label=f"chosen k = {X_train.shape[1]}")
    ax.set(xlabel="Number of top ANOVA features (k)", ylabel="Validation macro-F1",
           title="Gaussian Naive Bayes - effect of feature count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    k_path = os.path.join(FIGURES_DIR, "nb_feature_count.png")
    fig.savefig(k_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved {os.path.join(RESULTS_DIR, 'nb_tuning.csv')}")
    print(f"Saved {k_path}")
    plot_confusion(y_test, test_pred, class_names, "Naive Bayes", "nb_confusion_matrix.png")

    save_result(
        "Naive Bayes", val_acc, val_f1, y_test, test_pred, class_names, train_secs,
        notes=f"{X_train.shape[1]} features, {'scaled' if best_scale else 'unscaled'}, "
              f"var_smoothing={best_vs:.0e}",
    )
