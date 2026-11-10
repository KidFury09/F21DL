"""
EDA-2: Feature selection — how many of the 518 audio features do we actually need?

Ranks all 518 features on the training split ONLY using two methods:
    1. ANOVA F-test (f_classif) — linear between-genre vs. within-genre variance ratio
    2. Random Forest importance (feature_importances_) — non-linear split impurity reduction

Evaluates top-k subsets (k = 10, 25, 50, 75, 100, 150, 200, 300, 400, 518) on the
validation split across our baseline models (Random Forest, Decision Tree, Gaussian Naive Bayes)
and saves the performance curve + feature-group breakdown to:
    figures/eda_feature_selection.png

What the plot shows (key findings):
    - Random Forest plateaus around k = 150–200 features (~55.5% validation accuracy, vs. 56.6%
      with all 518 features), meaning we can drop ~71% of the columns with ~1% accuracy difference.
    - Gaussian Naive Bayes and Decision Tree actually do MUCH better with fewer features:
      Naive Bayes peaks at k = 150–200 (48.5–49.0%) and drops sharply to 35.0% when given all
      518 features because correlated/noisy columns (like kurtosis/min/max and duplicate chroma sets)
      break Naive Bayes' independence assumption. Decision Tree peaks at k = 25–100 (41.8–42.0%)
      and drops to 36.6% at k = 518.
    - Chosen feature set: top k = 150 features by ANOVA F-test (fitted on X_train only), added to
      load_split(n_features=True) / load_split(n_features=150).
    - Feature families in the top 150: mfcc (70 of 140), chroma_stft (29 of 84), and
      spectral_contrast (21 of 49) make up 80% of the selected features, while chroma_cqt (1 of 84)
      and chroma_cens (5 of 84) are mostly dropped as redundant with chroma_stft.

Run from the repo root (F21DL-main/):
    python scripts/data_prep/feature_selection.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Make "scripts" importable no matter where this file is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import DEFAULT_N_FEATURES, load_split  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
OUT_PLOT = os.path.join(FIGURES_DIR, "eda_feature_selection.png")

RANDOM_STATE = 42
K_VALUES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 518]

FEATURE_FAMILIES = [
    "mfcc",
    "chroma_stft",
    "chroma_cqt",
    "chroma_cens",
    "tonnetz",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_rolloff",
    "rmse",
    "zcr",
]


def get_family(col_name):
    """Map a column like 'spectral_contrast_mean_03' to its feature family."""
    for fam in FEATURE_FAMILIES:
        if col_name.startswith(fam + "_"):
            return fam
    return "other"


# AI Generated Function
# Prompt: Rank all 518 features on X_train using both ANOVA F-test and Random Forest feature importances, returning sorted feature lists and a ranking DataFrame
def rank_features(X_train, y_train):
    """Rank features on training split only using ANOVA F-test and Random Forest."""
    f_scores, p_values = f_classif(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_scores = rf.feature_importances_

    ranking_df = pd.DataFrame({
        "feature": X_train.columns,
        "family": [get_family(c) for c in X_train.columns],
        "anova_f": f_scores,
        "anova_p": p_values,
        "rf_importance": rf_scores,
    }).sort_values("anova_f", ascending=False).reset_index(drop=True)

    anova_order = ranking_df["feature"].tolist()
    rf_order = ranking_df.sort_values("rf_importance", ascending=False)["feature"].tolist()
    return ranking_df, anova_order, rf_order


# AI Generated Function
# Prompt: Evaluate validation accuracy across K_VALUES for Random Forest, Decision Tree, and Gaussian Naive Bayes using the ranked feature lists
def evaluate_k_subsets(X_train, X_val, y_train, y_val, anova_order, rf_order):
    """Evaluate validation accuracy for each k in K_VALUES."""
    rows = []
    for k in K_VALUES:
        cols_f = anova_order[:k]
        cols_rf = rf_order[:k]

        # 1. Random Forest (using ANOVA top-k and RF top-k)
        rf_f = RandomForestClassifier(
            n_estimators=200, max_depth=20, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_train[cols_f], y_train)
        acc_rf_anova = accuracy_score(y_val, rf_f.predict(X_val[cols_f]))

        rf_rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_train[cols_rf], y_train)
        acc_rf_rf = accuracy_score(y_val, rf_rf.predict(X_val[cols_rf]))

        # 2. Decision Tree (ANOVA top-k)
        dt = DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ).fit(X_train[cols_f], y_train)
        acc_dt = accuracy_score(y_val, dt.predict(X_val[cols_f]))

        # 3. Gaussian Naive Bayes (ANOVA top-k)
        nb = GaussianNB().fit(X_train[cols_f], y_train)
        acc_nb = accuracy_score(y_val, nb.predict(X_val[cols_f]))

        rows.append({
            "k": k,
            "RF (ANOVA top-k)": acc_rf_anova,
            "RF (RF-imp top-k)": acc_rf_rf,
            "Decision Tree (ANOVA top-k)": acc_dt,
            "Naive Bayes (ANOVA top-k)": acc_nb,
        })
        print(
            f"  k={k:3d}  RF(ANOVA)={acc_rf_anova:.4f}  RF(RF-imp)={acc_rf_rf:.4f}  "
            f"DT={acc_dt:.4f}  NB={acc_nb:.4f}"
        )

    return pd.DataFrame(rows)


# AI Generated Function
# Prompt: Plot validation accuracy vs number of features k and a bar chart of which feature families make up the chosen top-150 subset, saving to figures/eda_feature_selection.png
def plot_feature_selection(curve_df, ranking_df, chosen_k, out_path):
    """Save a 2-panel plot: accuracy vs k (left) and family composition of top-k (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Left panel: Validation accuracy vs. number of features k
    ax = axes[0]
    ax.plot(curve_df["k"], curve_df["RF (ANOVA top-k)"] * 100,
            marker="o", linewidth=2, label="Random Forest (ANOVA top-k)")
    ax.plot(curve_df["k"], curve_df["RF (RF-imp top-k)"] * 100,
            marker="s", linestyle="--", linewidth=1.8, label="Random Forest (RF-importance top-k)")
    ax.plot(curve_df["k"], curve_df["Naive Bayes (ANOVA top-k)"] * 100,
            marker="^", linewidth=2, label="Gaussian Naive Bayes (ANOVA top-k)")
    ax.plot(curve_df["k"], curve_df["Decision Tree (ANOVA top-k)"] * 100,
            marker="d", linewidth=2, label="Decision Tree (ANOVA top-k)")

    ax.axvline(chosen_k, color="black", linestyle=":", linewidth=1.8,
               label=f"Chosen k = {chosen_k}")
    ax.set_xlabel("Number of Selected Features (k)", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.set_title("Validation Accuracy vs. Number of Features", fontsize=12, fontweight="bold")
    ax.set_xticks(K_VALUES)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    # Right panel: How many features from each family are kept in the top chosen_k vs total
    ax2 = axes[1]
    top_sub = ranking_df.iloc[:chosen_k]
    total_counts = ranking_df["family"].value_counts()
    kept_counts = top_sub["family"].value_counts()

    fam_summary = pd.DataFrame({
        "Total in dataset (518)": total_counts,
        f"Kept in top {chosen_k}": kept_counts,
    }).fillna(0).astype(int)
    fam_summary = fam_summary.sort_values(f"Kept in top {chosen_k}", ascending=True)

    y_pos = np.arange(len(fam_summary))
    ax2.barh(y_pos - 0.18, fam_summary["Total in dataset (518)"], height=0.35,
             color="#b0c4de", label="Total in dataset (518)")
    ax2.barh(y_pos + 0.18, fam_summary[f"Kept in top {chosen_k}"], height=0.35,
             color="#2b5c8f", label=f"Selected in top {chosen_k}")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(fam_summary.index, fontsize=10)
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_title(f"Feature Families Kept in Top {chosen_k} (ANOVA F-test)",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle("EDA-2: Feature Selection on FMA-Small Training Split",
                 fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    X_train, X_val, _, y_train, y_val, _ = load_split(scale=True, n_features=None)

    print("1. Ranking all 518 features on X_train only...")
    ranking_df, anova_order, rf_order = rank_features(X_train, y_train)

    print("\nTop 15 features by ANOVA F-statistic:")
    print(ranking_df[["feature", "family", "anova_f", "rf_importance"]].head(15).to_string(index=False))

    print("\n2. Evaluating validation accuracy across feature counts k:")
    curve_df = evaluate_k_subsets(X_train, X_val, y_train, y_val, anova_order, rf_order)

    plot_feature_selection(curve_df, ranking_df, DEFAULT_N_FEATURES, OUT_PLOT)

    # Verify load_split(n_features=True) returns the chosen subset
    X_tr_sel, X_va_sel, X_te_sel, _, _, _ = load_split(scale=True, n_features=True)
    print(
        f"\n3. Shared loader check: load_split(n_features=True) -> "
        f"train={X_tr_sel.shape}, val={X_va_sel.shape}, test={X_te_sel.shape}"
    )
