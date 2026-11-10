"""
R2 K-means clustering and R1 objective "Clusters vs. genre".

Runs K-means clustering on the scaled FMA-small training audio features (top 150
features from EDA-2 as well as all 518 features for comparison) and evaluates:
    1. Internal cluster quality metrics (no genre labels used):
       - Inertia (within-cluster sum of squares)
       - Silhouette Score (-1 to +1, higher = tighter, better-separated clusters)
       - Davies-Bouldin Index (lower = better separation)
    2. External metrics against the true 8 genre labels:
       - Adjusted Rand Index (ARI: 0 = random, 1 = identical to genres)
       - Normalized Mutual Information (NMI: 0 = independent, 1 = identical)
       - Homogeneity & Completeness
       - Cluster Purity (% of songs matching their cluster's majority genre, vs. 12.5% chance)

Saves a 2-panel visual to:
    figures/eda_kmeans_clusters.png
    - Left panel:  Internal (Silhouette) and External (ARI, NMI, Purity) scores across k = 2..16
    - Right panel: 8x8 contingency heatmap of K-means clusters (k = 8) vs. true genres

Short Answer — Do the clusters line up with genres?
    Only partially (weakly):
    - At k = 8 (on the top 150 scaled features), external alignment with the 8 genres is well above
      random guessing (Purity = 32.6% vs. 12.5% chance, ARI = 0.101 vs. ~0.000 random, NMI = 0.159),
      and using the top 150 features from EDA-2 improves clustering over all 518 features
      (Purity 32.6% vs. 26.7%, ARI 0.101 vs. 0.067, Silhouette 0.058 vs. 0.023).
    - However, the internal Silhouette score is low (peaking at 0.160 at k = 2 and dropping to 0.058
      at k = 8), showing the songs form one continuous cloud rather than 8 separate clusters.
    - Looking at the k = 8 contingency table, K-means groups tracks by broad acoustic texture rather
      than human genre boundaries:
      * Cluster 0 (46% Folk, 23% Instrumental) and Cluster 2 (46% Instrumental, 19% Folk, 19% Experimental)
        group quiet, acoustic tracks together.
      * Cluster 5 (39% Electronic, 34% Hip-Hop) groups loud, bass-heavy synthetic/percussive tracks together.
      * Cluster 3 captures distorted guitar/loud band tracks (45% Rock).
      * Clusters 1, 4, and 7 are mixed bags where Pop, International, and Experimental spread across
        multiple clusters.
    - Because unsupervised Euclidean distance weights all features equally without knowing which ones
      define human genres, supervised classifiers (R3 baselines, MLP, CNN) are needed.

Run from the repo root (F21DL-main/):
    python scripts/data_prep/kmeans_clustering.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# Make "scripts" importable no matter where this file is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import get_label_names, load_split  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
OUT_PLOT = os.path.join(FIGURES_DIR, "eda_kmeans_clusters.png")

RANDOM_STATE = 42
K_VALUES = [2, 3, 4, 5, 6, 8, 10, 12, 16]


def cluster_purity(y_true, cluster_labels):
    """Fraction of samples belonging to the majority true class in each cluster."""
    ct = pd.crosstab(cluster_labels, y_true)
    return float(ct.max(axis=1).sum() / len(y_true))


# AI help [K-means]
# Prompt: Compute internal metrics (inertia, silhouette, davies_bouldin) and external metrics against genre labels (ARI, NMI, homogeneity, completeness, purity) for a fitted KMeans model
def evaluate_clustering(X, y_true, km):
    """Return a dict of internal and external clustering metrics."""
    labels = km.labels_
    return {
        "inertia": float(km.inertia_),
        "silhouette": float(silhouette_score(X, labels, random_state=RANDOM_STATE)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "NMI": float(normalized_mutual_info_score(y_true, labels)),
        "homogeneity": float(homogeneity_score(y_true, labels)),
        "completeness": float(completeness_score(y_true, labels)),
        "purity": cluster_purity(y_true, labels),
    }


# AI help [K-means]
# Prompt: Run KMeans across K_VALUES on X_train and collect internal and external evaluation metrics into a DataFrame
def sweep_k_values(X_train, y_train):
    """Fit K-means for each k in K_VALUES and return metrics DataFrame and the k=8 model."""
    rows = []
    km_k8 = None

    for k in K_VALUES:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        km.fit(X_train)
        metrics = evaluate_clustering(X_train, y_train, km)
        rows.append({"k": k, **metrics})
        print(
            f"  k={k:2d}  inertia={metrics['inertia']:9.0f}  "
            f"silhouette={metrics['silhouette']:.4f}  DBI={metrics['davies_bouldin']:.3f}  "
            f"ARI={metrics['ARI']:.4f}  NMI={metrics['NMI']:.4f}  purity={metrics['purity']:.4f}"
        )
        if k == 8:
            km_k8 = km

    return pd.DataFrame(rows), km_k8


# AI help [K-means]

# Prompt: Plot a 2-panel figure showing clustering metrics across k on the left and the k=8 cluster vs genre contingency heatmap on the right, saving to figures/eda_kmeans_clusters.png
def plot_kmeans_vs_genres(sweep_df, cluster_labels_k8, y_train, class_names, out_path):
    """Save a 2-panel plot: metrics across k (left) and k=8 cluster vs. genre heatmap (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Left panel: Internal and external metrics across k
    ax = axes[0]
    ax.plot(sweep_df["k"], sweep_df["purity"], marker="o", linewidth=2, label="Cluster Purity (vs. 0.125 random)")
    ax.plot(sweep_df["k"], sweep_df["NMI"], marker="s", linewidth=2, label="NMI vs. Genre")
    ax.plot(sweep_df["k"], sweep_df["ARI"], marker="^", linewidth=2, label="ARI vs. Genre")
    ax.plot(sweep_df["k"], sweep_df["silhouette"], marker="d", linestyle="--", linewidth=2,
            label="Silhouette Score (Internal)")

    ax.axvline(8, color="black", linestyle=":", linewidth=1.8, label="k = 8 (number of genres)")
    ax.axhline(1.0 / len(class_names), color="grey", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Random-guess purity (12.5%)")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("K-Means Internal & External Metrics Across k", fontsize=12, fontweight="bold")
    ax.set_xticks(K_VALUES)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)

    # Right panel: k=8 Cluster x Genre contingency table (normalised by cluster size, annotated with counts)
    ax2 = axes[1]
    genre_series = pd.Series([class_names[i] for i in y_train], name="True Genre")
    cluster_series = pd.Series(cluster_labels_k8, name="Cluster")
    ct = pd.crosstab(cluster_series, genre_series)[class_names]

    # Sort clusters by their dominant genre so the diagonal structure is easier to see
    dominant_genre_idx = np.argmax(ct.values, axis=1)
    order = np.argsort(dominant_genre_idx)
    ct = ct.iloc[order]

    row_shares = ct.div(ct.sum(axis=1), axis=0)
    im = ax2.imshow(row_shares.values, cmap="Blues", aspect="auto", vmin=0, vmax=row_shares.values.max())

    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    y_labels = [
        f"Cluster {cid} (n={int(ct.loc[cid].sum())}, top: {ct.loc[cid].idxmax()[:5]})"
        for cid in ct.index
    ]
    ax2.set_yticks(range(len(ct)))
    ax2.set_yticklabels(y_labels, fontsize=9)

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            count = int(ct.values[i, j])
            share = row_shares.values[i, j]
            text_color = "white" if share > 0.55 * row_shares.values.max() else "black"
            ax2.text(j, i, f"{count}\n({share * 100:.0f}%)", ha="center", va="center",
                     color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("Share of Cluster", fontsize=10)
    ax2.set_title("K-Means (k = 8) Clusters vs. True Genres (Count & Row %)",
                  fontsize=12, fontweight="bold")

    fig.suptitle("R2 Clustering: Do Unsupervised K-Means Clusters Line Up With Genres?",
                 fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")
    return ct


if __name__ == "__main__":
    class_names = get_label_names()

    # Load scaled training data with the chosen top-150 feature set from EDA-2
    X_train_150, _, _, y_train, _, _ = load_split(scale=True, n_features=True)
    # Also load all 518 scaled features to compare k=8 on 150 vs 518 features
    X_train_518, _, _, _, _, _ = load_split(scale=True, n_features=None)

    print("1. Sweeping k = 2..16 on top 150 scaled features (X_train):")
    sweep_df, km_k8 = sweep_k_values(X_train_150, y_train)

    print("\n2. Comparing k = 8 on Top-150 features vs. All-518 features:")
    km_518 = KMeans(n_clusters=8, n_init=20, random_state=RANDOM_STATE).fit(X_train_518)
    m_150 = evaluate_clustering(X_train_150, y_train, km_k8)
    m_518 = evaluate_clustering(X_train_518, y_train, km_518)
    comp_df = pd.DataFrame([
        {"feature_set": "Top 150 (EDA-2)", **m_150},
        {"feature_set": "All 518", **m_518},
    ])
    print(comp_df.round(4).to_string(index=False))

    # Reference against shuffled random labels to show k=8 is well above chance
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_y = rng.permutation(y_train)
    print(
        f"\nRandom-label baseline (shuffled genres): "
        f"ARI={adjusted_rand_score(shuffled_y, km_k8.labels_):.4f}  "
        f"NMI={normalized_mutual_info_score(shuffled_y, km_k8.labels_):.4f}  "
        f"purity={cluster_purity(shuffled_y, km_k8.labels_):.4f}"
    )

    ct = plot_kmeans_vs_genres(sweep_df, km_k8.labels_, y_train, class_names, OUT_PLOT)
    print("\n3. Contingency Table (k = 8 Clusters vs. True Genres on Training Set):")
    print(ct.to_string())
