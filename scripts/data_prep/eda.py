"""
EDA-1: Exploratory data analysis plots on the FMA-small training split.

Generates and saves 4 figures to `figures/`:
    1. figures/eda_feature_boxplots.png       - Boxplots of 6 key audio features across the 8 genres
    2. figures/eda_genre_feature_heatmap.png  - Mean z-score per genre across the 11 audio feature families
    3. figures/eda_pca_scatter.png            - 2D PCA projection of the 518 audio features coloured by genre
    4. figures/eda_spectrograms_per_genre.png - 2x4 grid of 1 example log-mel spectrogram per genre (from DATA-3)

What the plots show:
    - Instrumental and Folk have the lowest spectral centroid (659 and 850 Hz), RMS energy (2.75 and 3.48),
      and zero-crossing rate (0.029 and 0.033), but the highest spectral contrast (~20.1). Electronic,
      Hip-Hop, and Rock are at the opposite end (higher brightness, loudness, and ZCR).
    - In the heatmap, Electronic and Hip-Hop have very similar positive z-scores (+0.38 to +0.62) on
      spectral and energy features, while Instrumental and Folk are both strongly negative (-0.35 to -0.88).
    - Pop sits close to 0 across all 11 feature groups (-0.11 to +0.12), and Experimental has the widest
      spread in the boxplots.
    - 2D PCA only explains 18.9% of variance (PC1=11.6%, PC2=7.3%) with heavy overlap across all 8 genres,
      though the genre medians still separate acoustic genres from loud/electronic ones.
    - The spectrograms show beat lines (Hip-Hop, Pop), repeating bass bands (Electronic), dynamic build-ups
      (Rock), and quiet gaps (Instrumental, Folk) that get lost when averaging over 30 seconds.

Run from the repo root (F21DL-main/):
    python scripts/data_prep/eda.py
"""

import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Make "scripts" importable no matter where this file is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import load_split, get_label_names  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
SPECTROGRAM_DIR = os.path.join(REPO_ROOT, "data", "processed", "spectrograms")
AUDIO_DIR = os.path.join(REPO_ROOT, "data", "fma_small_audio")

# 6 interpretable / high-importance features across different acoustic families
BOXPLOT_FEATURES = [
    ("spectral_centroid_mean_01", "Spectral Centroid Mean (Hz)"),
    ("spectral_contrast_mean_03", "Spectral Contrast Band 3 Mean"),
    ("zcr_mean_01", "Zero-Crossing Rate Mean"),
    ("rmse_mean_01", "RMS Energy Mean (Loudness)"),
    ("mfcc_mean_01", "MFCC 1 Mean (Timbre / Overall Energy)"),
    ("mfcc_mean_04", "MFCC 4 Mean (Spectral Envelope)"),
]

# The 11 acoustic feature families in features.csv
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


# AI Generated Function
# Prompt: Plot a 2x3 grid of boxplots comparing 6 audio features across the 8 genres on the training set and save to figures/eda_feature_boxplots.png
def plot_feature_boxplots(X_train_raw, y_train, class_names, out_path):
    """Save a 2x3 grid of boxplots for 6 representative audio features by genre."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    palette = plt.cm.tab10(np.linspace(0, 0.8, len(class_names)))

    for ax, (col, title) in zip(axes.flat, BOXPLOT_FEATURES):
        data_by_genre = [X_train_raw.loc[y_train == g, col].values for g in range(len(class_names))]
        bp = ax.boxplot(
            data_by_genre,
            tick_labels=class_names,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Distribution of Key Audio Features Across Genres (Training Set, Outliers Hidden)",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# AI Generated Function
# Prompt: Compute the mean scaled feature value per genre across the 11 FMA feature families and save an annotated heatmap to figures/eda_genre_feature_heatmap.png
def plot_genre_feature_heatmap(X_train_scaled, y_train, class_names, out_path):
    """Save an 8 (genres) x 11 (feature families) heatmap of mean z-scores."""
    # Use the '_mean_' summary statistics within each feature family so every family
    # represents average level rather than mixing skew/kurtosis/min/max
    family_scores = {}
    for fam in FEATURE_FAMILIES:
        cols = [c for c in X_train_scaled.columns if c.startswith(f"{fam}_mean_")]
        family_scores[fam] = X_train_scaled[cols].mean(axis=1)

    fam_df = pd.DataFrame(family_scores, index=X_train_scaled.index)
    genre_means = fam_df.groupby(y_train).mean()
    genre_means.index = [class_names[i] for i in genre_means.index]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    vmax = np.abs(genre_means.values).max()
    im = ax.imshow(genre_means.values, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(FEATURE_FAMILIES)))
    ax.set_xticklabels(FEATURE_FAMILIES, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(genre_means.index, fontsize=10)

    # Annotate each cell with its z-score
    for i in range(genre_means.shape[0]):
        for j in range(genre_means.shape[1]):
            val = genre_means.values[i, j]
            text_color = "white" if abs(val) > 0.45 * vmax else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean Standardised Score (z-score)", fontsize=10)
    ax.set_title("Mean Acoustic Feature Family Profile per Genre (Training Set)",
                 fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return genre_means


# AI Generated Function
# Prompt: Project the 518 scaled training features into 2D using PCA and plot a scatter plot coloured by genre with group centroids to figures/eda_pca_scatter.png
def plot_pca_scatter(X_train_scaled, y_train, class_names, out_path):
    """Save a 2D PCA scatter plot of the training set coloured by genre."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_train_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100

    # Clip extreme 1% outliers on axes so the main cloud is clearly visible
    x_lo, x_hi = np.percentile(coords[:, 0], [0.5, 99.5])
    y_lo, y_hi = np.percentile(coords[:, 1], [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(class_names)))

    for g, (name, color) in enumerate(zip(class_names, colors)):
        mask = (y_train == g).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, color=color,
                   s=12, alpha=0.35, edgecolors="none")
        # Plot genre centroid as a bold marker
        cx, cy = np.median(coords[mask, 0]), np.median(coords[mask, 1])
        ax.scatter(cx, cy, color=color, edgecolors="black", s=140, marker="X", zorder=5)
        ax.annotate(name, (cx, cy), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")

    ax.set_xlim(x_lo - 1, x_hi + 1)
    ax.set_ylim(y_lo - 1, y_hi + 1)
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance explained)", fontsize=11)
    ax.set_title("2D PCA Projection of 518 Audio Features by Genre (X = Genre Median)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Genre", loc="upper right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path} (PC1={var1:.1f}%, PC2={var2:.1f}%)")
    return var1, var2


# AI Generated Function
# Prompt: Load one example log-mel spectrogram per genre from data/processed/spectrograms (or compute from data/fma_small_audio if available) and plot a 2x4 grid to figures/eda_spectrograms_per_genre.png
def plot_example_spectrograms(ids_train, y_train, class_names, out_path):
    """Plot 1 example log-mel spectrogram per genre in a 2x4 grid."""
    examples = {}
    missing_ids = []

    for g, name in enumerate(class_names):
        genre_ids = ids_train[y_train == g].astype(int).tolist()
        chosen_id = None
        spec = None

        # 1. Check if any track for this genre already has a .npy in data/processed/spectrograms/
        for tid in genre_ids:
            npy_path = os.path.join(SPECTROGRAM_DIR, f"{tid}.npy")
            if os.path.exists(npy_path):
                spec = np.load(npy_path).astype(np.float32)
                chosen_id = tid
                break

        # 2. Fallback: compute on the fly from data/fma_small_audio/<folder>/<tid>.mp3 if present
        if spec is None:
            for tid in genre_ids:
                folder = f"{tid:06d}"[:3]
                mp3_path = os.path.join(AUDIO_DIR, folder, f"{tid:06d}.mp3")
                if os.path.exists(mp3_path):
                    y_audio, sr = librosa.load(mp3_path, sr=22050, duration=30.0)
                    y_audio = librosa.util.fix_length(y_audio, size=int(22050 * 30.0))
                    mel = librosa.feature.melspectrogram(
                        y=y_audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
                    )
                    spec = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
                    chosen_id = tid
                    break

        if spec is not None:
            examples[name] = (chosen_id, spec)
        else:
            missing_ids.append((name, genre_ids[0]))

    if missing_ids:
        print("\n[Note] Could not create figures/eda_spectrograms_per_genre.png yet.")
        print("Place these 8 files from DATA-3 into data/processed/spectrograms/ and re-run:")
        for name, tid in missing_ids:
            print(f"  {name:15s}: data/processed/spectrograms/{tid}.npy")
        return False

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True, constrained_layout=True)
    im = None
    for ax, name in zip(axes.flat, class_names):
        tid, spec = examples[name]
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, 30.0, 0, spec.shape[0]], vmin=-80, vmax=0)
        ax.set_title(f"{name} (track {tid})", fontsize=11, fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mel Frequency Bin", fontsize=10)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Power (dB)", fontsize=10)
    fig.suptitle("Example Log-Mel Spectrogram per Genre (30s Clips, 128 Mel Bins)",
                 fontsize=14, fontweight="bold")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load unscaled training split for interpretable raw feature boxplots
    X_train_raw, _, _, y_train, _, _, ids_train, _, _ = load_split(scale=False, return_ids=True)
    # Load scaled training split for heatmap and PCA
    X_train_scaled, _, _, _, _, _ = load_split(scale=True)
    class_names = get_label_names()

    print(f"Loaded training split: {X_train_raw.shape[0]} tracks, {X_train_raw.shape[1]} features")
    print("Genre counts in training split:")
    for g, name in enumerate(class_names):
        print(f"  {name:15s}: {int((y_train == g).sum())}")
    print()

    # 1. Feature boxplots by genre
    plot_feature_boxplots(
        X_train_raw, y_train, class_names,
        os.path.join(FIGURES_DIR, "eda_feature_boxplots.png")
    )

    # 2. Genre x Feature Family heatmap
    genre_means = plot_genre_feature_heatmap(
        X_train_scaled, y_train, class_names,
        os.path.join(FIGURES_DIR, "eda_genre_feature_heatmap.png")
    )

    # 3. 2D PCA scatter plot
    var1, var2 = plot_pca_scatter(
        X_train_scaled, y_train, class_names,
        os.path.join(FIGURES_DIR, "eda_pca_scatter.png")
    )

    # 4. Example spectrogram per genre (from DATA-3)
    plot_example_spectrograms(
        ids_train, y_train, class_names,
        os.path.join(FIGURES_DIR, "eda_spectrograms_per_genre.png")
    )

    # Print key numerical takeaways to help write the README summary
    print("\n--- Summary Statistics (Medians on Raw Training Set) ---")
    raw_medians = X_train_raw[[col for col, _ in BOXPLOT_FEATURES]].groupby(y_train).median()
    raw_medians.index = class_names
    print(raw_medians.round(3).to_string())
