# F21DL: Music Genre Classification on FMA-small

We classify 30-second music clips from the [FMA-small](https://github.com/mdeff/fma) dataset into 8 genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock). The genres are balanced, so chance level is 12.5%.

We compare three classical baselines (Decision Tree, Naive Bayes, Random Forest) trained on the 518 pre-computed audio features with two neural networks: an MLP on the same features and a CNN on log-mel spectrograms.

**About the codes in the headings:** R1–R4 are the coursework requirements. DATA-, EDA-, BASE- and NN- codes are our GitHub issues.

## Contents
1. [Results](#results)
2. [Setup](#setup)
3. [Getting the data](#getting-the-data)
4. [Running the scripts](#running-the-scripts)
5. [Troubleshooting](#troubleshooting)
6. [Project structure](#project-structure)
7. Details:
   [Data pipeline](#data-pipeline-data-2-data-3) ·
   [Train/val/test split](#data-4-cleaning-and-fixed-trainvaltest-split) ·
   [EDA-1](#eda-1-exploratory-plots) ·
   [EDA-2](#eda-2-feature-selection) ·
   [K-means](#eda-3-k-means-clustering-r2) ·
   [Shared evaluation](#base-1-shared-evaluation) ·
   [Baselines](#r3-baselines) ·
   [MLP](#nn-1-mlp-on-the-518-features) ·
   [CNN](#nn-2-cnn-on-the-spectrograms) ·
   [Augmentation](#nn-3-cnn-data-augmentation)

---

# Results

Every model uses the same train/validation/test split ([`load_split()`](scripts/data_prep/load_split.py)). Settings were chosen on the **validation** set, so validation scores are optimistic. **Test is the number to report.**

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 | Setting |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FMA paper baseline (SVM on MFCCs) | – | – | 46.4% | – | Reference to beat |
| Decision Tree | 42.3% | 0.418 | 36.9% | 0.363 | 150 features, depth 10, min 20 songs per leaf |
| Naive Bayes | 49.9% | 0.481 | 41.8% | 0.399 | 300 features, unscaled |
| Random Forest | 57.1% | 0.555 | 47.3% | 0.471 | 518 features, 200 trees, depth 20 |
| MLP on 518 features | 55.5% | 0.547 | 45.3% | 0.453 | 1 hidden layer of 256 |
| CNN on spectrograms (run 1, NN-2) | 59.4% | 0.578 | 53.75% | 0.529 | lr 0.001, dropout 0.5 |
| CNN on spectrograms (run 2, NN-3 control) | 60.1% | 0.574 | 44.5% | 0.438 | Same setting as run 1 |
| CNN + `spec_mask` augmentation | 58.6% | 0.563 | 48.25% | 0.485 | Best augmentation on validation |

**Main points**
- **The CNN is the best model, but its exact score is uncertain.** The same CNN setting scored 53.75% on test in one run and 44.5% in another, because GPU training is not exactly repeatable. Treat the CNN as roughly 45–54% on test. See [NN-3](#nn-3-cnn-data-augmentation).
- **Random Forest is the best classical model** (47.3%), just above the FMA paper's 46.4%. The MLP on the same features is slightly behind it (45.3%).
- **Every model finds the same genres hard:** Folk, Experimental and Pop. Hip-Hop and Rock are the easiest.
- **Test is 5–10 points below validation for every model**, because the setting was chosen on validation.

The three baseline rows come from [results/results_summary.csv](results/results_summary.csv), which the [shared evaluation](#base-1-shared-evaluation) writes. The MLP and CNN rows are copied by hand from their sections below until those scripts call `save_result()` as well.


# Setup

Requires **Python 3.12 or newer** (`python --version` to check).

1. From the repo root, create and activate a virtual environment:
   ```bash
   python -m venv .venv

   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Windows (cmd)
   .venv\Scripts\activate.bat
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**VS Code:** if it says `No module named 'pandas'` or `'sklearn'`, it is using a different Python. Press `Ctrl+Shift+P` → **Python: Select Interpreter** and pick the `.venv` one.

# Getting the data

**You don't need to do anything here.** The processed data every script uses is already in the repo: `data/processed/fma_small_merged.csv` and one spectrogram per track in `data/processed/spectrograms/`. After cloning you can go straight to [Running the scripts](#running-the-scripts).

The steps below are only for **recreating the dataset from scratch** (about 8 GB to download, and slow to build).

1. Download [`fma_metadata.zip`](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) and unzip it into `data/`. We only use `features.csv` and `tracks.csv` from it.
2. Download [`fma_small.zip`](https://os.unil.cloud.switch.ch/fma/fma_small.zip) and unzip it into `data/`. It unzips to a folder called `fma_small`. Rename it to `fma_small_audio`.

   You should end up with:
   ```
   data/
   ├── fma_metadata/
   │   ├── features.csv
   │   ├── tracks.csv
   │   └── ... (other files, not used)
   └── fma_small_audio/
       ├── 000/
       │   ├── 000002.mp3
       │   └── ...
       └── 155/
   ```
3. Build the dataset from the repo root:
   ```bash
   python scripts/data_prep/final_data.py
   ```
   This removes the bad audio files, writes `data/processed/fma_small_merged.csv`, and saves one spectrogram per track in `data/processed/spectrograms/`. See [Data pipeline](#data-pipeline-data-2-data-3).

# Running the scripts

Run everything from the **repo root** (the folder this README is in), or open a file in VS Code and click **▶ Run Python File**. Model scripts call `load_split()` themselves, so nothing needs to be run beforehand.

| What | Command | Time |
| :--- | :--- | :--- |
| Build the dataset | `python scripts/data_prep/final_data.py` | Slow (reads every audio file) |
| Check the split | `python scripts/data_prep/load_split.py` | Seconds. Prints checks, saves nothing |
| EDA plots | `python scripts/data_prep/eda.py` | |
| Feature selection | `python scripts/data_prep/feature_selection.py` | |
| K-means clustering | `python scripts/data_prep/kmeans_clustering.py` | |
| Decision Tree | `python scripts/models/decision_tree.py` | ~75 s |
| Naive Bayes | `python scripts/models/naive_bayes.py` | ~5 s |
| Random Forest | `python scripts/models/random_forest.py` | ~30 s |
| MLP | `python scripts/models/mlp.py` | Under 1 min on a laptop CPU |
| CNN | `python scripts/models/cnn.py` | ~26 min on an M4 Max |
| CNN augmentation | `python scripts/models/cnn_augment.py` | ~30 min on an M4 Max |

Both CNN scripts accept `--quick` (1 setting, 2 epochs) to check they run before starting the full grid.

# Troubleshooting

| Error | Fix |
| :--- | :--- |
| `ModuleNotFoundError: pandas` / `sklearn` / `torch` | `pip install -r requirements.txt`, or select the right interpreter (see [Setup](#setup)) |
| `ModuleNotFoundError: scripts` | You are not in the repo root. `cd` into the folder this README is in |
| `FileNotFoundError: fma_small_merged.csv` | Run `python scripts/data_prep/final_data.py` first |
| CNN can't find spectrograms | Same: `final_data.py` creates `data/processed/spectrograms/` |

# Project structure
```
├── data/
│   ├── fma_metadata/            # features.csv, tracks.csv (not pushed, only needed to rebuild)
│   ├── fma_small_audio/         # MP3 clips (not pushed, only needed to rebuild)
│   └── processed/               # pushed, ready to use
│       ├── fma_small_merged.csv # merged features + metadata
│       └── spectrograms/        # one log-mel spectrogram per track (<track_id>.npy)
├── figures/                     # every plot the scripts save
├── results/                     # tuning CSVs + results_summary.csv (one row per model)
├── scripts/
│   ├── data_prep/
│   │   ├── final_data.py        # runs the whole data pipeline
│   │   ├── valid_audio.py       # finds and removes short / corrupted audio files
│   │   ├── merge.py             # merge helpers used by final_data.py
│   │   ├── spectogram.py        # makes the log-mel spectrograms
│   │   ├── load_split.py        # shared cleaning + train/val/test split
│   │   ├── eda.py               # EDA-1 plots
│   │   ├── feature_selection.py # EDA-2 feature selection
│   │   └── kmeans_clustering.py # EDA-3 K-means clustering
│   └── models/
│       ├── evaluate.py          # shared evaluation + results table
│       ├── decision_tree.py
│       ├── naive_bayes.py
│       ├── random_forest.py
│       ├── mlp.py               # NN-1
│       ├── cnn.py               # NN-2
│       └── cnn_augment.py       # NN-3
├── requirements.txt
└── README.md
```
The empty `__init__.py` files in `scripts/`, `scripts/data_prep/` and `scripts/models/` make them Python packages, so `from scripts.data_prep.load_split import load_split` works.

---

# Data pipeline (DATA-2, DATA-3)

[`final_data.py`](scripts/data_prep/final_data.py) runs these steps in order:

1. **Remove bad audio files.** [`valid_audio.py`](scripts/data_prep/valid_audio.py) checks that every MP3 is 30 s long (±0.5 s) and deletes the ones that are too short or can't be read. On the full download this removes 6 files:
   `098/098565.mp3`, `098/098567.mp3`, `098/098569.mp3`, `099/099134.mp3`, `108/108925.mp3`, `133/133297.mp3`
2. **Merge.** Joins the remaining audio files with `features.csv` (518 audio features) and `tracks.csv` (metadata, including the genre and FMA's train/val/test split) on `track_id`, and saves `data/processed/fma_small_merged.csv`.
3. **Spectrograms.** [`spectogram.py`](scripts/data_prep/spectogram.py) turns each clip into a log-mel spectrogram (22,050 Hz, 128 mel bins, `n_fft` 2048, hop 512, padded or trimmed to 30 s) with shape `(128, 1292)`, saved as `data/processed/spectrograms/<track_id>.npy`. Tracks that already have a file are skipped, so rerunning is fast.

# DATA-4: Cleaning and fixed train/val/test split

Every model **must** use `load_split()` from [scripts/data_prep/load_split.py](scripts/data_prep/load_split.py), so all results are comparable.

## Why
- Every model is trained and tested on the **same songs** with the **same features**, otherwise results can't be compared.
- It uses FMA's own split, which keeps songs by the same artist out of both train and test, so the score shows how well the model generalises.
- Scaling and filling missing values only use **training** statistics, so no test information leaks into training.

## How to use
```python
from scripts.data_prep.load_split import load_split, get_label_names

X_train, X_val, X_test, y_train, y_val, y_test = load_split()
class_names = get_label_names()   # class_names[i] is the genre for label i

# CNN: also get the track_ids, so the spectrograms use the same split
(X_train, X_val, X_test, y_train, y_val, y_test,
 ids_train, ids_val, ids_test) = load_split(return_ids=True)
```
- `scale=True` (default): a StandardScaler fitted on **train only**, then applied to val/test. Needed for the MLP and K-means. Tree models give the same results with `scale=False`. Naive Bayes is affected by scaling (see [Naive Bayes](#naive-bayes)).
- `n_features`: keep only the top-k features ranked by the [EDA-2](#eda-2-feature-selection) ANOVA F-test, fitted on `X_train` only. `None` keeps all 518, `True` uses the EDA-2 choice of 150.

To use it in a new script in `scripts/models/`, add this at the top so it runs from anywhere:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import load_split, get_label_names
```

## What `python scripts/data_prep/load_split.py` should show
- An 80 / 10 / 10 train / validation / test split, with roughly equal songs per genre. Training has slightly fewer than 6,400 songs because of the 6 removed files.
- 518 audio features per song
- No missing values in any split
- Training features with mean ≈ 0 and standard deviation ≈ 1 (when `scale=True`)
- No track ID in more than one split
- The 8 genres mapped to labels 0–7 in alphabetical order

## Missing values
- Rows missing the genre (`track_genre_top`) or the split (`set_split`) are dropped.
- Infinite or missing feature values are filled with the **training** median.

## Dropped columns
All 52 metadata columns from `tracks.csv` are dropped except `track_genre_top` (the target) and `set_split` (used for splitting). `track_id` is not a feature but is returned with `return_ids=True`.

| Group | Columns | Why dropped |
| :--- | :--- | :--- |
| Target leakage | `track_genres`, `track_genres_all`, `track_tags`, `album_tags`, `artist_tags` | Contain the genre directly or through user tags |
| Mostly empty (>60% missing) | `track_lyricist`, `track_publisher`, `track_information`, `track_composer`, `artist_active_year_end`, `artist_wikipedia_page`, `track_date_recorded`, `artist_related_projects`, `artist_associated_labels`, `track_language_code`, `album_engineer`, `album_producer`, `artist_active_year_begin`, `artist_latitude`, `artist_longitude` | 63.7%–99.7% missing, too sparse to fill in reliably |
| IDs / free text / URLs | `album_id`, `artist_id`, `track_title`, `album_title`, `artist_name`, `artist_bio`, `album_information`, `artist_website`, `artist_location`, `artist_members`, `track_license` | No acoustic meaning, and the model could memorise artists |
| Popularity stats, dates, constants | `album_comments`, `album_favorites`, `album_listens`, `album_tracks`, `album_type`, `album_date_created`, `album_date_released`, `artist_comments`, `artist_favorites`, `artist_date_created`, `track_bit_rate`, `track_comments`, `track_date_created`, `track_duration`, `track_favorites`, `track_interest`, `track_listens`, `track_number`, `set_subset` | Not audio content (`set_subset` is always "small") |

# EDA-1: Exploratory plots

Script: [scripts/data_prep/eda.py](scripts/data_prep/eda.py). Uses the training split (6,394 tracks, ~800 per genre) and saves 4 plots to `figures/`:
- `eda_feature_boxplots.png`: 6 audio features across the 8 genres
- `eda_genre_feature_heatmap.png`: average z-score per genre for each of the 11 feature groups in `features.csv`
- `eda_pca_scatter.png`: 2D PCA of all 518 scaled features
- `eda_spectrograms_per_genre.png`: one example log-mel spectrogram per genre

## What the plots show
- **Instrumental and Folk are the easiest to tell apart from the loud genres.** They have the lowest spectral centroid (659 Hz and 850 Hz), RMS energy (2.75 and 3.48) and zero-crossing rate (0.029 and 0.033), but the highest spectral contrast (~20.1). Electronic, Hip-Hop and Rock are at the other end: brighter (1265–1428 Hz), louder (5.25–5.56) and with a higher zero-crossing rate (0.050–0.058).
- **Some genres look very similar on average.** In the heatmap, Electronic and Hip-Hop have almost the same profile (+0.38 to +0.62 on spectral centroid, bandwidth, rolloff, RMS and `chroma_stft`), and Instrumental and Folk pair up on the negative side (−0.35 to −0.88). Models that only see averages will likely mix these pairs up.
- **Pop and Experimental have no clear profile.** Pop is close to 0 on every feature group (−0.11 to +0.12), so it looks like the average of the whole dataset. Experimental has the widest boxes on almost every feature because it is a catch-all genre.
- **The genres overlap heavily in 2D PCA.** PC1 (11.6%) and PC2 (7.3%) only explain 18.9% of the variance, and all 8 genres form one cloud. The genre medians still pull apart in the expected directions (Folk/Instrumental on one side, Electronic/Hip-Hop and Rock on the other), which suggests non-linear models like Random Forest or an MLP should beat linear ones.
- **Spectrograms show timing patterns that summary features miss.** Hip-Hop and Pop have regular vertical lines from steady drum beats, Electronic has repeating bass bands at the bottom (mel bins 0–35), Rock builds into a loud wall of sound around 10 s, and Instrumental/Folk have quiet upper frequencies with clear gaps between notes. This is why we also train a CNN on the spectrograms.

# EDA-2: Feature selection

Script: [scripts/data_prep/feature_selection.py](scripts/data_prep/feature_selection.py). Saves `figures/eda_feature_selection.png`.

Ranks the 518 features on the training split with an ANOVA F-test and with Random Forest importance, then scores the top-k features (k = 10 to 518) on validation with the three baselines.

- **Random Forest barely needs all 518 features.** It levels off around 150–200 features, about 1 point below using all 518.
- **Naive Bayes and Decision Tree do better with fewer features**, because many features are near-copies of each other.
- **Chosen setting: the top 150 features by ANOVA F-test.** This is what `load_split(n_features=True)` uses. Most of the 150 are MFCC, `chroma_stft` and spectral contrast features, and the other two chroma sets are mostly dropped as redundant.
- **Open question:** this script reports Naive Bayes at about 35% with all 518 features, but [Naive Bayes](#naive-bayes) gets 44–48% with 518 features depending on scaling. The two should be checked against each other before the report.

# EDA-3: K-means clustering (R2)

Script: [scripts/data_prep/kmeans_clustering.py](scripts/data_prep/kmeans_clustering.py). Saves `figures/eda_kmeans_clusters.png`.

**Do the clusters match the genres? Only weakly.**
- With k = 8 on the top 150 features, purity is 32.6% (chance is 12.5%), ARI 0.101 and NMI 0.159. The 150 features cluster better than all 518 (purity 26.7%).
- The silhouette score is low (0.058 at k = 8), so the songs form one continuous cloud rather than 8 separate groups.
- K-means groups songs by overall sound rather than genre: one cluster is quiet acoustic songs (mostly Folk and Instrumental), one is loud bass-heavy songs (Electronic and Hip-Hop), one is mostly Rock, and Pop, International and Experimental are spread across several clusters.
- So we need supervised models to predict genre.

# BASE-1: Shared evaluation

Script: [scripts/models/evaluate.py](scripts/models/evaluate.py)

Every model is scored by the same code, and all results end up in one table, [results/results_summary.csv](results/results_summary.csv). It has one row per model with validation and test accuracy and macro-F1, the test F1 of every genre, training time and the chosen setting, sorted by test macro-F1.

| Function | What it does |
| :--- | :--- |
| `evaluate(model, X, y)` | Predicts on `X` and returns accuracy, macro-F1 and the predictions |
| `print_test_report(y_test, pred, class_names)` | Prints per-genre precision / recall / F1 and a text confusion matrix |
| `plot_confusion(y_test, pred, class_names, name, filename)` | Saves the test confusion matrix to `figures/<filename>` |
| `save_result(name, val_acc, val_f1, y_test, pred, class_names, train_secs, notes)` | Adds the model's row to `results_summary.csv`, or replaces it if the model is already there |

```python
from scripts.models.evaluate import evaluate, print_test_report, plot_confusion, save_result

val_acc, val_f1, _ = evaluate(model, X_val, y_val)
test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)
print_test_report(y_test, test_pred, class_names)
plot_confusion(y_test, test_pred, class_names, "MLP", "mlp_confusion_matrix.png")
save_result("MLP", val_acc, val_f1, y_test, test_pred, class_names, train_secs, notes="1 hidden layer of 256")
```
Models without a scikit-learn style `.predict()` (the PyTorch CNN) skip `evaluate()` and pass their own test predictions to the other three functions.

## Metrics
- **Accuracy:** the test set is balanced (100 songs per genre), so accuracy is meaningful and can be compared with the FMA paper (46.4%) and chance (12.5%).
- **Macro-F1:** the F1 averaged over the 8 genres, each weighted equally, so a model can't score well by only getting the easy genres right. Every model is tuned on this.
- **Per-genre F1 and the confusion matrix:** show *which* genres get confused, which we compare with the EDA findings.

# R3 Baselines

All three baselines follow the same steps so they can be compared fairly:
1. Load the shared split with `load_split()`. The number of features is tuned too, using the EDA-2 ranking.
2. Try every setting in a small grid and score it on **validation** by macro-F1. Every setting is saved to `results/<model>_tuning.csv`.
3. Retrain the best setting and evaluate it **once** on test.
4. Print the per-genre report, save a test confusion matrix to `figures/`, and add a row to `results/results_summary.csv`.

Everything uses `random_state=42`, so the results are reproducible.

## Comparison

| Model | Test accuracy | Test macro-F1 | Train time (best setting) |
| :--- | :--- | :--- | :--- |
| Decision Tree | 36.9% | 0.363 | 0.5 s |
| Naive Bayes | 41.8% | 0.399 | 0.01 s |
| **Random Forest** | **47.3%** | **0.471** | 1.3 s |

Test F1 per genre:

| Model | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Decision Tree | 0.47 | 0.13 | 0.26 | 0.47 | 0.39 | 0.40 | 0.31 | 0.47 |
| Naive Bayes | 0.41 | 0.29 | 0.29 | 0.52 | 0.48 | 0.27 | 0.31 | 0.62 |
| Random Forest | 0.54 | 0.32 | 0.26 | 0.63 | 0.47 | 0.52 | 0.37 | 0.65 |

All three find Folk, Experimental and Pop the hardest, which matches the EDA: Pop has no clear profile, Experimental is a catch-all, and Folk overlaps with Instrumental.

## Decision Tree

Script: [scripts/models/decision_tree.py](scripts/models/decision_tree.py). The easiest model to interpret, since its rules can be read directly.

**Tried:** 72 settings ([results/dt_tuning.csv](results/dt_tuning.csv)).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 50, 150, 518 |
| Split criterion | `gini`, `entropy` |
| `max_depth` | 5, 10, 15, no limit |
| `min_samples_leaf` | 1, 5, 20 |

**Best:** 150 features, `gini`, `max_depth=10`, `min_samples_leaf=20` (192 leaves). Validation 42.3% / 0.418, **test 36.9% / 0.363**.

Figures: `figures/dt_confusion_matrix.png` and `figures/dt_tree_top_levels.png` (first two levels of the tree). The script also prints the first three levels as text rules.

**What we noticed**
- **The weakest baseline**, well below the FMA paper's 46.4%.
- **An unlimited tree overfits.** With no depth limit and 1 song per leaf it grows to depth 29 with 1,970 leaves and drops to 0.359 validation macro-F1.
- **Fewer features helped a little:** 150 and 50 features (0.418 / 0.417) beat all 518 (0.401).
- **The first split is `spectral_rolloff_mean_01`** (how much high-frequency energy a song has). The low side is mostly Instrumental and Folk, and the high side splits further towards Rock and Hip-Hop. This is the same quiet-vs-loud divide the EDA shows.
- **Practical use:** it trains in under a second, needs no scaling, and gives rules people can check (e.g. "low spectral rolloff and high MFCC 4 → Folk"). But a single tree is unstable and overfits easily, which is why Random Forest does about 10 points better.

## Naive Bayes

Script: [scripts/models/naive_bayes.py](scripts/models/naive_bayes.py). A simple probabilistic (generative) model that assumes every feature is independent given the genre.

**Tried:** 80 settings of Gaussian Naive Bayes ([results/nb_tuning.csv](results/nb_tuning.csv)).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 10, 25, 50, 100, 150, 200, 300, 518 |
| `var_smoothing` | `1e-9`, `1e-7`, `1e-5`, `1e-3`, `1e-1` |
| Features scaled? | no, yes |

**Best:** 300 features, unscaled, `var_smoothing=1e-9`. Validation 49.9% / 0.481, **test 41.8% / 0.399**.

Figures: `figures/nb_confusion_matrix.png` and `figures/nb_feature_count.png` (validation macro-F1 against number of features, scaled vs. unscaled).

**What we noticed**
- **The number of features matters most.** Validation macro-F1 rises from 0.35 with 10 features to 0.48 at 200–300, then falls slightly at 518. Many features are near-copies (e.g. mean and median of the same signal), so Naive Bayes counts the same evidence several times.
- **Scaling changes the result, which it normally wouldn't for Naive Bayes.** `var_smoothing` adds a fraction of the *largest* feature variance to every feature. Unscaled, that variance is huge, so it acts as strong smoothing. Scaled, every variance is about 1, so the smoothing is tiny. With all 518 features, unscaled scores 0.476 and scaled 0.443.
- **More smoothing hurt:** `1e-9` and `1e-7` were best, and `1e-1` dropped to 0.25.
- **It predicts Hip-Hop too often** (Electronic and Pop songs get labelled Hip-Hop).

## Random Forest

Script: [scripts/models/random_forest.py](scripts/models/random_forest.py). An ensemble of decision trees.

**Tried:** 16 settings, all with `max_features="sqrt"` ([results/rf_tuning.csv](results/rf_tuning.csv)).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 150, 518 |
| `n_estimators` (trees) | 200, 500 |
| `max_depth` | 20, no limit |
| `min_samples_leaf` | 1, 3 |

**Best:** all 518 features, 200 trees, `max_depth=20`, `min_samples_leaf=1`. Validation 57.1% / 0.555, **test 47.3% / 0.471**.

Figures: `figures/rf_confusion_matrix.png` and `figures/rf_feature_importance.png` (top 20 features).

**What we noticed**
- **The best baseline**, just above the FMA paper's 46.4% and the MLP's 45.3%.
- **About 10 points better than a single tree**, because averaging many trees trained on random samples of songs and features removes most of the overfitting.
- **The settings hardly matter:** all 16 are between 0.530 and 0.555 validation macro-F1.
- **No single feature dominates:** the top feature (`mfcc_mean_03`) has only 0.7% of the total importance. The top features are the same kinds the Decision Tree uses: low MFCCs, spectral rolloff, contrast and centroid.
- **Common mistakes:** Instrumental predicted as Folk, and Pop as Electronic.

# NN-1: MLP on the 518 features

Script: [scripts/models/mlp.py](scripts/models/mlp.py). Uses scikit-learn's `MLPClassifier` with `load_split(scale=True)`, since an MLP needs scaled inputs.

**Tried:** 12 settings, scored on validation by macro-F1 ([results/mlp_tuning.csv](results/mlp_tuning.csv)).

| Setting | Values tried |
| :--- | :--- |
| Hidden layers | `(256,)`, `(512, 256)`, `(256, 128, 64)` |
| L2 regularisation (`alpha`) | `0.0001`, `0.01` |
| Learning rate | `0.001`, `0.0003` |

Fixed for every run: ReLU, Adam, batch size 128, up to 200 epochs, `random_state=42`. Early stopping holds out 10% of the *training* data, so the validation set is only used to choose between settings.

**Best:** one hidden layer of 256, `alpha=0.01`, learning rate `0.0003`. Validation 55.5% / 0.547, **test 45.3% / 0.453**. Training curves: `figures/mlp_training_curve.png`.

**What we noticed**
- **Bigger was not better.** The deeper networks scored about 50–53% on validation against about 55% for one hidden layer. With only ~6,400 training songs they most likely overfit.
- **The lower learning rate won** in every pairing of the other settings.
- **Easy genres:** Hip-Hop (F1 0.70), Rock (0.62), Electronic (0.55). **Hardest:** Folk (0.16), confused with Experimental, International and Pop. Instrumental is often predicted as Folk.

# NN-2: CNN on the spectrograms

Script: [scripts/models/cnn.py](scripts/models/cnn.py). Needs the spectrograms in `data/processed/spectrograms/` and `torch`.

Uses the same split as every other model: the track ids from `load_split(return_ids=True)` are matched to their `(128, 1292)` spectrograms. Spectrograms are normalised with the mean and std of the **training** set only. Loading them all takes about 2.6 GB of RAM.

**Model:** 4 blocks of 3×3 conv → batch norm → ReLU → max-pool (16, 32, 64, 128 channels), then global average pooling, dropout and a linear layer to the 8 genres. AdamW, weight decay `0.01`, batch size 32, up to 30 epochs. The epoch with the best validation macro-F1 is kept, and training stops if that doesn't improve for 6 epochs.

**Tried:** 4 settings ([results/cnn_tuning.csv](results/cnn_tuning.csv)), on an Apple M4 Max (PyTorch `mps` backend).

| Learning rate | Dropout | Best epoch | Val accuracy | Val macro-F1 | Train time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.001** | **0.5** | 28 | **59.4%** | **0.578** | 7.9 min |
| 0.0003 | 0.5 | 28 | 58.0% | 0.560 | 8.0 min |
| 0.0003 | 0.3 | 15 | 58.0% | 0.559 | 5.6 min |
| 0.001 | 0.3 | 10 | 57.1% | 0.555 | 4.7 min |

**Best:** learning rate `0.001`, dropout `0.5`. Validation 59.4% / 0.578, **test 53.75% / 0.529** in this run. Training curves: `figures/cnn_training_curve.png`.

**What we noticed**
- **It beat the MLP by about 8.5 points in this run**, which fits the EDA: spectrograms keep timing and texture information that the 518 summary features throw away. But a second run of the same setting scored only 44.5% (see NN-3), so this gap is less certain than it looks.
- **Dropout 0.5 beat 0.3** at both learning rates.
- **Both dropout 0.5 runs peaked at epoch 28 of 30**, so the 30-epoch limit may be holding them back.
- **Validation is noisy:** it moves by several points between epochs, so 1–2 point gaps between settings are not reliable.
- **Easy genres:** Hip-Hop (F1 0.79), Rock (0.69), Electronic (0.61). **Hardest:** Folk (0.28) and Pop (0.39).

# NN-3: CNN data augmentation

Script: [scripts/models/cnn_augment.py](scripts/models/cnn_augment.py)

Retrains the best NN-2 CNN (learning rate `0.001`, dropout `0.5`) with different augmentations. Everything else is the same in every run (split, seed, 40-epoch limit), so the augmentation is the only change. Augmentation is only applied to training batches, never to validation or test. (`cnn.py` got one small change for this: `train()` takes an optional `augment` function.)

| Augmentation | What it does |
| :--- | :--- |
| none (control) | The plain CNN, rerun with the 40-epoch limit |
| `spec_mask` | SpecAugment-style: 2 random frequency bands (up to 20 mel bins) and 2 random time bands (up to 100 frames) set to the training mean, different for every sample |
| `time_shift` | Random circular shift in time of up to ±200 frames (about ±4.6 s) |
| `noise_and_gain` | Gaussian noise (std 0.1) plus a random per-sample loudness offset (std 0.2), in normalised units |

## Results
All runs: [results/cnn_augmentation.csv](results/cnn_augmentation.csv). Validation curves: `figures/cnn_augmentation_curves.png`.

| Augmentation | Best epoch | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| none (control) | 15 | 60.1% | 0.574 | 44.5% | 0.438 |
| **`spec_mask`** | 15 | 58.6% | 0.563 | 48.25% | 0.485 |
| `noise_and_gain` | 25 | 57.4% | 0.560 | – | – |
| `time_shift` | 6 | 53.5% | 0.519 | – | – |

Test was only run for the control and for the best augmentation on validation (`spec_mask`), so the test set was not used to choose between augmentations.

## Did augmentation help?
**We can't show that it did.**
- On validation, no augmentation beat the control. `spec_mask` was about 1 point behind, `noise_and_gain` about 1.4 and `time_shift` about 5.5.
- On test, `spec_mask` was about 3.75 points above the control, the opposite of validation.
- **The run-to-run noise is bigger than these gaps.** The control is the same setting as NN-2, which scored 53.75% on test. Here it scored 44.5%. GPU (`mps`) training is not exactly repeatable even with a fixed seed, the chosen epoch moved from 28 to 15, and the 800-song test set is small. So a few points' difference on test does not show that `spec_mask` helps.
- **`time_shift` probably hurts.** It is the only gap bigger than the noise. Shifting moves the beat and intro/outro around, which may matter for genre.
- **`spec_mask` and `noise_and_gain` look neutral.**

The same noise means the 53.75% NN-2 test score is probably on the lucky side. Treat the CNN test scores as roughly 45–54%, not exact values.

## What would settle it
Repeat each setting with several seeds and compare the averages, and use a longer epoch limit with a learning-rate schedule to reduce the jumps between epochs. We haven't done this yet because each run takes 5–10 minutes.
