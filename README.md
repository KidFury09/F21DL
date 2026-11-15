# F21DL

# Plan

## Data Processing Procedures
- [x] Merge tracks.csv, features.csv, genres.csv
- [x] Create the spectrograms from the audio files
- [x] Check for mismatch in files
- [x] Check for corrupted files

## Baseline Models
- [x] Pick three models (Decision Tree, Naive Bayes, Random Forest)
- [x] Decision Tree ([scripts/models/decision_tree.py](scripts/models/decision_tree.py))
- [x] Naive Bayes ([scripts/models/naive_bayes.py](scripts/models/naive_bayes.py))
- [x] Random Forest ([scripts/models/random_forest.py](scripts/models/random_forest.py))
- [x] Shared evaluation and results table ([scripts/models/evaluate.py](scripts/models/evaluate.py))

## Neural Networks
- [ ] Design the neural network
- [x] MLP on the 518 features ([scripts/models/mlp.py](scripts/models/mlp.py))
- [x] CNN on the spectrograms ([scripts/models/cnn.py](scripts/models/cnn.py))
- [x] CNN data augmentation ([scripts/models/cnn_augment.py](scripts/models/cnn_augment.py))
- [ ] Send the MLP and CNN results through the shared evaluation (`save_result()`)


# Results so far

Same split for every model ([load_split()](scripts/data_prep/load_split.py)). Validation is what each setting was tuned on, so **test is the number to report**. Details and tuning notes are in each model's section below. Chance level is 12.5% (8 balanced genres).

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FMA paper baseline (SVM on MFCCs) | – | – | 46.4% | – | Reference to beat |
| Decision Tree (R3) | 42.3% | 0.418 | 36.9% | 0.363 | 150 features, depth 10, min 20 songs per leaf |
| Naive Bayes (R3) | 49.9% | 0.481 | 41.8% | 0.399 | 300 features, unscaled |
| Random Forest (R3) | 57.1% | 0.555 | 47.3% | 0.471 | 518 features, 200 trees, depth 20 |
| MLP on 518 features (NN-1) | 55.5% | 0.547 | 45.3% | 0.453 | 1 hidden layer of 256 |
| CNN on spectrograms (NN-2) | 59.4% | 0.578 | 53.75% | 0.529 | Single run, probably on the lucky side (see below) |
| CNN rerun, no augmentation (NN-3 control) | 60.1% | 0.574 | 44.5% | 0.438 | Same setting as NN-2, different run |
| CNN + `spec_mask` (NN-3) | 58.6% | 0.563 | 48.25% | 0.485 | Best augmentation on validation |
| CNN + `noise_and_gain` (NN-3) | 57.4% | 0.560 | – | – | Test not evaluated |
| CNN + `time_shift` (NN-3) | 53.5% | 0.519 | – | – | Test not evaluated |

**Read the CNN rows with care:** the NN-2 setting scored 53.75% on test in one run and 44.5% in another (the GPU run is not exactly repeatable), so the CNN's real test accuracy is roughly 45-54% and the gaps between the CNN rows are within noise. See the NN-3 section.

The three baseline rows come from [results/results_summary.csv](results/results_summary.csv), which is written by the shared evaluation (see [BASE-1](#base-1-shared-evaluation)). The MLP and CNN rows are copied from their sections below until those scripts call `save_result()` too.

## Project structure
```
F21DL-main/
├── data/
│   ├── fma_metadata/            # features.csv, tracks.csv, genres.csv (not pushed, too big)
│   ├── fma_small_audio/         # MP3 clips (not pushed, too big)
│   └── processed/
│       ├── fma_small_merged.csv # output of final_data.py
│       └── spectrograms/        # one log-mel spectrogram per track (<track_id>.npy)
├── figures/                     # every plot the scripts save
├── results/                     # tuning CSVs + results_summary.csv (one row per model)
├── scripts/
│   ├── __init__.py
│   ├── data_prep/
│   │   ├── __init__.py
│   │   ├── valid_audio.py       # finds corrupted / short audio files
│   │   ├── merge.py             # merges features.csv + tracks.csv for fma_small
│   │   ├── spectogram.py        # makes the log-mel spectrograms
│   │   ├── final_data.py        # runs the whole data pipeline
│   │   ├── load_split.py        # shared cleaning + train/val/test split (DATA-4)
│   │   ├── eda.py               # EDA-1 plots
│   │   ├── feature_selection.py # EDA-2 feature selection
│   │   └── kmeans_clustering.py # R2 K-means clustering
│   └── models/
│       ├── __init__.py
│       ├── evaluate.py          # BASE-1 shared evaluation + results table
│       ├── decision_tree.py     # R3 baseline
│       ├── naive_bayes.py       # R3 baseline
│       ├── random_forest.py     # R3 baseline
│       ├── mlp.py               # NN-1 MLP on the 518 features
│       ├── cnn.py               # NN-2 CNN on the spectrograms
│       └── cnn_augment.py       # NN-3 CNN with data augmentation
├── requirements.txt
└── README.md
```
The empty `__init__.py` files mark `scripts`, `scripts/data_prep` and `scripts/models` as Python packages so `from scripts.data_prep.load_split import load_split` works.

## Setup
```powershell
pip install -r requirements.txt
```
If VS Code says `No module named 'pandas'` / `'sklearn'`, it is using a different Python. Press `Ctrl+Shift+P` → **Python: Select Interpreter** and pick the one you installed into, or run `python -m pip install -r requirements.txt`.

## Running the scripts
Open the VS Code terminal (`` Ctrl+` ``) and move into the project folder first. VS Code usually opens in the parent `F21DML-main` folder:
```powershell
cd C:\Users\tehan\Downloads\F21DML-main\F21DL-main
```

| What | Command | Notes |
| :--- | :--- | :--- |
| Rebuild merged CSV | `python scripts/data_prep/merge.py` | Needs `data/fma_metadata` and `data/fma_small_audio` |
| Check the split | `python scripts/data_prep/load_split.py` | Prints shapes, genre counts, NaNs, scaling, overlap. Saves nothing |
| Decision Tree baseline | `python scripts/models/decision_tree.py` | ~75 seconds |
| Naive Bayes baseline | `python scripts/models/naive_bayes.py` | ~5 seconds |
| Random Forest baseline | `python scripts/models/random_forest.py` | ~30 seconds |

You can also open any of these files and click **▶ Run Python File** in the top right of VS Code.

Model scripts call `load_split()` themselves, so you don't need to run `load_split.py` first.

## Troubleshooting
| Error | Fix |
| :--- | :--- |
| `ModuleNotFoundError: pandas` / `sklearn` | `pip install -r requirements.txt` or select the right interpreter (see Setup) |
| `ModuleNotFoundError: scripts` | Run from the `F21DL-main` folder |
| `FileNotFoundError: fma_small_merged.csv` | Run `python scripts/data_prep/merge.py` first |

# Dataset Processing stage

## Issues
- track.csv too big to push
- feature.csv too big to push
- fma_small_audio too big to push

## Audio Files Not Long Enough
Thes audio files were found to be not long enough and were manually removed from the database
- 098/098565.mp3
- 098/098567.mp3
- 098/098569.mp3
- 099/099134.mp3
- 108/108925.mp3
- 133/133297.mp3

> [!WARNING]
> Ask Proffessor whether to create a script that can do this automatically

## Actual Task Order
- Check for corrupted files
- Merged tracks, features with the audio files (skipped out genre for now)
- Cleaned merged dataset and created fixed train/val/test split (DATA-4)
- Random Forest baseline (R3)
- Decision Tree and Naive Bayes baselines, shared evaluation (R3, BASE-1)

---
---
# Environment Setup

Requires **Python 3.12 or newer**. Check your version with `python --version` (or `python3 --version`).

## Step 1: Create a virtual environment
From the repo root, run:

```bash
python -m venv .venv
```

Then activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

## Step 2: Install the dependencies

```bash
pip install -r requirements.txt
```

# Data Set Preparation Steps

## Step 1: Download the metadata
Download [`fma_metadata.zip`](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) and unzip it into `data/`.

We will only be using `features.csv`, `genres.csv` and `tracks.csv`.

## Step 2: Download the audio
Download [`fma_small.zip`](https://os.unil.cloud.switch.ch/fma/fma_small.zip) and unzip it into `data/`.

It unzips to a folder called `fma_small`. Rename it to `fma_small_audio`.

You should end up with something like this:

```
data/
├── fma_metadata/
│   ├── features.csv
│   ├── genres.csv
│   ├── tracks.csv
│   └── ... (other files, not used)
└── fma_small_audio/
    ├── 000/
    │   ├── 000002.mp3
    │   └── ...
    ├── 001/
    ├── ...
    └── 155/
```

## Step 3: Build the dataset
From the repo root, run:

```bash
python scripts/data_prep/final_data.py
```

The final dataset will be stored in `data/processed/`.

# EDA-1: Exploratory Data Analysis Plots

Script: [scripts/data_prep/eda.py](scripts/data_prep/eda.py)

## How to run
From the `F21DL-main` folder:
```powershell
python scripts/data_prep/eda.py
```
This uses the training split from `load_split()` (6,394 tracks, ~800 per genre) and saves 4 plots into `figures/`:
- `figures/eda_feature_boxplots.png` — boxplots for 6 audio features across the 8 genres
- `figures/eda_genre_feature_heatmap.png` — average z-score per genre for each of the 11 feature groups in `features.csv`
- `figures/eda_pca_scatter.png` — 2D PCA plot of all 518 scaled features
- `figures/eda_spectrograms_per_genre.png` — 1 example log-mel spectrogram per genre (from DATA-3)

## What the plots show
- **Instrumental and Folk are the easiest to tell apart from the loud genres:** In the boxplots, Instrumental and Folk have the lowest spectral centroid (`659 Hz` and `850 Hz`), lowest RMS energy (`2.75` and `3.48`), and lowest zero-crossing rate (`0.029` and `0.033`), but the highest spectral contrast (`~20.1`). Electronic, Hip-Hop, and Rock sit at the opposite end with much higher brightness (`1265–1428 Hz`), loudness (`5.25–5.56`), and zero-crossing rate (`0.050–0.058`).
- **Some genres look very similar on average:** In the heatmap, Electronic and Hip-Hop have almost the same red profile (`+0.38` to `+0.62` on spectral centroid, bandwidth, rolloff, `rmse`, and `chroma_stft`). Instrumental and Folk also pair up on the negative side (`-0.35` to `-0.88`). Models relying only on overall averages will likely mix these pairs up.
- **Pop and Experimental don't have a clear profile:** Pop sits close to `0` across every feature group in the heatmap (`-0.11` to `+0.12`), so it looks like the average of the whole dataset. Experimental has the widest boxes on almost every feature because it is a broad catch-all genre.
- **2D PCA has heavy overlap:** `PC1` (`11.6%`) and `PC2` (`7.3%`) only cover `18.9%` of the variance, and all 8 genres overlap in one big cloud. The genre medians (`X` markers) still pull apart in the expected directions (Folk/Instrumental on the right, Electronic/Hip-Hop top-left, Rock on the left, Pop/International/Experimental in the middle), which suggests non-linear models like Random Forest or MLP should do better than simple linear ones.
- **Spectrograms show timing and beat patterns that summary stats miss:** In the spectrogram grid, Hip-Hop and Pop have regular vertical lines from steady drum beats, Electronic has heavy repeating bass bands at the bottom (`mel bins 0–35`), Rock builds up into a loud wall of sound around `10s`, and Instrumental/Folk have quiet upper frequencies and clear gaps between notes. This is why we also train a CNN on the spectrograms instead of only using the 518 summary features.



# DATA-4: Cleaning and Fixed Train/Val/Test Split

Every model **must** use `load_split()` from [scripts/data_prep/load_split.py](scripts/data_prep/load_split.py) so all results are comparable.

## How to use
Run from the repo root (`F21DL-main/`):
```python
from scripts.data_prep.load_split import load_split, get_label_names

X_train, X_val, X_test, y_train, y_val, y_test = load_split()
class_names = get_label_names()   # class_names[i] is the genre for label i

# CNN: get track_ids so spectrograms use the same split
(X_train, X_val, X_test, y_train, y_val, y_test,
 ids_train, ids_val, ids_test) = load_split(return_ids=True)
```
- `scale=True` (default): StandardScaler fitted on **train only**, then applied to val/test. Needed for MLP and K-means. Tree models can use `scale=False` and get the same results. Naive Bayes is affected by scaling through `var_smoothing` (see the Naive Bayes section).
- `n_features`: keep only the top-k features ranked by the EDA-2 ANOVA F-test, fitted on `X_train` only. `None` keeps all 518, `True` uses the EDA-2 choice of 150.
- Sanity check: `python scripts/data_prep/load_split.py`

## Why
- Every model must be trained and tested on the **same songs** with the **same features**, otherwise results can't be compared
- Using FMA's own `set_split` keeps songs by the same artist out of both train and test, so the score reflects how well the model generalises
- Scaling and filling missing values only use **training** statistics, so no test information leaks into training

## Expected outcome
Running `python scripts/data_prep/load_split.py` should show:
- An 80 / 10 / 10 train / validation / test split, with roughly equal songs per genre
- 518 audio features per song
- No missing values in any split
- Training features with mean ≈ 0 and standard deviation ≈ 1 (when `scale=True`)
- No track ID appearing in more than one split
- The 8 genres mapped to labels 0–7 in alphabetical order

Training has slightly fewer than 6,400 songs because the 6 short/corrupted audio files above were removed.

## Dropped columns
All 52 metadata columns from `tracks.csv` are dropped except `track_genre_top` (target) and `set_split` (used for splitting). `track_id` is not a feature but is returned with `return_ids=True`.

| Group | Columns | Why dropped |
| :--- | :--- | :--- |
| Target leakage | `track_genres`, `track_genres_all`, `track_tags`, `album_tags`, `artist_tags` | Contain the genre directly or through user tags |
| Mostly empty (>60% missing) | `track_lyricist`, `track_publisher`, `track_information`, `track_composer`, `artist_active_year_end`, `artist_wikipedia_page`, `track_date_recorded`, `artist_related_projects`, `artist_associated_labels`, `track_language_code`, `album_engineer`, `album_producer`, `artist_active_year_begin`, `artist_latitude`, `artist_longitude` | 63.7%–99.7% missing, too sparse to fill in reliably |
| IDs / free text / URLs | `album_id`, `artist_id`, `track_title`, `album_title`, `artist_name`, `artist_bio`, `album_information`, `artist_website`, `artist_location`, `artist_members`, `track_license` | No acoustic meaning; the model could memorise artists |
| Popularity stats, dates, constants | `album_comments`, `album_favorites`, `album_listens`, `album_tracks`, `album_type`, `album_date_created`, `album_date_released`, `artist_comments`, `artist_favorites`, `artist_date_created`, `track_bit_rate`, `track_comments`, `track_date_created`, `track_duration`, `track_favorites`, `track_interest`, `track_listens`, `track_number`, `set_subset` | Not audio content (`set_subset` is always "small") |

## Missing values
- Rows missing `track_genre_top` or `set_split` are dropped
- Infinite or missing feature values are filled with the **training** median

## Using load_split() in a new model script
Put the script in `scripts/models/` and add these lines at the top so it runs from anywhere:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.data_prep.load_split import load_split, get_label_names

X_train, X_val, X_test, y_train, y_val, y_test = load_split()   # scale=False for tree models / Naive Bayes
```

# BASE-1: Shared evaluation

Script: [scripts/models/evaluate.py](scripts/models/evaluate.py)

## Why
- Every model is scored by the **same code**, so the numbers in the results table can be compared directly
- All results end up in **one table**, [results/results_summary.csv](results/results_summary.csv), instead of being spread across terminal output
- Removes the scoring and plotting code that was copied into every model script

## What it provides
| Function | What it does |
| :--- | :--- |
| `evaluate(model, X, y)` | Predicts on `X` and returns accuracy, macro-F1 and the predictions |
| `print_test_report(y_test, pred, class_names)` | Prints the per-genre precision / recall / F1 and a text confusion matrix |
| `plot_confusion(y_test, pred, class_names, name, filename)` | Saves the test confusion matrix to `figures/<filename>` |
| `save_result(name, val_acc, val_f1, y_test, pred, class_names, train_secs, notes)` | Adds this model's row to `results/results_summary.csv`, or replaces it if the model is already there |

`results_summary.csv` has one row per model with validation and test accuracy / macro-F1, the **test F1 of every genre** (`f1_Electronic`, `f1_Folk`, ...), training time and a note with the chosen setting. It is sorted by test macro-F1.

## How to use it in a model script
```python
from scripts.models.evaluate import evaluate, print_test_report, plot_confusion, save_result

val_acc, val_f1, _ = evaluate(model, X_val, y_val)
test_acc, test_f1, test_pred = evaluate(model, X_test, y_test)
print_test_report(y_test, test_pred, class_names)
plot_confusion(y_test, test_pred, class_names, "MLP", "mlp_confusion_matrix.png")
save_result("MLP", val_acc, val_f1, y_test, test_pred, class_names, train_secs, notes="1 hidden layer of 256")
```
Models without a scikit-learn style `.predict()` (the PyTorch CNN) can skip `evaluate()` and pass their own test predictions to the other three functions.

## Metrics used and why
- **Accuracy:** the test set is balanced (100 songs per genre), so accuracy is meaningful and can be compared with the FMA paper's 46.4% and with chance (12.5%).
- **Macro-F1:** the average F1 over the 8 genres, each weighted equally. A model can't score well by being good at the easy genres (Hip-Hop, Rock) and ignoring the hard ones (Folk, Experimental). It is also the metric every model is tuned on.
- **Per-genre precision, recall and F1, plus the confusion matrix:** show *which* genres are confused with each other, which we can compare with the EDA findings.

# R3 Baselines: common setup

All three baselines (Decision Tree, Naive Bayes, Random Forest) follow the same steps, so they can be compared fairly:
1. Load the shared split with `load_split()`. The number of features is tuned too, using the top-k features ranked by the EDA-2 ANOVA F-test (fitted on the training set only).
2. Try every setting in a small grid and score it on the **validation** set by macro-F1. Every setting is saved to `results/<model>_tuning.csv`.
3. Retrain the best setting and evaluate it **once** on the test set.
4. Print the per-genre report, save a test confusion matrix to `figures/`, and add a row to `results/results_summary.csv` through the shared evaluation.

Everything uses `random_state=42`, so results are reproducible.

# R3 Baseline: Decision Tree

Script: [scripts/models/decision_tree.py](scripts/models/decision_tree.py)

## Why
- One of the three classical baselines required by R3
- The easiest model to **interpret**: its rules can be read directly, which shows which audio features separate the genres
- Shows the starting point that Random Forest (many trees averaged) improves on

## How to run
```bash
python scripts/models/decision_tree.py
```
Takes about 75 seconds.

## What was tried
72 settings. Every run is saved in [results/dt_tuning.csv](results/dt_tuning.csv).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 50, 150, 518 |
| Split criterion | `gini`, `entropy` |
| `max_depth` | 5, 10, 15, no limit |
| `min_samples_leaf` | 1, 5, 20 |

## Results
Best setting: 150 features, `gini`, `max_depth=10`, `min_samples_leaf=20` (depth 10, 192 leaves).

| Split | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| Validation | 42.3% | 0.418 |
| **Test** | **36.9%** | **0.363** |

Saved figures: `figures/dt_confusion_matrix.png` and `figures/dt_tree_top_levels.png` (the first two levels of the tree). The script also prints the first three levels as text rules.

## What we noticed
- **It is the weakest of the three baselines** and well below the FMA paper's 46.4%.
- **An unlimited tree overfits badly:** with no depth limit and 1 song per leaf it grows to depth 29 and 1,970 leaves, and drops to 0.359 validation macro-F1. Limiting the depth or requiring 20 songs per leaf gives the best results.
- **Fewer features helped a little:** 150 and 50 features (0.418 / 0.417) beat all 518 (0.401), which matches EDA-2.
- **The first split is `spectral_rolloff_mean_01`** (how much high-frequency energy the song has). The low side is mostly Instrumental and Folk (quiet, acoustic), and the high side then splits towards Rock and Hip-Hop. This is the same quiet-vs-loud divide the EDA plots show.
- **Easy genres:** Rock and Hip-Hop (F1 0.47), Electronic (0.47). **Hard genres:** Experimental (0.13), Folk (0.26) and Pop (0.31).

## Practical use for this topic
- **Explainable genre tagging:** a music platform could auto-tag new uploads with rules people can check, e.g. "low spectral rolloff and high MFCC 4 → Folk".
- **Cheap and fast:** it trains in under a second and needs no feature scaling, so it can run on very small devices.
- **Feature insight:** the top splits show which features matter, which helped confirm the EDA-2 feature selection.
- **Limits:** a single tree is unstable (small changes in the data give a very different tree) and overfits easily, which is why Random Forest does about 10 points better. It also can't work on raw spectrograms, which is why we use a CNN for those.

# R3 Baseline: Naive Bayes

Script: [scripts/models/naive_bayes.py](scripts/models/naive_bayes.py)

## Why
- One of the three classical baselines required by R3, and a simple **generative** (probabilistic) model, as opposed to the trees
- Very fast (trains in about 0.01 seconds)
- Tests how much the 518 features break its assumption that every feature is independent given the genre

## How to run
```bash
python scripts/models/naive_bayes.py
```
Takes about 5 seconds.

## What was tried
80 settings of Gaussian Naive Bayes. Every run is saved in [results/nb_tuning.csv](results/nb_tuning.csv).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 10, 25, 50, 100, 150, 200, 300, 518 |
| `var_smoothing` | `1e-9`, `1e-7`, `1e-5`, `1e-3`, `1e-1` |
| Features scaled? | no, yes |

## Results
Best setting: 300 features, unscaled, `var_smoothing=1e-9`.

| Split | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| Validation | 49.9% | 0.481 |
| **Test** | **41.8%** | **0.399** |

Saved figures: `figures/nb_confusion_matrix.png` and `figures/nb_feature_count.png` (validation macro-F1 against the number of features, scaled vs. unscaled).

## What we noticed
- **The number of features matters most.** Validation macro-F1 rises from 0.35 with 10 features to 0.48 at 200-300 features, then falls slightly at 518. Many of the 518 features are near-copies of each other (e.g. mean / median of the same signal, three chroma sets), so Naive Bayes counts the same evidence several times.
- **Scaling makes a difference, which it normally wouldn't for Naive Bayes.** `var_smoothing` adds a fraction of the *largest* feature variance to every feature. Unscaled, the largest variance is huge, so this acts as strong smoothing. Scaled, every variance is about 1, so the smoothing is tiny. With all 518 features, unscaled scores 0.476 and scaled 0.443.
- **Larger `var_smoothing` hurt:** `1e-9` and `1e-7` were best, and `1e-1` dropped to 0.25.
- **Note for EDA-2:** EDA-2 reported Naive Bayes at about 35% with all 518 features. We get 44-48% here depending on scaling, so the two should be checked against each other before the report.
- **Easy genres:** Rock (F1 0.62) and Hip-Hop (0.52). **Hard genres:** International (0.27), Folk (0.29) and Experimental (0.29). It predicts Hip-Hop too often (Electronic and Pop songs get labelled Hip-Hop).

# R3 Baseline: Random Forest

Script: [scripts/models/random_forest.py](scripts/models/random_forest.py)

## Why
- One of the three classical baselines required by R3, and an **ensemble** method
- Gives a reference score that the MLP and CNN need to beat
- Its feature importances show which audio features matter most, which helps with feature selection (R2)

## How to run
```bash
python scripts/models/random_forest.py
```
Takes about 30 seconds.

## What was tried
16 settings, all with `max_features="sqrt"`. Every run is saved in [results/rf_tuning.csv](results/rf_tuning.csv).

| Setting | Values tried |
| :--- | :--- |
| Number of features | 150, 518 |
| `n_estimators` (trees) | 200, 500 |
| `max_depth` | 20, no limit |
| `min_samples_leaf` | 1, 3 |

## Results
Best setting: all 518 features, 200 trees, `max_depth=20`, `min_samples_leaf=1`.

| Split | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| Validation | 57.1% | 0.555 |
| **Test** | **47.3%** | **0.471** |

Saved figures: `figures/rf_confusion_matrix.png` and `figures/rf_feature_importance.png` (20 most important features).

## What we noticed
- **Best of the three baselines** and just above the FMA paper's 46.4%. It is also slightly above the MLP (45.3%).
- **About 10 points better than a single Decision Tree.** Averaging many trees, each trained on a random sample of songs and features, removes most of the single tree's overfitting.
- **The settings hardly matter:** all 16 settings are between 0.530 and 0.555 validation macro-F1, and 150 vs. 518 features is within 1 point.
- **No single feature dominates:** the most important feature (`mfcc_mean_03`) only gets 0.7% of the total importance. The top features are the same kinds the Decision Tree uses: low MFCCs, spectral rolloff, spectral contrast and spectral centroid.
- **Easy genres:** Rock (F1 0.65), Hip-Hop (0.63) and Electronic (0.54). **Hard genres:** Folk (0.26), Experimental (0.32) and Pop (0.37). Instrumental is often predicted as Folk, and Pop as Electronic.

# R3 Comparison

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

- **Random Forest is the best baseline**, Naive Bayes is second and the Decision Tree is last.
- **All three find the same genres hard:** Folk, Experimental and Pop. This matches the EDA: Pop has no clear profile, Experimental is a catch-all genre, and Folk overlaps with Instrumental. The MLP and CNN also find Folk the hardest.
- **Test is 5-10 points below validation for every model**, because the setting was chosen on validation. The test score is the one to report.

# NN-1: MLP on the 518 features

Script: [scripts/models/mlp.py](scripts/models/mlp.py)

## How to run
From the repo root:
```bash
python scripts/models/mlp.py
```
Takes under a minute on a laptop CPU (the final model trains in about 2 seconds). It uses scikit-learn's `MLPClassifier` on the shared split with `load_split(scale=True)`, since an MLP needs scaled inputs.

## What was tried
12 settings, all scored on the **validation** set by macro-F1. Every run is saved in [results/mlp_tuning.csv](results/mlp_tuning.csv).

| Setting | Values tried |
| :--- | :--- |
| Hidden layers | `(256,)`, `(512, 256)`, `(256, 128, 64)` |
| L2 regularisation (`alpha`) | `0.0001`, `0.01` |
| Learning rate | `0.001`, `0.0003` |

Fixed for every run: ReLU, Adam, batch size 128, up to 200 epochs, `random_state=42`. Early stopping holds out 10% of the *training* data, so the validation set is only used to pick between settings.

## Results
Best setting: one hidden layer of 256 units, `alpha=0.01`, learning rate `0.0003`.

| Split | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| Validation | 55.5% | 0.547 |
| **Test** | **45.3%** | **0.453** |

The FMA paper's baseline is 46.4% (SVM on MFCCs), so the MLP lands just under it. Training curves are in `figures/mlp_training_curve.png`.

## What we noticed
- **Bigger was not better:** the deeper networks scored about 50-53% on validation against about 55% for the single hidden layer. With only ~6,400 training songs they most likely overfit.
- **The lower learning rate won** in every pairing of the other settings.
- **Validation is higher than test** because the setting was chosen on validation, so the test score is the one to report.
- **Easy genres:** Hip-Hop (F1 0.70), Rock (0.62) and Electronic (0.55).
- **Hard genres:** Folk is the worst (F1 0.16) and is confused with Experimental, International and Pop. Instrumental is often predicted as Folk. This matches the EDA finding that Folk and Instrumental overlap heavily.

## Still to do
- Add a `save_result(...)` call from the shared evaluation ([BASE-1](#base-1-shared-evaluation)) so the MLP gets a row in `results/results_summary.csv`.

# NN-2: CNN on the spectrograms

Script: [scripts/models/cnn.py](scripts/models/cnn.py)

## How to run
From the repo root (needs the spectrograms in `data/processed/spectrograms/` and `torch` from `requirements.txt`):
```bash
python scripts/models/cnn.py          # full grid, about 26 minutes on an M4 Max
python scripts/models/cnn.py --quick  # 1 setting, 2 epochs, just checks it runs
```
It uses the same split as every other model: the track ids come from `load_split(return_ids=True)`, and each id is matched to its cached `(128, 1292)` log-mel spectrogram. Spectrograms are normalised with the mean and std of the **training** set only.

## Model
4 blocks of conv (3x3) → batch norm → ReLU → max-pool (16, 32, 64, 128 channels), then global average pooling, dropout and a linear layer to the 8 genres. AdamW, weight decay `0.01`, batch size 32, up to 30 epochs. The epoch with the best validation macro-F1 is kept, and training stops early if that doesn't improve for 6 epochs.

## What was tried
4 settings, all scored on the **validation** set. Every run is saved in [results/cnn_tuning.csv](results/cnn_tuning.csv).

| Learning rate | Dropout | Best epoch | Val accuracy | Val macro-F1 | Train time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.001** | **0.5** | 28 | **59.4%** | **0.578** | 7.9 min |
| 0.0003 | 0.5 | 28 | 58.0% | 0.560 | 8.0 min |
| 0.0003 | 0.3 | 15 | 58.0% | 0.559 | 5.6 min |
| 0.001 | 0.3 | 10 | 57.1% | 0.555 | 4.7 min |

## Results
Best setting: learning rate `0.001`, dropout `0.5`.

| Split | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| Validation | 59.4% | 0.578 |
| **Test** | **53.75%** | **0.529** |

| Model | Test accuracy | Test macro-F1 |
| :--- | :--- | :--- |
| FMA paper baseline (SVM on MFCCs) | 46.4% | – |
| MLP on 518 features (NN-1) | 45.3% | 0.453 |
| **CNN on spectrograms (NN-2)** | **53.75%** | **0.529** |

Training curves are in `figures/cnn_training_curve.png`.

## Training time and hardware
Apple M4 Max (macOS, PyTorch `mps` backend). The best setting trained for 30 epochs in 7.9 minutes (about 16 seconds per epoch), and the full 4-setting grid took about 26 minutes. Loading all spectrograms into memory takes about 2.6 GB of RAM.

## What we noticed
- **The CNN beats the MLP by about 8.5 points.** The spectrograms keep timing and texture information that the 518 summary features throw away, which is what the EDA suggested.
- **Dropout 0.5 did better than 0.3** at both learning rates.
- **Both dropout 0.5 runs peaked at epoch 28 of 30**, so the 30-epoch limit may be holding them back a little.
- **Validation is noisy:** it moves by several points from one epoch to the next, so gaps of 1-2 points between settings are not reliable. The 4-setting grid is also small.
- **Easy genres:** Hip-Hop (F1 0.79), Rock (0.69) and Electronic (0.61).
- **Hard genres:** Folk (0.28) and Pop (0.39). Folk is also the MLP's worst genre and gets mixed up with Experimental, Instrumental and International.

## Still to do
- Add a `save_result(...)` call from the shared evaluation ([BASE-1](#base-1-shared-evaluation)). The CNN has no `.predict()`, so pass its test predictions to `save_result()` directly.
- NN-3 (#24): try augmentation and compare it against this CNN.

# NN-3: CNN data augmentation

Script: [scripts/models/cnn_augment.py](scripts/models/cnn_augment.py)

## How to run
```bash
python scripts/models/cnn_augment.py          # all 4 runs, about 30 minutes on an M4 Max
python scripts/models/cnn_augment.py --quick  # 1 run, 2 epochs, just checks it runs
```
It retrains the best NN-2 CNN (learning rate `0.001`, dropout `0.5`) with different augmentations. Everything else is the same in every run (split, seed, 40-epoch limit), so the augmentation is the only thing that changes. Augmentation is only applied to training batches, never to validation or test. `cnn.py` got one small change for this: `train()` takes an optional `augment` function.

## What was tried
| Augmentation | What it does |
| :--- | :--- |
| none (control) | The plain CNN, rerun with the same 40-epoch limit |
| `spec_mask` | SpecAugment-style: 2 random frequency bands (up to 20 mel bins) and 2 random time bands (up to 100 frames) set to the training mean, different for every sample |
| `time_shift` | Random circular shift along time of up to ±200 frames (about ±4.6 s) |
| `noise_and_gain` | Gaussian noise (std 0.1) plus a random per-sample loudness offset (std 0.2), in normalised units |

All runs are in [results/cnn_augmentation.csv](results/cnn_augmentation.csv); validation curves are in `figures/cnn_augmentation_curves.png`.

## Results
| Augmentation | Best epoch | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| none (control) | 15 | 60.1% | 0.574 | 44.5% | 0.438 |
| **`spec_mask`** | 15 | 58.6% | 0.563 | 48.25% | 0.485 |
| `noise_and_gain` | 25 | 57.4% | 0.560 | – | – |
| `time_shift` | 6 | 53.5% | 0.519 | – | – |

Test was only evaluated for the control and for the augmentation with the best validation score (`spec_mask`), so the test set was not used to choose between runs.

## Did augmentation help?
**We can't show that it did.** On validation, the plain CNN was best (0.574 macro-F1), `spec_mask` was about 1 point behind, `noise_and_gain` about 1.4 behind and `time_shift` about 5.5 behind. On test, `spec_mask` came out about 3.75 points above the control, but that goes the opposite way to validation.

The main reason to be careful is run-to-run noise. The control here is the same setting as the NN-2 CNN, which scored **53.75%** on test. In this run it scored **44.5%**. The GPU (`mps`) run is not exactly repeatable even with a fixed seed, and the epoch picked by validation macro-F1 changed from 28 to 15. Validation macro-F1 also jumps by several points from one epoch to the next, and the 800-song test set is small. A test gap of a few points between two runs is smaller than this noise, so it should not be read as `spec_mask` helping.

- **`time_shift` looks worse.** It is the only gap larger than the noise on validation (about 5.5 points). A shift moves the beat and intro/outro structure around, which may matter for genre.
- **`noise_and_gain` and `spec_mask` look neutral.** They are within noise of the control.
- **Caveat for NN-2:** because of the same noise, the 53.75% test score reported for the plain CNN is probably on the lucky side, and the true figure is likely lower. Treat NN-2 and NN-3 test numbers as roughly 45-54%, not as exact values.

## What would settle it
Repeat each setting with several seeds and compare the averages, and use a longer epoch limit with a learning-rate schedule to cut the epoch-to-epoch jumps. We did not do this because each run takes 5-10 minutes.

## Still to do
- Add a `save_result(...)` call from the shared evaluation ([BASE-1](#base-1-shared-evaluation)) for the control and `spec_mask` runs.
