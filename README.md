# F21DL

# Plan

## Data Processing Procedures
- [x] Merge tracks.csv, features.csv, genres.csv
- [x] Create the spectrograms from the audio files
- [x] Check for mismatch in files
- [x] Check for corrupted files

## Baseline Models
- [ ] Pick three models
- [x] Random Forest ([scripts/models/random_forest.py](scripts/models/random_forest.py))

## Neural Networks
- [ ] Design the neural network
- [x] MLP on the 518 features ([scripts/models/mlp.py](scripts/models/mlp.py))
- [x] CNN on the spectrograms ([scripts/models/cnn.py](scripts/models/cnn.py))
- [ ] CNN data augmentation


## Project structure
```
F21DL-main/
├── data/
│   ├── fma_metadata/            # features.csv, tracks.csv, genres.csv (not pushed, too big)
│   ├── fma_small_audio/         # MP3 clips (not pushed, too big)
│   └── processed/
│       └── fma_small_merged.csv # output of merge.py
├── scripts/
│   ├── __init__.py
│   ├── data_prep/
│   │   ├── __init__.py
│   │   ├── valid_audio.py       # finds corrupted / short audio files
│   │   ├── merge.py             # merges features.csv + tracks.csv for fma_small
│   │   └── load_split.py        # shared cleaning + train/val/test split (DATA-4)
│   └── models/
│       ├── __init__.py
│       ├── random_forest.py     # R3 baseline
│       ├── mlp.py               # NN-1 MLP on the 518 features
│       └── cnn.py               # NN-2 CNN on the spectrograms
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
| Random Forest baseline | `python scripts/models/random_forest.py` | ~20 seconds |

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
- `scale=True` (default): StandardScaler fitted on **train only**, then applied to val/test. Needed for MLP and K-means. Tree models and Naive Bayes can use `scale=False` and get basically the same results.
- Feature selection (R2) is **not** part of `load_split()`. Fit it on `X_train` only.
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

# R3 Baseline: Random Forest

Script: [scripts/models/random_forest.py](scripts/models/random_forest.py)

## Why
- One of the three classical baselines (Decision Tree, Random Forest, Naive Bayes) required by R3
- Gives a reference score that the MLP and CNN need to beat
- Its feature importances show which audio features matter most, which helps with feature selection (R2)

## How to run
From the `F21DL-main` folder:
```powershell
python scripts/models/random_forest.py
```
Takes about 20 seconds.

## What it does
1. Loads the shared split with `load_split(scale=False)` (trees don't need scaling)
2. Tries 4 settings (`n_estimators` 200/500 × `max_depth` None/20, `max_features="sqrt"`) and picks the best by **validation** macro-F1
3. Retrains with the best setting and evaluates **once** on the test set
4. Uses `random_state=42`, so results are reproducible

## Expected outcome
The script prints:
- Validation accuracy and macro-F1 for each setting tried, and the best setting
- Final validation and test accuracy and macro-F1
- A per-genre classification report (precision, recall, F1)
- A test confusion matrix showing which genres get mixed up
- The 10 most important features

Test accuracy should be well above chance (12.5%) and around the FMA paper's baseline of 46.4% (SVM on MFCCs). Validation scores will be somewhat higher than test because the setting is chosen on validation, so always report the **test** score.

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
- Run the results through the shared evaluation (BASE-1, #18) once it exists, so the MLP gets a row in the shared results table.

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
- Run the results through the shared evaluation (BASE-1, #18) once it exists.
- NN-3 (#24): try augmentation and compare it against this CNN.
