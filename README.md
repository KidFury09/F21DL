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
│       └── random_forest.py     # R3 baseline
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