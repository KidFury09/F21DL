"""
DATA-4: Clean the merged dataset and create a single fixed train/val/test split.

Every model (Decision Tree, Random Forest, Naive Bayes, MLP, CNN) must use the
same split and the same feature set, otherwise results can't be compared.

Usage (from the repo root, F21DL-main/):

    from scripts.data_prep.load_split import load_split, get_label_names

    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    class_names = get_label_names()          # index i -> genre name

    # CNN owner: get the track_ids belonging to each split
    (X_train, X_val, X_test, y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = load_split(return_ids=True)

Pipeline:
    1. Load data/processed/fma_small_merged.csv (output of merge.py)
    2. Drop rows missing the target or split
    3. Keep only the 518 numeric audio features (all metadata dropped, see DROPPED_COLUMNS)
    4. Split using FMA's own `set_split` column (no random re-splitting)
    5. Handle missing / infinite feature values with TRAIN medians
    6. Encode `track_genre_top` (8 classes) as integers 0..7
    7. Optionally scale with StandardScaler fitted on TRAIN only
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler


# File paths for easier calls
script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_MERGED_CSV = os.path.normpath(
    os.path.join(script_dir, "..", "..", "data", "processed", "fma_small_merged.csv")
)

TARGET_COL = "track_genre_top"
SPLIT_COL = "set_split"
ID_COL = "track_id"
SPLIT_NAMES = ("training", "validation", "test")

# Default number of features chosen by EDA-2 (ANOVA F-test on X_train).
# Using the top 150 features keeps validation accuracy on par with all 518
# while dropping ~71% of redundant/noisy columns.
DEFAULT_N_FEATURES = 150


# Metadata columns from tracks.csv that are dropped, grouped by the reason.
# Missing percentages are measured on the 7,994 tracks in fma_small_merged.csv.
DROPPED_COLUMNS = {
    "Target leakage (encode genre directly or via user tags)": [
        "track_genres", "track_genres_all", "track_tags", "album_tags", "artist_tags",
    ],
    "Mostly empty (>60% missing, too sparse to impute)": [
        "track_lyricist",            # 99.7% missing
        "track_publisher",           # 99.0%
        "track_information",         # 98.0%
        "track_composer",            # 97.7%
        "artist_active_year_end",    # 95.5%
        "artist_wikipedia_page",     # 94.9%
        "track_date_recorded",       # 94.2%
        "artist_related_projects",   # 91.3%
        "artist_associated_labels",  # 89.4%
        "track_language_code",       # 87.4%
        "album_engineer",            # 85.7%
        "album_producer",            # 82.5%
        "artist_active_year_begin",  # 80.7%
        "artist_latitude",           # 63.7%
        "artist_longitude",          # 63.7%
    ],
    "Identifiers / free text / URLs (no acoustic meaning, risk memorising artists)": [
        "album_id", "artist_id", "track_title", "album_title", "artist_name",
        "artist_bio", "album_information", "artist_website", "artist_location",
        "artist_members", "track_license",
    ],
    "Platform popularity stats, dates and constants (not audio content)": [
        "album_comments", "album_favorites", "album_listens", "album_tracks",
        "album_type", "album_date_created", "album_date_released",
        "artist_comments", "artist_favorites", "artist_date_created",
        "track_bit_rate", "track_comments", "track_date_created",
        "track_duration", "track_favorites", "track_interest",
        "track_listens", "track_number",
        "set_subset",                # always "small"
    ],
}

METADATA_PREFIXES = ("album_", "artist_", "set_", "track_")


def _feature_columns(df):
    """The 518 audio feature columns from features.csv (everything that isn't metadata)."""
    return [c for c in df.columns if c != ID_COL and not c.startswith(METADATA_PREFIXES)]


def _check_dropped_columns(df):
    """Make sure every metadata column is accounted for in DROPPED_COLUMNS.
    track_id is kept out of X but returned separately when return_ids=True."""
    documented = {c for cols in DROPPED_COLUMNS.values() for c in cols}
    metadata = {c for c in df.columns if c.startswith(METADATA_PREFIXES)}
    metadata -= {TARGET_COL, SPLIT_COL, ID_COL}
    undocumented = metadata - documented
    if undocumented:
        raise ValueError(f"Metadata columns with no documented drop reason: {sorted(undocumented)}")


# AI Generated Function
# Prompt: Return the sorted unique genre names from track_genre_top so label index i maps to genre name
def get_label_names(csv_path=PATH_MERGED_CSV):
    """Genre names in label-encoded order, i.e. get_label_names()[i] is class i."""
    genres = pd.read_csv(csv_path, usecols=[TARGET_COL])[TARGET_COL].dropna()
    return list(np.sort(genres.unique()))


# AI Generated Function
# Prompt: Load fma_small_merged.csv, drop metadata columns, split into train/val/test using set_split, fill missing values with train medians, encode track_genre_top, optionally select top-k features using ANOVA F-test fitted on train only, and scale features using StandardScaler fitted on train only
def load_split(csv_path=PATH_MERGED_CSV, scale=True, return_ids=False, n_features=None):
    """
    Load the merged FMA small dataset and return the fixed train/val/test split.

    Args:
        csv_path:   path to fma_small_merged.csv
        scale:      if True, StandardScaler is fitted on X_train only and applied
                    to X_val / X_test. Needed for MLP and K-means; harmless for
                    tree models and Naive Bayes.
        return_ids: if True, also return the track_id Series for each split
                    (so the CNN can use the same split on spectrograms).
        n_features: number of top features to keep (ranked by ANOVA F-statistic
                    on X_train, y_train only). Pass None or 518 to keep all 518
                    features, or True / DEFAULT_N_FEATURES (150) for the chosen
                    EDA-2 feature set.

    Returns:
        X_train, X_val, X_test : pd.DataFrame, feature columns
        y_train, y_val, y_test : pd.Series of ints 0..7 (see get_label_names())
        [ids_train, ids_val, ids_test] : pd.Series of track_ids, if return_ids
    """
    df = pd.read_csv(csv_path, low_memory=False)
    _check_dropped_columns(df)

    # Drop rows without a label or split (none in the current file, but be safe)
    df = df.dropna(subset=[TARGET_COL, SPLIT_COL]).reset_index(drop=True)

    feature_cols = _feature_columns(df)
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)

    # Use FMA's own split column, no random re-splitting
    masks = {name: (df[SPLIT_COL] == name).to_numpy() for name in SPLIT_NAMES}
    X_train, X_val, X_test = (X[masks[n]].copy() for n in SPLIT_NAMES)

    # Missing values: fill with TRAIN medians so no val/test info leaks in
    train_medians = X_train.median()
    X_train, X_val, X_test = (x.fillna(train_medians) for x in (X_train, X_val, X_test))

    # Encode target labels (alphabetical order, matches get_label_names())
    encoder = LabelEncoder().fit(df[TARGET_COL])
    y = pd.Series(encoder.transform(df[TARGET_COL]), name=TARGET_COL)
    y_train, y_val, y_test = (y[masks[n]] for n in SPLIT_NAMES)

    # Feature selection (EDA-2): rank on TRAIN only using ANOVA F-statistic
    if n_features is True:
        n_features = DEFAULT_N_FEATURES
    if n_features is not None and n_features < len(feature_cols):
        f_scores, _ = f_classif(X_train, y_train)
        top_idx = np.argsort(f_scores)[::-1][: int(n_features)]
        feature_cols = [feature_cols[i] for i in top_idx]
        X_train, X_val, X_test = (x[feature_cols] for x in (X_train, X_val, X_test))

    # Scaling: fit on train ONLY, then apply to val and test
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_val, X_test = (
            pd.DataFrame(scaler.transform(x), index=x.index, columns=feature_cols)
            for x in (X_train, X_val, X_test)
        )

    result = [X_train, X_val, X_test, y_train, y_val, y_test]
    if return_ids:
        ids = df[ID_COL]
        result += [ids[masks[n]] for n in SPLIT_NAMES]
    return tuple(result)


if __name__ == "__main__":
    # Quick sanity check: python scripts/data_prep/load_split.py
    (X_train, X_val, X_test, y_train, y_val, y_test,
     ids_train, ids_val, ids_test) = load_split(return_ids=True)
    names = get_label_names()

    print(f"Features: {X_train.shape[1]}")
    for split, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        counts = {names[k]: int(v) for k, v in y.value_counts().sort_index().items()}
        print(f"{split:5s} X={X.shape}  y={y.shape}  NaNs={int(X.isna().sum().sum())}  {counts}")

    print(f"Train mean ~0: {X_train.values.mean():.4f}   Train std ~1: {X_train.values.std():.4f}")

    overlap = (set(ids_train) & set(ids_val)) | (set(ids_train) & set(ids_test)) | (set(ids_val) & set(ids_test))
    print(f"Track IDs shared between splits: {len(overlap)}")
    print(f"Classes: {dict(enumerate(names))}")
