import pandas as pd
import os
import glob

# File paths for easier calls
script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_FMA_METADATA = os.path.join(script_dir, "../..", "data", "fma_metadata")
PATH_FMA_AUDIO = os.path.join(script_dir, "../..", "data", "fma_small_audio")

def get_track_id(path=PATH_FMA_AUDIO):
    files = glob.glob(os.path.join(path, "*", "*.mp3"))
    ids = []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        ids.append(int(stem))

    return pd.DataFrame({"track_id": sorted(ids)})

# AI Generated Function
# Prompt: Complete this function to join two dataframes. CSV files may also be passed so convert them into dataframes. Function definition is erge_frames(csv_1, csv_2, field="track_id")
def merge_frames(csv_1, csv_2, field="track_id"):
    # features.csv / tracks.csv are read with track_id as the index, so move it to a column
    if field not in csv_1.columns and csv_1.index.name == field:
        csv_1 = csv_1.reset_index()
    if field not in csv_2.columns and csv_2.index.name == field:
        csv_2 = csv_2.reset_index()

    # inner join: keep only track_ids present in both frames
    # one_to_one: raise if either side has duplicate track_ids
    return pd.merge(csv_1, csv_2, on=field, how="inner", validate="one_to_one")


if __name__ == "__main__":

    # AI help
    # features_df and tracks_df produiced by Claude Opus 5.5
    # Prompt: convert features and tracks into panda dataframes and name them features_df and tracks_df
    features = os.path.join(PATH_FMA_METADATA, "features.csv")
    features_df = pd.read_csv(features, index_col=0, header=[0, 1, 2])
    features_df.columns = ["_".join(col) for col in features_df.columns]

    tracks = os.path.join(PATH_FMA_METADATA, "tracks.csv")
    tracks_df = pd.read_csv(tracks, index_col=0, header=[0, 1])
    tracks_df.columns = ["_".join(col) for col in tracks_df.columns]

    extracted_s_id_df = get_track_id()

    audio_features_df = merge_frames(extracted_s_id_df, features_df)
    audio_features_tracks_df = merge_frames(audio_features_df, tracks_df)

    path_processed = os.path.join(script_dir, "../..", "data", "processed")
    os.makedirs(path_processed, exist_ok=True)
    audio_features_tracks_df.to_csv(os.path.join(path_processed, "fma_small_merged.csv"), index=False)

    