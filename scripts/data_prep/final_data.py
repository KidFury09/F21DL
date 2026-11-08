from valid_audio import check_song_length, remove_bad_songs
from merge import get_track_id, merge_frames
import pandas as pd
import os

# File paths for easier calls
script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_FMA_METADATA = os.path.join(script_dir, "../..", "data", "fma_metadata")
PATH_FMA_AUDIO = os.path.join(script_dir, "../..", "data", "fma_small_audio")


if __name__ == '__main__':
    bad_songs = []

    bad_songs.extend(check_song_length(PATH_FMA_AUDIO))
    remove_bad_songs(bad_songs, PATH_FMA_AUDIO)
    
    # AI help
    # features_df and tracks_df produiced by Claude Opus 5.5
    # Prompt: convert features and tracks into panda dataframes and name them features_df and tracks_df
    features = os.path.join(PATH_FMA_METADATA, "features.csv")
    features_df = pd.read_csv(features, index_col=0, header=[0, 1, 2])
    features_df.columns = ["_".join(col) for col in features_df.columns]

    tracks = os.path.join(PATH_FMA_METADATA, "tracks.csv")
    tracks_df = pd.read_csv(tracks, index_col=0, header=[0, 1])
    tracks_df.columns = ["_".join(col) for col in tracks_df.columns]

    extracted_s_id_df = get_track_id(PATH_FMA_AUDIO)

    audio_features_df = merge_frames(extracted_s_id_df, features_df)
    audio_features_tracks_df = merge_frames(audio_features_df, tracks_df)

    path_processed = os.path.join(script_dir, "../..", "data", "processed")
    os.makedirs(path_processed, exist_ok=True)
    audio_features_tracks_df.to_csv(os.path.join(path_processed, "fma_small_merged.csv"), index=False)
