import pandas as pd
import os
import glob


def get_track_id(path):
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