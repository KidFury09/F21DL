# F21DL

# Plan

## Data Processing Procedures
- [x] Merge tracks.csv, features.csv, genres.csv
- [ ] Create the spectrograms from the audio files
- [ ] Check for mismatch in files
- [x] Check for corrupted files

## Baseline Models
- [ ] Pick three models
- [x] Random Forest ([scripts/models/random_forest.py](scripts/models/random_forest.py))

## Neural Networks
- [ ] Design the neural network

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