"""Load the crypto tweet data for sentiment and emotion fine-tuning."""

# import necessary libraries for data handling and manipulation

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

#Defines the three sentiment labels and five emotion labels,
#  as well as the project directory and default data path.

LABELS = ["negative", "neutral", "positive"]
EMOTIONS = ["Happy", "Angry", "Surprise", "Sad", "Fear"]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "SM_perceptions_51-Attacks"
    / "Datasets"
    / "cleaned datasets"
    / "Dataset_benchmark_bitcoin_15_05_2014.csv"
)

#Loads the sentiment data from a CSV file, returning a DataFrame with 
# usable tweet text and its corresponding sentiment label.
# also performs data cleaning and filtering to ensure the text and label columns are valid and non-empty.

def load_sentiment_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Return usable tweet text and its positive/neutral/negative label."""
    frame = pd.read_csv(path, usecols=["clean_content", "sentiment_cat"], nrows=max_rows)
    frame = frame.rename(columns={"clean_content": "text", "sentiment_cat": "label"})
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    frame = frame[(frame["text"] != "") & frame["label"].isin(LABELS)]
    return frame.reset_index(drop=True)

# Loads the emotion data from a CSV file, returning a DataFrame with
#  usable tweet text and its corresponding five emotion scores.
# also performs data cleaning and filtering to ensure the text column is valid and non-empty,
# and that the emotion scores are numeric values between 0 and 1.

def load_emotion_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Return usable tweet text and the five emotion scores, each from 0 to 1."""
    frame = pd.read_csv(path, usecols=["clean_content", *EMOTIONS], nrows=max_rows)
    frame = frame.rename(columns={"clean_content": "text"}).dropna(subset=["text"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    for emotion in EMOTIONS:
        frame[emotion] = pd.to_numeric(frame[emotion], errors="coerce").fillna(0).clip(0, 1)
    return frame.reset_index(drop=True)

# Splits the data into train, validation, and test sets with a 70/15/15 ratio,
# ensuring reproducibility with a random seed defaulted to 42.
# returns three DataFrames for the train, validation, and test sets, respectively.
def split_data(
    frame: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible 70% train, 15% validation, and 15% test splits."""
    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame["label"]
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed, stratify=remainder["label"]
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
