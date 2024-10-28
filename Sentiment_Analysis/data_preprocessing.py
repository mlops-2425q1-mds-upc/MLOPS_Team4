"""
This module provides functions for preprocessing text data,
including cleaning and preparing data for LSTM training.
"""
import pickle
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Attempt to import TensorFlow Keras components; handle import errors
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
except ImportError as e:
    raise ImportError("Ensure TensorFlow is installed and accessible.") from e

from .config import PARAMS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

nltk.download("stopwords", quiet=True)  # Download stopwords if not present


def clean_text(text: str) -> str:
    """Clean the input text by removing URLs, mentions, hashtags, numbers,
    punctuation, and stopwords, and applying stemming.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Remove URLs, mentions, hashtags, and numbers
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"\d+", "", text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r"[^\w\s]", "", text).lower()
    # Tokenize, remove stopwords, and apply stemming
    tokens = text.split()
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def data_preprocessing() -> None:
    """Preprocess the dataset for sentiment analysis by cleaning the text
    data, tokenizing, and saving the cleaned data for LSTM training.

    Raises:
        KeyError: If required keys are not found in params.yaml.
    """
    # Load parameters from YAML
    with open(PARAMS_DIR, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
        lstm_params = params.get("lstm_train")
        preprocessing_params = params.get("data_preprocessing")

    # Check for required parameters
    if lstm_params is None:
        raise KeyError("'lstm_train' key not found in params.yaml")
    if preprocessing_params is None:
        raise KeyError("'data_preprocessing' key not found in params.yaml")

    # Extract parameters
    track_emissions = preprocessing_params["track_emissions"]
    max_vocab_size = lstm_params["max_vocab_size"]
    max_len = lstm_params["max_len"]

    # Start emissions tracking if enabled
    if track_emissions:
        emissions_tracker = EmissionsTracker(
            project_name="Team4",
            experiment_name="cleaning",
            output_file="model.csv",
            output_dir="./emissions/",
            save_to_file=True,
            measure_power_secs=5,
        )
        emissions_tracker.start()

    # Construct the path to the dataset
    df_path = RAW_DATA_DIR / "training.1600000.processed.noemoticon.csv"

    # Check if the file exists
    if not df_path.exists():
        print(f"File not found: {df_path}")
        return

    cols = ["sentiment", "id", "date", "query_string", "user", "text"]

    # Read the file
    try:
        df = pd.read_csv(df_path, header=None, names=cols, encoding="latin-1")
        print("File loaded successfully.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Feature selection
    df["positive"] = df["sentiment"].replace([0, 4], [0, 1])
    df.drop(columns=["id", "query_string", "sentiment", "date", "user"], inplace=True)
    df.dropna(inplace=True)
    print(f"DataFrame shape after dropping missing values: {df.shape}")

    # Data cleaning
    df["cleaned_text"] = df["text"].apply(clean_text)
    y = df["positive"].values
    cleaned_text_data = df["cleaned_text"]

    # Ensure the processed data directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    cleaned_data_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved at {cleaned_data_path}.")

    # **LSTM Data Preparation**
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_text_data)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"LSTM Vocabulary Size: {vocab_size}")

    # Padding sequences
    padded_sequences = pad_sequences(
        tokenizer.texts_to_sequences(cleaned_text_data),
        maxlen=max_len,
        padding="post",
    )

    # Save padded_sequences and y as .npz files
    save_path_lstm = PROCESSED_DATA_DIR / "X_y_data_lstm.npz"
    np.savez(save_path_lstm, X=padded_sequences, y=y)
    print(f"LSTM data saved successfully at {save_path_lstm}.")

    # Save tokenizer for later use
    tokenizer_path = PROCESSED_DATA_DIR / "tokenizer.pickle"
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved at {tokenizer_path}.")

    # Stop tracking emissions if enabled
    if track_emissions:
        emissions_tracker.stop()


if __name__ == "__main__":
    data_preprocessing()
