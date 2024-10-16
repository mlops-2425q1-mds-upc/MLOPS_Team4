# Implemented by Umut Ekin Gezer
import pickle
import re

import nltk
import numpy as np
import pandas as pd
import yaml
from config import INTERIM_DATA_DIR
from config import PARAMS_DIR
from config import PROCESSED_DATA_DIR
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download("stopwords")


def clean_text(text):
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


def data_preprocessing():
    # Load parameters from YAML
    with open(PARAMS_DIR, "r") as file:
        params = yaml.safe_load(file)
        lstm_params = params.get("lstm_train")

    # Check if lstm_params and logreg_params are available
    if lstm_params is None:
        raise KeyError("'lstm_train' key not found in params.yaml")

    # Extract parameters
    max_vocab_size = lstm_params["max_vocab_size"]
    max_len = lstm_params["max_len"]

    # Construct the path to the 'clean_dataset.csv' in 'data/interim' folder
    df_clean_path = INTERIM_DATA_DIR / "clean_dataset.csv"

    # Check if the file exists
    if df_clean_path.exists():
        df = pd.read_csv(df_clean_path)
        print("File loaded successfully.")

        # Drop rows with missing values
        df = df.dropna()
        print(f"DataFrame shape after dropping missing values: {df.shape}")

        # Data cleaning
        df["cleaned_text"] = df["text"].apply(clean_text)
        y = df["positive"].values
        X_text = df["cleaned_text"]

        # Ensure the processed data directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save cleaned data
        cleaned_data_path = PROCESSED_DATA_DIR / "cleaned_data.csv"
        df.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved at {cleaned_data_path}.")

        # **LSTM Data Preparation**
        # Tokenization
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_text)
        sequences = tokenizer.texts_to_sequences(X_text)
        vocab_size = len(tokenizer.word_index) + 1
        print(f"LSTM Vocabulary Size: {vocab_size}")

        # Padding sequences
        X_lstm = pad_sequences(sequences, maxlen=max_len, padding="post")

        # Save X_lstm and y as .npz files
        save_path_lstm = PROCESSED_DATA_DIR / "X_y_data_lstm.npz"
        np.savez(save_path_lstm, X=X_lstm, y=y)
        print(f"LSTM data saved successfully at {save_path_lstm}.")

        # Save tokenizer for later use
        tokenizer_path = PROCESSED_DATA_DIR / "tokenizer.pickle"
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved at {tokenizer_path}.")

    else:
        print(f"File not found: {df_clean_path}")


if __name__ == "__main__":
    data_preprocessing()
