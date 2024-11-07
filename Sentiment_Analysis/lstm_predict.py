import pickle

import tensorflow as tf
import yaml
from config import MODELS_DIR
from config import PARAMS_DIR
from config import PROCESSED_DATA_DIR
from data_preprocessing import clean_text
from tensorflow.keras.preprocessing.sequence import (  # type: ignore
    pad_sequences,
)  # pylint: disable=import-error,no-name-in-module

# Open tokenizer, model and its parameters
tokenizer_path = PROCESSED_DATA_DIR / "tokenizer.pickle"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
# Subfolders

with open(PARAMS_DIR, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)
    lstm_params = params.get("lstm_train")

if lstm_params is None:
    raise KeyError("'lstm_train' key not found in params.yaml")

model_name = lstm_params["model_name"]
max_len = lstm_params["max_len"]
final_model_path = MODELS_DIR / f"{model_name}_final.keras"


with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)


def predict_sentiment(text, loaded_model=None):
    """
    Predicts the sentiment of the given text using a trained LSTM model.

    This function preprocesses the input text by cleaning and tokenizing it,
    and then pads the sequence to match the input shape required by the model.
    It loads the trained LSTM model, makes a prediction, and returns the
    predicted sentiment as either "Positive" or "Negative."

    Args:
        text (str): The input text for sentiment prediction.

    Returns:
        str: The predicted sentiment ("Positive" or "Negative").
    """

    # Clean the text
    cleaned = clean_text(text)
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    # Load model
    if loaded_model is None:
        loaded_model = tf.keras.models.load_model(final_model_path)
    # Predict
    pred_prob = loaded_model.predict(padded)
    pred_class = (pred_prob > 0.5).astype("int32")
    sentiment = "Positive" if pred_class[0][0] == 1 else "Negative"
    return sentiment


# Example prediction
sample_text = "I absolutely love this product! It works wonders."
print(f"Sample Text: {sample_text}")
print(f"Predicted Sentiment: {predict_sentiment(sample_text)}")
