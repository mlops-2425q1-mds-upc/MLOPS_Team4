import os
import random
import pickle
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from config import MODELS_DIR
from config import PROCESSED_DATA_DIR
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from lstm_predict import predict_sentiment
from pydantic import BaseModel

try:
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError as e:
    raise ImportError("Ensure TensorFlow is installed and accessible.") from e

# Global variables
model = None
MODEL_INFO = {}
DATASET = {}


class TextInput(BaseModel):
    """
    Class for post data input. Contains a list of sentences (tweets).
    """

    tweets: List[str]


@asynccontextmanager
async def lifespan(app):
    """
    Load the clean dataset, model and its info before starting
    """
    global model
    global DATASET

    print("Loading dataset...")
    positive_path = os.path.join(PROCESSED_DATA_DIR, "positive_subset_examples.pkl")
    negative_path = os.path.join(PROCESSED_DATA_DIR, "negative_subset_examples.pkl")

    if os.path.exists(positive_path) and os.path.exists(negative_path):
        DATASET["positive"] = pickle.load(open(positive_path,'rb'))
        DATASET["negative"] = pickle.load(open(negative_path,'rb'))
    else:
        raise FileNotFoundError("Clean data file not found!")

    print("Loading model...")
    model_path = os.path.join(MODELS_DIR, "optimized_lstm_final.keras")
    print(f"Model PATH: {model_path}")
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError("Model file not found!")

    # Load model info during startup
    print("Loading model info...")
    metrics_path = os.path.join(MODELS_DIR, "metrics", "lstm_metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            MODEL_INFO["metrics"] = f.read()
    else:
        raise FileNotFoundError("Metrics file not found!")

    yield

    # Cleanup during shutdown
    print("Shutting down...")


# Initialize the FastAPI app with the lifespan
app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Return custom error if the format of post in "predict sentiment"
    is not a list
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": f"Invalid input format: {exc.errors()}"},
    )


# Endpoint 1: Predict sentiment
@app.post("/predict")
async def predict_sentiment_api(input_data: TextInput):
    """
    Predict the sentiment of the input text. This endpoint processes the input
    text and returns predictions for each tweet in the list.

    Args:
    - input_data (TextInput): A list of tweets to analyze.

    Returns:
    - dict: A dictionary where each tweet is mapped to its predicted sentiment.
    """
    tweets = input_data.tweets
    if not tweets:
        raise HTTPException(status_code=400, detail="Text input is required")

    prediction_dict = {}
    for tweet in tweets:
        prediction_dict[tweet] = predict_sentiment(tweet, model)

    return prediction_dict


# Endpoint 2: Get model characteristics
@app.get("/model_info")
async def get_model_info():
    """
    Get model information, like accuracy, from the model.info file
    """
    if MODEL_INFO:
        return {"model_info": MODEL_INFO}
    raise HTTPException(status_code=404, detail="Model information not available")


# Endpoint 3: Extract random examples
@app.get("/random_example")
async def get_random_examples():
    """
    Extract one positive and negative examples from the dataset
    """
    try:
        if DATASET["positive"] and DATASET["negative"]:
            return {
                "positive_example": DATASET["positive"].pop(),
                "negative_example": DATASET["negative"].pop(),
            }
        else:
            raise KeyError("Empty dataset")
    except (KeyError, IndexError):
        print("Dataset not found, returning default examples")
        default_positive_examples = ["gonna go lay floor new bathroom today pronounc baff bath beef im common"]
        default_negative_examples = ["instead run spent copious amount time buri code websit epic nerd fail suck"]
        positive = random.choice(default_positive_examples)
        negative = random.choice(default_negative_examples)
        return {"positive_default_example": positive, "negative_default_example": negative}
