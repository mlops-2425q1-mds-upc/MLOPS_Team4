"""
API using FastAPI. It creates three endpoints to interact with the LSTM model.
It can run in local by doing "fastapi devel path" or in AWS virtual sercer
"""
import os
import random
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

# FastAPI app initialization
app = FastAPI()

# Global variables
model = None
MODEL_INFO = {}


class TextInput(BaseModel):
    """
    Class for post data input. Contains a list of sentences (tweets).
    """

    tweets: List[str]


@asynccontextmanager
async def lifespan(application):
    """
    Load the model and its info before
    """
    global model

    print("Loading model...")
    model_path = os.path.join(MODELS_DIR, "optimized_lstm_final.keras")
    print(f"Model PATH: {model_path}")
    model = load_model(model_path)

    # Load model info during startup
    print("Loading model info...")
    metrics_path = os.path.join(MODELS_DIR, "metrics", "lstm_metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            MODEL_INFO["metrics"] = f.read()
    else:
        raise FileNotFoundError("Metrics file not found!")

    yield application

    # Cleanup during shutdown
    print("Shutting down...")


# Register lifespan function with FastAPI
app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(exc: RequestValidationError):
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
        df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
        positive = df["cleaned_text"][df["positive"] == 1].sample(n=1).iloc[0]
        negative = df["cleaned_text"][df["positive"] == 0].sample(n=1).iloc[0]
        return {"positive_example": positive, "negative_example": negative}
    except FileNotFoundError:
        default_positive_examples = [
            # Add your default positive examples here...
        ]
        default_negative_examples = [
            # Add your default negative examples here...
        ]
        positive = random.choice(default_positive_examples)
        negative = random.choice(default_negative_examples)
        return {"positive_example": positive, "negative_example": negative}
