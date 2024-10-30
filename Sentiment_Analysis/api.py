import os
import random
from contextlib import asynccontextmanager

from config import MODELS_DIR
from fastapi import FastAPI
from fastapi import HTTPException
from lstm_predict import predict_sentiment
from pydantic import BaseModel
from tensorflow import keras

app = FastAPI()

model = None
model_info = {}


class TextInput(BaseModel):
    text: str


# Async context manager for the lifespan of the FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_info
    # Load the model during startup
    print("Loading model...")
    model_path = (
        str(MODELS_DIR) + "/optimized_lstm_final.keras"
    )  # Replace with actual model name
    print(f"Model PATH: {model_path}")
    model = keras.models.load_model(model_path)

    # Load model info during startup
    print("Loading model info...")
    metrics_path = str(MODELS_DIR) + "/metrics/lstm_metrics.txt"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_info["metrics"] = f.read()
    else:
        raise FileNotFoundError("Metrics file not found!")

    yield  # This is where the FastAPI app will run

    # Cleanup during shutdown
    print("Shutting down...")


# Register lifespan function with FastAPI
app = FastAPI(lifespan=lifespan)

# Endpoint 1: Predict sentiment
@app.post("/predict")
async def predict_sentiment_api(input_data: TextInput):
    text = input_data.text
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    prediction = predict_sentiment(text, model)
    return {"input_text": text, "prediction": prediction}


# Endpoint 2: Get model characteristics
@app.get("/model_info")
async def get_model_info():
    if model_info:
        return {"model_info": model_info}
    else:
        raise HTTPException(status_code=404, detail="Model information not available")


# Endpoint 3: Extract random examples
@app.get("/random_example")
async def get_random_examples():
    positive_examples = [
        "Example of a positive outcome 1",
        "Example of a positive outcome 2",
    ]  # Replace with actual examples
    negative_examples = [
        "Example of a negative outcome 1",
        "Example of a negative outcome 2",
    ]  # Replace with actual examples
    if positive_examples and negative_examples:
        positive = random.choice(positive_examples)
        negative = random.choice(negative_examples)
        return {"positive_example": positive, "negative_example": negative}
    else:
        raise HTTPException(status_code=404, detail="Examples not available")


# Run the app with: uvicorn app:app --reload
