import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import api


@pytest.fixture
def client():
    from contextlib import asynccontextmanager

    # Define a mock lifespan function to bypass actual loading
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    # Patch the app's lifespan function
    api.app.router.lifespan_context = mock_lifespan

    # Initialize TestClient with the patched app
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_globals():
    """
    Reset global variables before each test to ensure a clean state.
    """
    api.model = None  # Mocked model
    api.MODEL_INFO = {"metrics": "Test metrics"}  # Mocked model info
    api.DATASET = {
        "positive": set(["Positive example"]),
        "negative": set(["Negative example"]),
    }  # Mocked dataset


def test_predict_sentiment(client):
    """
    Test the /predict endpoint with valid input data.
    """
    input_data = {"tweets": ["I love this!", "I hate this!"]}

    with patch("api.predict_sentiment") as mock_predict:
        # Define side effects based on input tweets
        def side_effect(tweet, model):
            if "love" in tweet:
                return "Positive"
            elif "hate" in tweet:
                return "Negative"
            else:
                return "Neutral"

        mock_predict.side_effect = side_effect

        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        predictions = response.json()
        assert predictions["I love this!"] == "Positive"
        assert predictions["I hate this!"] == "Negative"


def test_predict_empty_input(client):
    """
    Test the /predict endpoint with empty input data.
    """
    input_data = {"tweets": []}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Text input is required"


def test_predict_invalid_input(client):
    """
    Test the /predict endpoint with invalid input format.
    """
    input_data = {"tweets": "Not a list"}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422
    data = response.json()
    assert "Invalid input format" in data["detail"]


def test_model_info(client):
    """
    Test the /model_info endpoint when model information is available.
    """
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data == {"model_info": {"metrics": "Test metrics"}}


def test_model_info_not_available(client):
    """
    Test the /model_info endpoint when model information is not available.
    """
    api.MODEL_INFO = {}  # Clear model info
    response = client.get("/model_info")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Model information not available"


def test_random_example(client):
    """
    Test the /random_example endpoint with available examples.
    """
    response = client.get("/random_example")
    assert response.status_code == 200
    data = response.json()
    assert data["positive_example"] == "Positive example"
    assert data["negative_example"] == "Negative example"


def test_random_example_empty_dataset(client):
    """
    Test the /random_example endpoint when the dataset is empty.
    """
    api.DATASET = {"positive": set(), "negative": set()}
    response = client.get("/random_example")
    assert response.status_code == 200
    data = response.json()
    assert data["positive_example"] == "test2"
    assert data["negative_example"] == "test1"
