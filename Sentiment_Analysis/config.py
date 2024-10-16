"""
config.py

This module is responsible for loading environment variables and defining
the project's directory structure for a sentiment analysis application.

It utilizes the `dotenv` library to load environment variables from a
`.env` file if it exists, ensuring that sensitive configurations
are not hard-coded into the source code.

Key functionalities:
- Loads environment variables from a `.env` file.
- Defines and logs the project root directory.
- Sets up paths for various directories used throughout the project,
  including data storage, models, reports, and figures.
- Configures logging with `loguru`, providing a more user-friendly
  output when using tqdm.

Paths defined in this module:
- `PROJ_ROOT`: The root directory of the project.
- `DATA_DIR`: Directory for all data-related files.
- `RAW_DATA_DIR`: Subdirectory for raw data files.
- `INTERIM_DATA_DIR`: Subdirectory for interim data files.
- `PROCESSED_DATA_DIR`: Subdirectory for processed data files.
- `ENCODED_DATA_DIR`: Subdirectory for encoded data files.
- `MODELS_DIR`: Directory for storing trained models.
- `REPORTS_DIR`: Directory for storing reports generated by the application.
- `FIGURES_DIR`: Subdirectory within reports for storing figures.
- `MLFLOW_TRACKING_URI`: URI for tracking experiments with MLflow.

Usage:
    Import this module to access project paths and configurations:

    from Sentiment_Analysis.config import PROJ_ROOT, DATA_DIR
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PARAMS_DIR = PROJ_ROOT / "params.yaml"  # Updated to point to the project root

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENCODED_DATA_DIR = DATA_DIR / "encoded"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/daniel.cantabella.cantabella/MLOPS_Team4.mlflow"
)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
