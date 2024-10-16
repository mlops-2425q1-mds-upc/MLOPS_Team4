# logistic_regression_training.py
import os
import pickle
import re

import dagshub  # Added import
import matplotlib.pyplot as plt
import mlflow.sklearn
import nltk
import numpy as np
import seaborn as sns
import yaml
from config import MLFLOW_TRACKING_URI
from config import PARAMS_DIR
from config import PROCESSED_DATA_DIR
from config import REPORTS_DIR
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

# Import config from the same directory

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


def train_logistic_regression_model():
    # Load parameters from YAML
    with open(PARAMS_DIR, "r") as file:
        params = yaml.safe_load(file)
        logreg_params = params.get("logreg_train")

    if logreg_params is None:
        raise KeyError("'logreg_train' key not found in params.yaml")

    # Extract parameters
    max_features = logreg_params["max_features"]
    test_size = logreg_params["test_size"]
    random_state = logreg_params["random_state"]
    model_name = logreg_params["model_name"]
    solver = logreg_params["solver"]
    max_iter = logreg_params["max_iter"]

    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Initialize DagsHub
    dagshub.init(
        repo_owner="daniel.cantabella.cantabella", repo_name="MLOPS_Team4", mlflow=True
    )

    with mlflow.start_run(run_name=f"{model_name}_LogisticRegression"):
        # Log parameters
        mlflow.log_params(
            {
                "max_features": max_features,
                "test_size": test_size,
                "random_state": random_state,
                "model_name": model_name,
                "solver": solver,
                "max_iter": max_iter,
            }
        )

        # Create directories for saving models and results
        results_dir = REPORTS_DIR / "logistic_regression_model"
        results_dir.mkdir(parents=True, exist_ok=True)
        # Subfolders
        model_dir = results_dir / "models"
        plots_dir = results_dir / "plots"
        metrics_dir = results_dir / "metrics"
        reports_dir = results_dir / "reports"
        for directory in [model_dir, plots_dir, metrics_dir, reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load preprocessed data
        data_path = PROCESSED_DATA_DIR / "X_y_data_logreg.npz"
        vectorizer_path = PROCESSED_DATA_DIR / "tfidf_vectorizer.pickle"

        with np.load(data_path, allow_pickle=True) as data:
            X_data = data["X"]
            y = data["y"]
            indices = data["indices"]
            indptr = data["indptr"]
            shape = data["shape"]

        from scipy.sparse import csr_matrix

        X_tfidf = csr_matrix((X_data, indices, indptr), shape)

        # Load TF-IDF vectorizer
        with open(vectorizer_path, "rb") as handle:
            tfidf_vectorizer = pickle.load(handle)

        # Split the data
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=test_size, random_state=random_state
        )

        # Initialize Logistic Regression model
        log_reg_model = LogisticRegression(max_iter=max_iter, solver=solver, n_jobs=-1)

        # Train the model
        log_reg_model.fit(X_train_tfidf, y_train)

        # Make predictions
        y_pred = log_reg_model.predict(X_test_tfidf)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

        # Save metrics to a file
        metrics_path = metrics_dir / "logreg_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")

        # Log metrics file as artifact
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        # Classification report
        report = classification_report(y_test, y_pred)
        print("Logistic Regression Classification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix plot
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Logistic Regression Confusion Matrix")
        cm_plot_path = plots_dir / "logreg_confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()

        mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")

        # Save classification report to a text file and log as artifact
        report_path = reports_dir / "logreg_classification_report.txt"
        with open(report_path, "w") as f:
            f.write("Logistic Regression Classification Report:\n")
            f.write(report)

        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # Save the model
        final_model_path = model_dir / f"{model_name}_LogisticRegression.pkl"
        with open(final_model_path, "wb") as model_file:
            pickle.dump(log_reg_model, model_file)

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            log_reg_model, artifact_path="model_LogisticRegression"
        )

        # Function to predict sentiment using the trained model
        def predict_sentiment_logreg(text):
            # Clean the text
            cleaned = clean_text(text)
            # Transform text
            X_tfidf_new = tfidf_vectorizer.transform([cleaned])
            # Load model
            with open(final_model_path, "rb") as model_file:
                loaded_model = pickle.load(model_file)
            # Predict
            pred_class = loaded_model.predict(X_tfidf_new)
            sentiment = "Positive" if pred_class[0] == 1 else "Negative"
            return sentiment

        # Example prediction
        sample_text = "I absolutely love this product! It works wonders."
        print(f"Sample Text: {sample_text}")
        print(
            f"Predicted Sentiment (Logistic Regression): {predict_sentiment_logreg(sample_text)}"
        )


if __name__ == "__main__":
    train_logistic_regression_model()
