# lstm_training.py
"""
LSTM Model Training Script
This script trains an LSTM model for sentiment analysis using preprocessed data.
It also includes model testing steps using Deepchecks.
"""
import pickle

import dagshub
import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from codecarbon import EmissionsTracker
from config import (
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PARAMS_DIR,
    PROCESSED_DATA_DIR,
)
from data_preprocessing import clean_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Updated Deepchecks imports to avoid deprecation warnings
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation


def train_lstm_model():
    """
    Train an LSTM model for sentiment analysis using preprocessed data.
    """
    # Load parameters from YAML
    with open(PARAMS_DIR, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
        lstm_params = params.get("lstm_train")

    if lstm_params is None:
        raise KeyError("'lstm_train' key not found in params.yaml")

    # Extract parameters
    max_vocab_size = lstm_params["max_vocab_size"]
    max_len = lstm_params["max_len"]
    embedding_dim = lstm_params["embedding_dim"]
    lstm_units = lstm_params["lstm_units"]
    batch_size = lstm_params["batch_size"]
    num_epochs = lstm_params["num_epochs"]
    learning_rate = lstm_params["learning_rate"]
    model_name = lstm_params["model_name"]
    random_state = lstm_params["random_state"]
    test_size = lstm_params["test_size"]
    run_deepchecks = lstm_params.get("run_deepchecks", False)

    # Start CodeCarbon
    if lstm_params["track_emissions"]:
        EMISSIONS_TRACKER = EmissionsTracker(
            project_name="Team4",
            experiment_name="training",
            output_file="model.csv",
            output_dir="./emissions/",
            save_to_file=True,
            measure_power_secs=5,
        )
        print("CodeCarbon correctly configured...")
        EMISSIONS_TRACKER.start()

    # Set up MLflow tracking
    if lstm_params["upload_experiment"]:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Initialize DagsHub
        dagshub.init(
            repo_owner="your_username",
            repo_name="your_repo",
            mlflow=True,
        )

        with mlflow.start_run(run_name=f"{model_name}_LSTM"):
            # Log parameters
            mlflow.log_params(
                {
                    "max_vocab_size": max_vocab_size,
                    "max_len": max_len,
                    "embedding_dim": embedding_dim,
                    "lstm_units": lstm_units,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "model_name": model_name,
                    "random_state": random_state,
                    "test_size": test_size,
                }
            )

    # Create directories for saving models and results
    results_dir = MODELS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    # Subfolders
    plots_dir = results_dir / "plots"
    metrics_dir = results_dir / "metrics"
    reports_dir = results_dir / "reports"
    for directory in [plots_dir, metrics_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    data_path = PROCESSED_DATA_DIR / "X_y_data_lstm.npz"
    tokenizer_path = PROCESSED_DATA_DIR / "tokenizer.pickle"

    loaded_data = np.load(data_path)
    x = loaded_data["X"]
    y = loaded_data["y"]

    # Load tokenizer
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    vocab_size = len(tokenizer.word_index) + 1
    print(f"LSTM Vocabulary Size: {vocab_size}")

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Save the split
    np.savez(PROCESSED_DATA_DIR / "train.npz", X=x_train, y=y_train)
    np.savez(PROCESSED_DATA_DIR / "test.npz", X=x_test, y=y_test)

    # Build the LSTM model
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(
        input_dim=max_vocab_size, output_dim=embedding_dim
    )(inputs)
    lstm_layer = LSTM(lstm_units)(embedding_layer)
    norm_layer = BatchNormalization()(lstm_layer)
    dense_layer = Dense(32, activation="relu")(norm_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    outputs = Dense(1, activation="sigmoid")(dropout_layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=1,
        restore_best_weights=True,
    )

    # ModelCheckpoint callback
    model_checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_path, save_best_only=True, monitor="val_loss", mode="min"
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
    )

    # Predict on test data
    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Save metrics to a file
    metrics_path = metrics_dir / "lstm_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")

    # Classification report
    report = classification_report(y_test, y_pred)
    print("LSTM Model Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save accuracy and loss
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("LSTM Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    accuracy_plot_path = plots_dir / "lstm_accuracy.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("LSTM Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    loss_plot_path = plots_dir / "lstm_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # Confusion matrix plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("LSTM Model Confusion Matrix")
    cm_plot_path = plots_dir / "lstm_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()

    # Save classification report to a text file
    report_path = reports_dir / "lstm_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LSTM Model Classification Report:\n")
        f.write(report)

    # Save the final model
    final_model_path = MODELS_DIR / f"{model_name}_final.keras"
    model.save(final_model_path)

    # Run Deepchecks model evaluation suite if enabled
    if run_deepchecks:
        print("Running Deepchecks model evaluation suite...")
        # Convert x_train and x_test to DataFrames
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)

        # Specify that there are no categorical features
        cat_features = []

        # Create Deepchecks Datasets
        train_dataset = Dataset(
            x_train_df,
            label=y_train,
            cat_features=cat_features,
            dataset_name='Train Dataset'
        )
        test_dataset = Dataset(
            x_test_df,
            label=y_test,
            cat_features=cat_features,
            dataset_name='Test Dataset'
        )

        # Set task type after initialization
        train_dataset.task_type = 'classification'
        test_dataset.task_type = 'classification'

        # Create a wrapper for the Keras model
        class KerasClassifierWrapper:
            def __init__(self, model):
                self.model = model
                self.classes_ = np.array([0, 1])

            def predict(self, x):
                proba = self.model.predict(x)
                return (proba > 0.5).astype('int32').flatten()

            def predict_proba(self, x):
                proba = self.model.predict(x).flatten()
                proba_stacked = np.vstack((1 - proba, proba)).T
                return proba_stacked

        wrapped_model = KerasClassifierWrapper(model)

        # Run Deepchecks model evaluation suite
        suite = model_evaluation()
        suite_result = suite.run(
            train_dataset=train_dataset, test_dataset=test_dataset, model=wrapped_model
        )

        # Save the suite result to an HTML file
        deepchecks_report_path = reports_dir / 'deepchecks_model_evaluation.html'
        suite_result.save_as_html(str(deepchecks_report_path))
        print(f"Deepchecks report saved at {deepchecks_report_path}")

        # If using MLflow, log the Deepchecks report as an artifact
        if lstm_params["upload_experiment"]:
            mlflow.log_artifact(str(deepchecks_report_path), artifact_path="reports")

    # Log metrics, artifacts, and model to MLflow
    if lstm_params["upload_experiment"]:
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )
        mlflow.log_artifact(str(loss_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(accuracy_plot_path), artifact_path="plots")

        # Log the model to MLflow
        mlflow.keras.log_model(model, artifact_path="model_LSTM")

    # Stop CodeCarbon tracker if enabled
    if lstm_params["track_emissions"]:
        EMISSIONS_TRACKER.stop()


if __name__ == "__main__":
    train_lstm_model()
