# lstm_training.py
# pylint: disable=all
"""
LSTM Model Training Script
This script trains an LSTM model for sentiment analysis using preprocessed data.
"""
import pickle

import dagshub
import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from codecarbon import EmissionsTracker
from config import MLFLOW_TRACKING_URI
from config import MODELS_DIR
from config import PARAMS_DIR
from config import PROCESSED_DATA_DIR
from data_preprocessing import clean_text
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.callbacks import (  # type: ignore # type: ignore
    ModelCheckpoint,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    Dense,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    Dropout,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    Embedding,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    Input,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import (  # type: ignore
    Model,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import (  # type: ignore
    Adam,
)  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.preprocessing.sequence import (  # type: ignore
    pad_sequences,
)  # pylint: disable=import-error,no-name-in-module


def train_lstm_model():
    """
    Train an LSTM model for sentiment analysis using preprocessed data.

    This function reads parameters from a YAML file, sets up MLflow tracking,
    initializes DagsHub integration, loads the preprocessed data, and builds
    an LSTM model based on the extracted parameters. The model is trained, and
    various metrics are calculated and logged using MLflow. The function also
    saves accuracy and loss plots, as well as the confusion matrix and classification
    report as artifacts.

    Key Steps:
    - Load training parameters from a YAML file.
    - Set up directories for saving models and results.
    - Train an LSTM model with early stopping and model checkpointing.
    - Evaluate the model on test data and log metrics and plots using MLflow.
    - Save the final trained model for future use.
    - Demonstrate an example prediction using the trained model.

    Raises:
        KeyError: If 'lstm_train' key is not found in the parameter file.
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

    # Start CodeCarbon
    if lstm_params["track_emissions"] == True:
        EMISSIONS_TRACKER = EmissionsTracker(
            project_name="Team4",
            # experiment_id
            experiment_name="training",
            output_file="model.csv",
            output_dir="./emissions/",
            save_to_file=True,
            measure_power_secs=5,
        )
        print("CodeCarbon correctly configured...")
        EMISSIONS_TRACKER.start()

    # Set up MLflow tracking
    if lstm_params["upload_experiment"] == True:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Initialize DagsHub
        dagshub.init(
            repo_owner="daniel.cantabella.cantabella",
            repo_name="MLOPS_Team4",
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

    # save the split
    np.savez(PROCESSED_DATA_DIR / "train.npz", X=x_train, y=y_train)
    np.savez(PROCESSED_DATA_DIR / "test.npz", X=x_test, y=y_test)

    # Build the LSTM model
    inputs = Input(shape=(max_len,))
    embedding_layer = Embedding(
        input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_len
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

    # Adjusted EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=1,
        restore_best_weights=True,
    )

    # ModelCheckpoint callback
    model_checkpoint_path = MODELS_DIR / f"{model_name}_final.keras"
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
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Save metrics to a files
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

    # Save accuracy plot
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

    # Save loss plot
    loss_plot_path = plots_dir / "lstm_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # Confusion matrix plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("LSTM Model Confusion Matrix")

    # Save confusion matrix plot
    cm_plot_path = plots_dir / "lstm_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()

    # Save classification report to a text file and log as artifact
    report_path = reports_dir / "lstm_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LSTM Model Classification Report:\n")
        f.write(report)

    # Save the final model
    final_model_path = MODELS_DIR / f"{model_name}_final.keras"
    model.save(final_model_path)

    # Log metrics, artifacts and model to MLflow
    if lstm_params["upload_experiment"] == True:
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

    if lstm_params["track_emissions"]:
        EMISSIONS_TRACKER.stop()


if __name__ == "__main__":
    train_lstm_model()
