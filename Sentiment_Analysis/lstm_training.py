# lstm_training.py

import os
import numpy as np
import pickle
import yaml
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import tensorflow as tf
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import dagshub
# Import config from the same directory
from config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    PARAMS_DIR,
    MLFLOW_TRACKING_URI,
)

nltk.download('stopwords')



def clean_text(text):
    # Remove URLs, mentions, hashtags, and numbers
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    # Tokenize, remove stopwords, and apply stemming
    tokens = text.split()
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def train_lstm_model():
    # Load parameters from YAML
    with open(PARAMS_DIR, 'r') as file:
        params = yaml.safe_load(file)
        lstm_params = params.get('lstm_train')

    if lstm_params is None:
        raise KeyError("'lstm_train' key not found in params.yaml")

    # Extract parameters
    max_vocab_size = lstm_params['max_vocab_size']
    max_len = lstm_params['max_len']
    embedding_dim = lstm_params['embedding_dim']
    lstm_units = lstm_params['lstm_units']
    batch_size = lstm_params['batch_size']
    num_epochs = lstm_params['num_epochs']
    learning_rate = lstm_params['learning_rate']
    model_name = lstm_params['model_name']
    random_state = lstm_params['random_state']
    test_size = lstm_params['test_size']

    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Initialize DagsHub
    dagshub.init(repo_owner='daniel.cantabella.cantabella', repo_name='MLOPS_Team4', mlflow=True)

    with mlflow.start_run(run_name=f"{model_name}_LSTM"):
        # Log parameters
        mlflow.log_params({
            'max_vocab_size': max_vocab_size,
            'max_len': max_len,
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'model_name': model_name,
            'random_state': random_state,
            'test_size': test_size
        })

        # Create directories for saving models and results
        results_dir = REPORTS_DIR / 'lstm_model'
        results_dir.mkdir(parents=True, exist_ok=True)
        # Subfolders
        model_dir = results_dir / 'models'
        plots_dir = results_dir / 'plots'
        metrics_dir = results_dir / 'metrics'
        reports_dir = results_dir / 'reports'
        for directory in [model_dir, plots_dir, metrics_dir, reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load preprocessed data
        data_path = PROCESSED_DATA_DIR / 'X_y_data_lstm.npz'
        tokenizer_path = PROCESSED_DATA_DIR / 'tokenizer.pickle'

        loaded_data = np.load(data_path)
        X = loaded_data['X']
        y = loaded_data['y']

        # Load tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        vocab_size = len(tokenizer.word_index) + 1
        print(f"LSTM Vocabulary Size: {vocab_size}")

        # Log vocabulary size
        mlflow.log_param('vocab_size', vocab_size)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Build the LSTM model
        inputs = Input(shape=(max_len,))
        embedding_layer = Embedding(
            input_dim=max_vocab_size,
            output_dim=embedding_dim,
            input_length=max_len
        )(inputs)
        lstm_layer = LSTM(lstm_units)(embedding_layer)
        norm_layer = BatchNormalization()(lstm_layer)
        dense_layer = Dense(32, activation='relu')(norm_layer)
        dropout_layer = Dropout(0.5)(dense_layer)
        outputs = Dense(1, activation='sigmoid')(dropout_layer)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Adjusted EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=3,
            verbose=1,
            restore_best_weights=True
        )

        # ModelCheckpoint callback
        model_checkpoint_path = model_dir / f'{model_name}_LSTM.keras'
        model_checkpoint = ModelCheckpoint(
            model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint]
        )

        # Predict on test data
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32")

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Log metrics to MLflow
        mlflow.log_metrics({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        })

        # Save metrics to a file
        metrics_path = metrics_dir / 'lstm_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")

        # Log metrics file as artifact
        mlflow.log_artifact(str(metrics_path), artifact_path='metrics')

        # Classification report
        report = classification_report(y_test, y_pred)
        print("LSTM Model Classification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot and save accuracy and loss
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('LSTM Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        # Save accuracy plot
        accuracy_plot_path = plots_dir / 'lstm_accuracy.png'
        plt.savefig(accuracy_plot_path)
        plt.close()

        mlflow.log_artifact(str(accuracy_plot_path), artifact_path='plots')

        plt.figure()
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        # Save loss plot
        loss_plot_path = plots_dir / 'lstm_loss.png'
        plt.savefig(loss_plot_path)
        plt.close()

        mlflow.log_artifact(str(loss_plot_path), artifact_path='plots')

        # Confusion matrix plot
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('LSTM Model Confusion Matrix')
        # Save confusion matrix plot
        cm_plot_path = plots_dir / 'lstm_confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()

        mlflow.log_artifact(str(cm_plot_path), artifact_path='plots')

        # Save classification report to a text file and log as artifact
        report_path = reports_dir / 'lstm_classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("LSTM Model Classification Report:\n")
            f.write(report)

        mlflow.log_artifact(str(report_path), artifact_path='reports')

        # Save the final model
        final_model_path = model_dir / f'{model_name}_LSTM_final.keras'
        model.save(final_model_path)

        # Log the model to MLflow
        mlflow.keras.log_model(model, artifact_path='model_LSTM')

        # Function to predict sentiment using the trained model
        def predict_sentiment(text):
            # Clean the text
            cleaned = clean_text(text)
            # Tokenize and pad
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=max_len, padding='post')
            # Load model
            loaded_model = tf.keras.models.load_model(final_model_path)
            # Predict
            pred_prob = loaded_model.predict(padded)
            pred_class = (pred_prob > 0.5).astype("int32")
            sentiment = 'Positive' if pred_class[0][0] == 1 else 'Negative'
            return sentiment

        # Example prediction
        sample_text = "I absolutely love this product! It works wonders."
        print(f"Sample Text: {sample_text}")
        print(f"Predicted Sentiment: {predict_sentiment(sample_text)}")


if __name__ == "__main__":
    train_lstm_model()
