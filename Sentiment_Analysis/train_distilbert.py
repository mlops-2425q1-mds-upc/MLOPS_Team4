import yaml
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import mlflow
import sys
import dagshub
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from config import RAW_DATA_DIR, INTERIM_DATA_DIR, PARAMS_DIR, PROJ_ROOT, ENCODED_DATA_DIR, MODELS_DIR, MLFLOW_TRACKING_URI

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_DistilBert():
    with open(os.path.join(str(PROJ_ROOT), PARAMS_DIR), "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["distilbert-train"]
        except yaml.YAMLError as exc:
            print(exc)
    
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    dagshub.init(repo_owner='daniel.cantabella.cantabella', repo_name='MLOPS_Team4', mlflow=True)

    os.environ["TOKENIZERS_PARALLELISM"] = params['tokenizers_parallelism']
    os.environ["HF_DAGSHUB_MODEL_NAME"] = params['model_name']
    
    # Extract parameters
    model_name = params['model_name']   
    output_dir = params['output_dir']  
    num_train_epochs = params['num_train_epochs']  
    per_device_train_batch_size = params['per_device_train_batch_size']  
    per_device_eval_batch_size = params['per_device_eval_batch_size']  
    learning_rate = params['learning_rate']
    logging_steps = params['logging_steps']
    evaluation_strategy = params['evaluation_strategy']
    save_strategy = params['save_strategy']
    load_best_model_at_end=params['load_best_model_at_end']  # Load the best model when finished training (default metric: "loss")
    metric_for_best_model=params['metric_for_best_model']  # Metric to use for the best model
    greater_is_better=params['greater_is_better']  # Indicates whether a higher score is better
    save_total_limit=params['save_total_limit']

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Log parameters to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
        mlflow.log_param("per_device_eval_batch_size", per_device_eval_batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("logging_steps", logging_steps)
        mlflow.log_param("evaluation_strategy", evaluation_strategy)
        mlflow.log_param("save_strategy", save_strategy)
        mlflow.log_param("load_best_model_at_end", load_best_model_at_end)
        mlflow.log_param("metric_for_best_model", metric_for_best_model)
        mlflow.log_param("greater_is_better", greater_is_better)
        mlflow.log_param("save_total_limit", save_total_limit)
        
        # Load datasets
        encoded_dir = os.path.join(str(PROJ_ROOT), ENCODED_DATA_DIR)
        os.makedirs(encoded_dir, exist_ok=True)
        train_dataset_path = os.path.join(encoded_dir, 'train_dataset.pt')
        test_dataset_path = os.path.join(encoded_dir, 'test_dataset.pt')

        train_dataset = torch.load(train_dataset_path)
        test_dataset = torch.load(test_dataset_path)

        # Log dataset version (assuming you're using DVC)
        dataset_version = os.popen('dvc version').read().strip()  # Get DVC version
        mlflow.log_param("dataset_version", dataset_version)

        # Log dataset details as parameters
        mlflow.log_param("train_dataset_size", len(train_dataset))
        mlflow.log_param("test_dataset_size", len(test_dataset))
        mlflow.log_param("train_dataset_path", train_dataset_path)
        mlflow.log_param("test_dataset_path", test_dataset_path)

        # Load model
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(str(MODELS_DIR), 'checkpoints', model_name),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_dir= os.path.join(str(MODELS_DIR), 'logs', model_name),
            logging_steps=logging_steps,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,  # Load the best model when finished training (default metric: "loss")
            metric_for_best_model=metric_for_best_model,  # Metric to use for the best model
            greater_is_better=greater_is_better,  # Indicates whether a higher score is better
            save_total_limit=save_total_limit,  # Keep only the last 3 checkpoints
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the test dataset
        metrics = trainer.evaluate()

        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Save the model and log it to MLflow
        trainer.save_model(os.path.join(MODELS_DIR, model_name))
        mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)

        # Log datasets to MLflow as artifacts
        mlflow.log_artifact(train_dataset_path, "train_dataset.pt")
        mlflow.log_artifact(test_dataset_path, "test_dataset.pt")

# Execute the training function
if __name__ == "__main__":
    train_DistilBert()
