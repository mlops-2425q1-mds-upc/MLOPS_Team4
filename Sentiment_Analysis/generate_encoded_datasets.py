import yaml
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import mlflow
import sys
import dagshub
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

def generate_encoded_datasets():
    
    with open(os.path.join(str(PROJ_ROOT), PARAMS_DIR), "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["generate-encoded-datasets"]
        except yaml.YAMLError as exc:
            print(exc)
    
    data_dir = os.path.join(str(PROJ_ROOT), INTERIM_DATA_DIR)
    test_size = params['test_size'] 
    random_state = params['random_state']
    truncation = params['truncation']  
    padding = params['padding']  
    
    encoded_dir = os.path.join(str(PROJ_ROOT), ENCODED_DATA_DIR)
    os.makedirs(encoded_dir, exist_ok=True)
    train_dataset_path = os.path.join(encoded_dir, 'train_dataset.pt')
    test_dataset_path = os.path.join(encoded_dir, 'test_dataset.pt')
    
    
    df = pd.read_csv(data_dir + '/test_tweets.csv', sep=',')
    
    # Prepare input features and labels
    X = list(df['text'])
    y = list(df['label'])
    
    # Map labels to integers
    label_to_int = {'pos': 1, 'neu': 0, 'neg': 2}
    y = list(map(label_to_int.get, y))
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Tokenization
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(X_train, truncation=truncation, padding=padding, return_tensors='pt')
    test_encodings = tokenizer(X_test, truncation=truncation, padding=padding, return_tensors='pt')
    
    #Tensors
    train_dataset = TweetDataset(train_encodings, y_train)
    test_dataset = TweetDataset(test_encodings, y_test)
    torch.save(train_dataset, train_dataset_path)
    torch.save(test_dataset, test_dataset_path)

if __name__ == "__main__":
    generate_encoded_datasets()    