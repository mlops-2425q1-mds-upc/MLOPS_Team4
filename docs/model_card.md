---
license: mit
datasets:
- stanfordnlp/sentiment140
language:
- en
metrics:
- accuracy
- recall
- precision
- f1
- confusion_matrix
base_model:
- LSTM
pipeline_tag: text-classification
library_name: keras
tags:
- sentiment analysis
- twitter
- text classification
---
# Model Card for Sentiment Analysis Models

<!-- Provide a quick summary of what the model is/does. -->

This model card documents a Long Short-Term Memory (LSTM) neural network model aimed to predict the sentiment—positive or negative—of tweets based on their text content.


## Model Details

### Model Description

The LSTM Neural Network was chosen for its ability to handle sequential data and capture temporal dependencies in text, enhancing context understanding in sentiment analysis.

Model Sources
* **Repository**: MLOPS_Team4
* **Parameters**: Detailed in [params.yaml](../params.yaml) under **lstm_train** tag.


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

These models can be directly used for sentiment classification on Twitter data or similar textual datasets. Users can input tweet text to obtain sentiment predictions.


### Downstream Use 

The model can be integrated into larger systems for social media monitoring, customer feedback analysis, or any application requiring sentiment analysis.

### Out-of-Scope Use

This model is not intended for real-time streaming data analysis without further optimization. It may not perform well on languages other than English or highly specialized jargon.


## Bias, Risks, and Limitations

Since LSTMs process text sequentially, too long or too complex text could lead to diminished performance as the model may lose information over extended sequences.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

To ensure optimal performance of the LSTM model, it is recommended not to use excessively long texts for inference. 

## How to Get Started with the Model

Use the code below to get started with the model.

```python 
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('path_to_lstm_model.keras')

# Load tokenizer
with open('path_to_tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Prepare the text
text = "Sample tweet text"
cleaned_text = clean_text(text)
sequence = tokenizer.texts_to_sequences([cleaned_text])
padded = pad_sequences(sequence, maxlen=80, padding='post')

# Predict sentiment
pred_prob = model.predict(padded)
sentiment = 'Positive' if pred_prob[0][0] > 0.5 else 'Negative'

```



## Training Details

### Training Data

**Dataset**: Sentiment140 Twitter dataset with 1.6 million labeled tweets.

**Preprocessing**: Extensive text cleaning including removal of URLs, mentions, hashtags, numbers, punctuation, lowercasing, tokenization, stopword removal, and stemming.

- See dataset card [here](datset_card.md).

- See dataset [here](MLOps_Team4/data/processed).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing 

Consistent preprocessing steps were applied to both models to ensure data uniformity and comparability.

#### Training Hyperparameters

**LSTM Model**:
- Max Vocabulary Size: 10,000
- Max Sequence Length: 80
- Embedding Dimension: 64
- LSTM Units: 32
- Batch Size: 256
- Number of Epochs: 50
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Binary cross-entropy


For detailed hyperparameters, refer to [params.yaml](../params.yaml)



#### Speeds, Sizes, Times 
 * Training speed: 16.8 min
 * Model size: 16.7 MB


<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

## Evaluation

### Testing Data & Metrics

#### Testing Data

The model was evaluated on a test set comprising 30% of the original dataset to assess their performance on unseen data.




#### Metrics

The model was evaluated using the following metrics:

* **Accuracy**: Measures the proportion of correct predictions.
* **Precision**: Indicates the proportion of positive identifications that were correct.
* **Recall**: Measures the proportion of actual positives that were correctly identified.
* **F1 Score**: Harmonic mean of precision and recall.
* **Confusion Matrix**: Provides a detailed breakdown of true vs. predicted classes.


### Results

**LSTM Model:**

* **Accuracy**: 77.64%
* **Precision**: 75.63%
* **Recall**: 81.68%
* **F1 Score**: 78.54%

#### Summary

The LSTM model provides results that can assist us in our task of predicting whether a text is positive or negative in a sentiment analysis setting. Additionally, the training times are feasible for our project, and the model size is 16.7 MB, which will facilitate the loading and deployment of the model in production.


## Environmental Impact (TBD in Milestone 3)

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions have been estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** TBD
- **Hours used:** TBD
- **Cloud Provider:** AWS
- **Carbon Emitted:** TBD

## Technical Specifications

### Model Architecture and Objective

LSTM Neural Network consists of a recurrent neural network capable of capturing sequential dependencies in text data for sentiment classification.

### Compute Infrastructure

Private Infrastructure


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@article{article,
author = {Hochreiter, Sepp and Schmidhuber, Jürgen},
year = {1997},
month = {12},
pages = {1735-80},
title = {Long Short-term Memory},
volume = {9},
journal = {Neural computation},
doi = {10.1162/neco.1997.9.8.1735}
}



## Model Card Authors 


daniel.cantabella.cantabella@estudiantat.upc.edu,
dinara.kurmangaliyeva@estudiantat.upc.edu,
umut.ekin.gezer@estudiantat.upc.edu,
victor.garcia.pizarro@estudiantat.upc.edu


## Model Card Contact

daniel.cantabella.cantabella@estudiantat.upc.edu,
dinara.kurmangaliyeva@estudiantat.upc.edu,
umut.ekin.gezer@estudiantat.upc.edu,
victor.garcia.pizarro@estudiantat.upc.edu