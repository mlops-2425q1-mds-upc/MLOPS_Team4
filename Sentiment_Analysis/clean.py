# %%
"""
This module contains functions and operations for preprocessing text data
for sentiment analysis, including cleaning, tokenization, and lemmatization.
"""
import os
import re

import contractions
import nltk
import pandas as pd
import yaml
from config import INTERIM_DATA_DIR
from config import PARAMS_DIR
from config import RAW_DATA_DIR
from loguru import logger

nltk.download("stopwords")  # pylint: disable=wrong-import-position
nltk.download("wordnet")  # pylint: disable=wrong-import-position
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

# %%
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

# Load the dataset
cols = ["sentiment", "id", "date", "query_string", "user", "text"]

df = pd.read_csv(
    RAW_DATA_DIR / "training.1600000.processed.noemoticon.csv",
    header=None,
    names=cols,
    encoding="latin-1",
)

# Define target variable: Positive sentences are labelled as 1
df["positive"] = df["sentiment"].replace([0, 4], [0, 1])

# Drop unnecessary columns
df = df.drop(columns=["id", "query_string", "sentiment", "date", "user"], axis=1)

# Cleaning sentences
# Extracted from https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm  # pylint: disable=line-too-long
# From Arun Pandian R
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# Updated constant name to conform to UPPER_CASE naming style
TEXT_CLEANING_RE = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def preprocess(text, stem=False):
    """Preprocess the input text by cleaning and tokenizing it.

    Args:
        text (str): The text to preprocess.
        stem (bool): Flag to indicate if stemming should be applied.

    Returns:
        str: The cleaned and processed text.
    """
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


# Lemmatization, if it is indicated by the user
with open(
    PARAMS_DIR, "r", encoding="utf-8"
) as params_file:  # Specified encoding explicitly
    try:
        params = yaml.safe_load(params_file)
        params = params["clean"]
    except yaml.YAMLError as exc:
        print(exc)

# Use 'params["steam"]' instead of 'params["steam"]'
df.text = df.text.apply(lambda x: preprocess(x, stem=params["steam"]))

# Extra cleaning
# Extracted from:
# https://pub.aimind.so/a-comprehensive-guide-to-text-preprocessing-for-twitter-data-getting-ready-for-sentiment-analysis-e7f91cd03671

# Removing numbers
df["text"] = df["text"].apply(lambda x: re.sub(r"\d+", "", x))

# Remove contractions from the "text" column
df["text"] = df["text"].apply(contractions.fix)

logger.info("All sentences have been cleaned")

# Put target variable at the 1st position
df = df.reindex(["positive", "text"], axis=1)

# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# POS tag mapping dictionary
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}


def lemmatize_text(text):
    """Lemmatize the input text based on POS tagging.

    Args:
        text (str): The text to lemmatize.

    Returns:
        str: The lemmatized text.
    """
    # Get the POS tags for the words
    pos_tags = nltk.pos_tag(text)

    # Perform Lemmatization
    lemmatized_words = []
    for word, tag in pos_tags:
        # Map the POS tag to WordNet POS tag
        pos = wordnet_map.get(tag[0].upper(), wordnet.NOUN)
        # Lemmatize the word with the appropriate POS tag
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Add the lemmatized word to the list
        lemmatized_words.append(lemmatized_word)

    lemmatize_sentence = " ".join(lemmatized_words)
    return lemmatize_sentence


if params["lemmarize"]:  # Check for truthiness instead of using == True
    # Tokenize
    try:
        df["tokens"] = df["text"].apply(word_tokenize)  # Removed unnecessary lambda
    except LookupError:
        import nltk

        nltk.download("punkt_tab")
        df["tokens"] = df["text"].apply(word_tokenize)

    try:
        # Apply Lemmatization to the 'tokens' column
        df["tokens"] = df["tokens"].apply(lemmatize_text)  # Removed unnecessary lambda
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")
        df["lemm_text"] = df["tokens"].apply(lemmatize_text)

    del df["tokens"]
    logger.info("Lemmatizer is applied to all sentences")

# Save file
df.to_csv(str(INTERIM_DATA_DIR) + "/" + "clean_dataset.csv", index=False)

# %%
