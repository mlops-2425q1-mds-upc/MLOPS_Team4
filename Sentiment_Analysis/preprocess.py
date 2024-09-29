from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PARAMS_DIR

import pandas as pd
import re
import yaml
from loguru import logger
import contractions

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

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

## Clean sentences
# extracted from https://pub.aimind.so/a-comprehensive-guide-to-text-preprocessing-for-twitter-data-getting-ready-for-sentiment-analysis-e7f91cd03671

# Lowercase
df["text"] = df["text"].apply(lambda x: x.lower())

# Removing punctuation
import string

df["text"] = df["text"].apply(
    lambda x: x.translate(str.maketrans("", "", string.punctuation))
)

# Removing numbers
df["text"] = df["text"].apply(lambda x: re.sub(r"\d+", "", x))

# Removing extra spaces
df["text"] = df["text"].apply(lambda x: " ".join(x.split()))

# Replacing repetitions of punctuation
df["text"] = df["text"].apply(lambda x: re.sub(r"(\W)\1+", r"\1", x))

# Removing special characters
df["text"] = df["text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))

# Remove contractions from the "text" column
df["text"] = df["text"].apply(lambda x: contractions.fix(x))

logger.info("All sentences have been cleaned")

# Put target variable at the 1rst position
df = df.reindex(["positive", "text"], axis=1)

# Lemmarization, if it is indicated by the user
with open(PARAMS_DIR, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["preprocessing"]
    except yaml.YAMLError as exc:
        print(exc)

# Function to perform Lemmatization on a text
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}


def lemmatize_text(text):
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


if params["lemmarize"] == True:

    # Tokenize
    try:
        df["tokens"] = df["text"].apply(lambda x: word_tokenize(x))
    except LookupError:
        import nltk

        nltk.download("punkt_tab")
        df["tokens"] = df["text"].apply(lambda x: word_tokenize(x))

    try:
        # Apply Lemmatization to the 'tokens' column
        df["tokens"] = df["tokens"].apply(lemmatize_text)
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng")
        df["lemm_text"] = df["tokens"].apply(lemmatize_text)

    del df["tokens"]
    logger.info("Lemmarizer is appliyed to all sentences")

# save file
df.to_csv(str(PROCESSED_DATA_DIR) + "/" + "preprocessed_dataset.csv", index=False)
