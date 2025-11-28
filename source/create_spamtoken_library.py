"""

create_spamtoken_library.py: This module creates the input spam token library for create_input_dataset.py

It gets retrieves the email bodies from the spamtoken dataset and processes the data to extract spammy tokens.

Inputs: spam email JSONL dataset containing email bodies
Output: spam token library in CSV format

Usage: e.g., from the root Gmail-Spam-Classifier directory, python3 source/create_spamtoken_library.py

"""

import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from config import DATA_PROCESSED, DATA_RAW

THRESHOLD = 15 # MODIFY AS NEEDED!!!!
OUT_FILE = DATA_PROCESSED / "spamtoken_library.csv"
IN_FILE = DATA_RAW / "spamtoken_dataset.jsonl"

def load_dataset(jsonl_path: str):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_dataset(IN_FILE)

def create_spamtoken_library(data: list) -> CountVectorizer:
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(data) # email, token matrix

    word_counts = np.asarray(X.sum(axis=0)).flatten()

    # Create spammy token list
    spammy_tokens = []
    for word, idx in vectorizer.vocabulary_.items():
        if word_counts[idx] > THRESHOLD:
            spammy_tokens.append(word)

    # Save to CSV
    with open(OUT_FILE, "w") as f:
        f.write("token\n")
        for token in spammy_tokens:
            f.write(f"{token}\n")

create_spamtoken_library(data)