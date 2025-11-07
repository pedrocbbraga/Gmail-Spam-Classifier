import json
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

THRESHOLD = 15 # MODIFY AS NEEDED!!!!
OUT_CSV = "spamtoken_library.csv"

def load_dataset(jsonl_path: str):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_dataset("spamtoken_dataset.jsonl")

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
    with open(OUT_CSV, "w") as f:
        f.write("token\n")
        for token in spammy_tokens:
            f.write(f"{token}\n")

create_spamtoken_library(data)