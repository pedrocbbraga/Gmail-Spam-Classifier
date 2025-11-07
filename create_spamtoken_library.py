import json
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

THRESHOLD = 10 # MODIFY AS NEEDED!!!!

def load_dataset(jsonl_path: str):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_spamtoken_library(data: list) -> CountVectorizer:
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(data) # email, token matrix

    word_counts = np.asarray(X.sum(axis=0)).flatten()

    spammy_tokens = []
    for word, idx in vectorizer.vocabulary_.items():
        if word_counts[idx] > THRESHOLD:
            spammy_tokens.append(word)

    return spammy_tokens