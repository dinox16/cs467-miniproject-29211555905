import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "you", "your", "i", "me", "my", "we", "our", "they", "their", "he", "she", "his", "her",
    "can", "will", "would", "could", "should", "may", "might", "do", "does", "did", "has", "have", "had",
    "very", "so", "too", "also", "just", "some", "many", "much", "more", "most", "any", "all",
    "near", "up", "out", "into", "over", "as", "not", "no", "if", "then", "than", "about", "like",
    "want", "wants", "love", "looking", "place", "places", "visit", "visitors", "people", "trip",
    "good", "nice", "great", "best", "little", "lot",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split(" ")


def remove_stop_words(tokens):
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def simple_stem(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "ous", "is")):
        return word[:-1]
    return word


def preprocess(text):
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    return [simple_stem(token) for token in tokens]


def build_item_text(row, tag_weight=2):
    tags = str(row["activities_tags"]).replace("|", " ")
    parts = [str(row["description"])] + [tags] * tag_weight
    return " ".join(parts)


class NLPPipeline:
    def __init__(self, tag_weight=2):
        self.tag_weight = tag_weight
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess, lowercase=False, token_pattern=None)
        self.item_ids = []
        self.item_matrix = None
        self.feature_names = None

    def fit(self, items):
        texts = [build_item_text(row, self.tag_weight) for _, row in items.iterrows()]
        self.item_ids = list(items["item_id"])
        self.item_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self

    def transform_text(self, text):
        return self.vectorizer.transform([text])

    def query_is_known(self, text):
        return self.transform_text(text).nnz > 0

    def similarity_to_items(self, vector):
        vector = np.asarray(vector.todense()) if hasattr(vector, "todense") else np.asarray(vector)
        vector = vector.reshape(1, -1)
        if not np.any(vector):
            return np.zeros(len(self.item_ids))
        return cosine_similarity(vector, self.item_matrix)[0]

    def similarity_between_items(self, item_index):
        return cosine_similarity(self.item_matrix[item_index], self.item_matrix)[0]

    def build_profile(self, item_indexes, weights):
        weights = np.asarray(weights, dtype=float).reshape(-1, 1)
        dense = self.item_matrix[item_indexes].toarray()
        return (dense * weights).sum(axis=0)

    def matching_keywords(self, vector, item_index, top_n=3):
        vector = np.asarray(vector.todense()).ravel() if hasattr(vector, "todense") else np.asarray(vector).ravel()
        item_vector = self.item_matrix[item_index].toarray().ravel()
        overlap = vector * item_vector
        order = np.argsort(-overlap)
        keywords = [self.feature_names[i] for i in order[:top_n] if overlap[i] > 0]
        return keywords


def load_items(path):
    return pd.read_csv(path)
