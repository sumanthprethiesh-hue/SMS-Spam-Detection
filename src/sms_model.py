"""Data and model helpers for SMS spam classification."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_dataset(csv_path: str | Path) -> tuple[list[str], list[str]]:
    """Load common SMS CSV formats and normalize labels to ham/spam."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Add your labeled CSV there first; "
            "see README.md for the required columns (label,message or v1,v2)."
        )
    frame = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")

    if {"label", "message"}.issubset(frame.columns):
        label_column, message_column = "label", "message"
    elif {"v1", "v2"}.issubset(frame.columns):
        label_column, message_column = "v1", "v2"
    else:
        raise ValueError(
            "CSV must have 'label' and 'message' columns, or UCI-style 'v1' and 'v2' columns."
        )

    data = frame[[label_column, message_column]].dropna()
    label_map = {"ham": "ham", "spam": "spam", "0": "ham", "1": "spam"}
    labels = data[label_column].astype(str).str.strip().str.lower().map(label_map)
    messages = data[message_column].astype(str).str.strip()
    valid = labels.notna() & messages.ne("")
    labels, messages = labels[valid], messages[valid]

    if labels.nunique() < 2:
        raise ValueError("The dataset must contain examples of both ham and spam.")
    return messages.tolist(), labels.tolist()


def build_model() -> Pipeline:
    """Create a TF-IDF + Multinomial Naive Bayes text classification pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )
