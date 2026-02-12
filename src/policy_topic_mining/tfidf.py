from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _identity(x):
    return x


def compute_tfidf(tokens_list: list[list[str]], top_n: int = 40) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
    )
    tfidf_matrix = vectorizer.fit_transform(tokens_list)
    vocab = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:top_n]
    return pd.DataFrame(top_pairs, columns=["term", "tfidf_score"])


def tokens_to_text(tokens_list: Iterable[list[str]]) -> list[str]:
    return [" ".join(tokens) for tokens in tokens_list]


def term_frequencies(tokens_list: Iterable[list[str]], top_n: int = 40) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for tokens in tokens_list:
        counter.update(tokens)
    top = counter.most_common(top_n)
    return pd.DataFrame(top, columns=["term", "count"])
