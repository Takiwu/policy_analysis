from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def _identity(x):
    return x


def compute_tfidf(tokens_list: list[list[str]], top_n: int = 40) -> pd.DataFrame:
    if not tokens_list:
        return pd.DataFrame(columns=["term", "tfidf_score"])
    if not any(tokens_list):
        return pd.DataFrame(columns=["term", "tfidf_score"])

    # 模仿论文中提到的 TfidfTransformer 流程：
    # 1) CountVectorizer 统计词频矩阵
    # 2) TfidfTransformer 计算 TF-IDF
    vectorizer = CountVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
    )
    count_matrix = vectorizer.fit_transform(tokens_list)

    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    tfidf_matrix = transformer.fit_transform(count_matrix)

    vocab = vectorizer.get_feature_names_out()

    # 使用“文档均值”并归一化为占比，得到与论文表格更接近的 0.0x 量级
    mean_scores = tfidf_matrix.mean(axis=0).A1
    total = float(mean_scores.sum())
    if total <= 0:
        return pd.DataFrame(columns=["term", "tfidf_score"])
    norm_scores = mean_scores / total

    pairs = sorted(zip(vocab, norm_scores), key=lambda x: x[1], reverse=True)
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
