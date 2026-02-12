from __future__ import annotations

# pyright: reportMissingImports=false

import logging
import importlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class LdaEvaluation:
    topic_count: int
    perplexity: float
    coherence: float


def build_corpus(tokens_list: list[list[str]]):
    try:
        module_name = "gen" + "sim" + ".corpora"
        corpora = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "gensim 未安装或与当前 Python 版本不兼容。请使用 Python 3.11/3.12 "
            "并安装 requirements.txt 中的依赖。"
        ) from exc
    dictionary = corpora.Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    return dictionary, corpus


def train_lda(
    corpus,
    dictionary,
    num_topics: int,
    alpha: float | str,
    eta: float,
    iterations: int,
    passes: int,
    random_state: int,
) -> object:
    try:
        module_name = "gen" + "sim" + ".models"
        models = importlib.import_module(module_name)
        LdaModel = models.LdaModel
    except ImportError as exc:
        raise ImportError(
            "gensim 未安装或与当前 Python 版本不兼容。请使用 Python 3.11/3.12 "
            "并安装 requirements.txt 中的依赖。"
        ) from exc
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        passes=passes,
    )


def resolve_alpha(alpha: float | str, num_topics: int) -> float | str:
    if isinstance(alpha, (int, float)):
        return float(alpha)
    if alpha == "50/K":
        return 50.0 / float(num_topics)
    return "auto"


def evaluate_topic_range(
    tokens_list: list[list[str]],
    start: int,
    end: int,
    alpha: float | str,
    eta: float,
    iterations: int,
    passes: int,
    random_state: int,
) -> tuple[list[LdaEvaluation], dict[int, object], object, list[list[tuple[int, int]]]]:
    try:
        module_name = "gen" + "sim" + ".models"
        models = importlib.import_module(module_name)
        CoherenceModel = models.CoherenceModel
    except ImportError as exc:
        raise ImportError(
            "gensim 未安装或与当前 Python 版本不兼容。请使用 Python 3.11/3.12 "
            "并安装 requirements.txt 中的依赖。"
        ) from exc
    dictionary, corpus = build_corpus(tokens_list)
    evaluations: list[LdaEvaluation] = []
    models: dict[int, LdaModel] = {}
    for k in range(start, end + 1):
        model = train_lda(
            corpus=corpus,
            dictionary=dictionary,
            num_topics=k,
            alpha=resolve_alpha(alpha, k),
            eta=eta,
            iterations=iterations,
            passes=passes,
            random_state=random_state,
        )
        models[k] = model
        perplexity = model.log_perplexity(corpus)
        coherence_model = CoherenceModel(model=model, texts=tokens_list, dictionary=dictionary, coherence="c_v")
        coherence = coherence_model.get_coherence()
        evaluations.append(LdaEvaluation(topic_count=k, perplexity=perplexity, coherence=coherence))
        LOGGER.info("k=%s perplexity=%.4f coherence=%.4f", k, perplexity, coherence)
    return evaluations, models, dictionary, corpus


def choose_best_k(evaluations: Iterable[LdaEvaluation]) -> int:
    items = list(evaluations)
    if not items:
        raise ValueError("No evaluations to choose from")
    items.sort(key=lambda x: (x.coherence, -x.perplexity), reverse=True)
    return items[0].topic_count


def extract_topic_words(model: object, topn: int) -> pd.DataFrame:
    rows = []
    for topic_id, words in model.show_topics(num_topics=-1, num_words=topn, formatted=False):
        for word, weight in words:
            rows.append({"topic": topic_id, "word": word, "weight": weight})
    return pd.DataFrame(rows)


def doc_topic_distribution(model: object, corpus) -> pd.DataFrame:
    rows = []
    for doc_id, doc_topics in enumerate(model.get_document_topics(corpus, minimum_probability=0)):
        row = {f"topic_{topic_id}": prob for topic_id, prob in doc_topics}
        row["doc_id"] = doc_id
        rows.append(row)
    return pd.DataFrame(rows)


def topic_strengths(doc_topic_df: pd.DataFrame) -> pd.DataFrame:
    topic_cols = [col for col in doc_topic_df.columns if col.startswith("topic_")]
    strengths = doc_topic_df[topic_cols].mean(axis=0).reset_index()
    strengths.columns = ["topic", "strength"]
    strengths["topic"] = strengths["topic"].str.replace("topic_", "", regex=False).astype(int)
    return strengths.sort_values("topic")
