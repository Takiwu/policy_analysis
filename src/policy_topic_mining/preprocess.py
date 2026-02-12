from __future__ import annotations

import logging
import re
from pathlib import Path

import jieba

LOGGER = logging.getLogger(__name__)


def load_stopwords(*paths: Path) -> set[str]:
    stopwords: set[str] = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        content = Path(path).read_text(encoding="utf-8")
        for line in content.splitlines():
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords


def add_custom_dict(path: Path | None) -> None:
    if path and path.exists():
        jieba.load_userdict(str(path))


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [w.strip() for w in jieba.lcut(text, cut_all=False) if w.strip()]


def filter_tokens(tokens: list[str], stopwords: set[str], min_len: int = 2) -> list[str]:
    filtered: list[str] = []
    for token in tokens:
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if token.isnumeric():
            continue
        if not re.search(r"[\u4e00-\u9fff]", token):
            continue
        filtered.append(token)
    return filtered


def preprocess_docs(
    docs: list[dict],
    stopwords: set[str],
    min_token_len: int = 2,
) -> list[dict]:
    processed: list[dict] = []
    for doc in docs:
        text = normalize_text(doc["text"])
        tokens = tokenize(text)
        tokens = filter_tokens(tokens, stopwords, min_len=min_token_len)
        processed.append({**doc, "tokens": tokens})
    LOGGER.info("Preprocessed %d documents", len(processed))
    return processed
