from __future__ import annotations

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

LOGGER = logging.getLogger(__name__)


def plot_evaluations(eval_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(eval_df["topic_count"], eval_df["perplexity"], marker="o", color="#1f77b4")
    ax1.set_xlabel("Topic Count")
    ax1.set_ylabel("Perplexity (log)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(eval_df["topic_count"], eval_df["coherence"], marker="s", color="#ff7f0e")
    ax2.set_ylabel("Coherence (c_v)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    fig.tight_layout()
    fig.savefig(output_dir / "topic_eval.png", dpi=200)
    plt.close(fig)


def plot_topic_strengths(strength_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(strength_df["topic"].astype(str), strength_df["strength"], color="#2ca02c")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Strength")
    ax.set_title("Topic Strengths")
    fig.tight_layout()
    fig.savefig(output_dir / "topic_strengths.png", dpi=200)
    plt.close(fig)


def generate_wordcloud(term_df: pd.DataFrame, output_path: Path, font_path: Path | None = None) -> None:
    freq = {row["term"]: row["tfidf_score"] for _, row in term_df.iterrows() if row["tfidf_score"] > 0}
    if not freq:
        LOGGER.warning("词云未生成：TF-IDF 为空或全为 0。")
        return
    if font_path and not Path(font_path).exists():
        LOGGER.warning("词云未生成：字体路径不存在 %s", font_path)
        return
    wc = WordCloud(
        font_path=str(font_path) if font_path else None,
        width=1000,
        height=600,
        background_color="white",
    )
    wc.generate_from_frequencies(freq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
