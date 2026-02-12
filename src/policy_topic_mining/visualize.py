from __future__ import annotations

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger(__name__)


def _font_can_render_chinese(font_path: Path, sample_text: str = "个人信息") -> bool:
    try:
        img = Image.new("RGB", (320, 120), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(str(font_path), 42)
        draw.text((8, 30), sample_text, font=font, fill="black")
        extrema = img.getbbox()
        if extrema is None:
            return False
        # 精确判定是否有非白像素
        px = img.convert("RGB").getdata()
        return any(p != (255, 255, 255) for p in px)
    except Exception:
        return False


def _pick_chinese_font(font_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if font_path:
        candidates.append(Path(font_path))

    candidates.extend(
        [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/STSONG.TTF"),
        ]
    )

    for cand in candidates:
        if not cand.exists():
            continue
        if _font_can_render_chinese(cand):
            return cand
    return None


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


def generate_wordcloud(term_df: pd.DataFrame, output_path: Path, font_path: Path | None = None) -> str | None:
    freq = {row["term"]: row["tfidf_score"] for _, row in term_df.iterrows() if row["tfidf_score"] > 0}
    if not freq:
        LOGGER.warning("词云未生成：TF-IDF 为空或全为 0。")
        return None

    chosen_font = _pick_chinese_font(font_path)
    if chosen_font is None:
        LOGGER.warning("词云未生成：未找到可渲染中文的字体，请安装或指定可用字体。")
        return None

    if font_path and Path(font_path) != chosen_font:
        LOGGER.warning("指定字体不可用，已自动回退到系统中文字体：%s", chosen_font)

    wc = WordCloud(
        font_path=str(chosen_font),
        width=1000,
        height=600,
        background_color="white",
        max_words=400,
        min_font_size=6,
        margin=1,
        prefer_horizontal=0.95,
        collocations=False,
        repeat=True,
        relative_scaling=0.35,
    )
    wc.generate_from_frequencies(freq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
    return str(chosen_font)
