from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    stopwords_path: Path
    user_stopwords_path: Path | None = None
    custom_dict_path: Path | None = None
    min_token_len: int = 2
    use_tfidf: bool = True
    tfidf_top_n: int = 40
    topic_range: tuple[int, int] = (5, 20)
    chosen_topics: int | None = None
    evaluate_topic_range_first: bool = True
    lda_iterations: int = 100
    lda_passes: int = 10
    lda_alpha: str | float = "50/K"
    lda_eta: float = 0.01
    topn_words: int = 15
    random_state: int = 42
    enable_ocr: bool = True
    ingest_workers: int | None = None
    tesseract_cmd: str | None = None
    wordcloud_font_path: Path | None = None
    # 为空时自动按识别到的年份逐年分段
    year_stages: str | None = None

    @property
    def resolved_alpha(self) -> float:
        if isinstance(self.lda_alpha, (int, float)):
            return float(self.lda_alpha)
        if self.lda_alpha == "50/K" and self.chosen_topics:
            return 50.0 / float(self.chosen_topics)
        if self.lda_alpha == "auto":
            return 1.0
        if self.chosen_topics is None:
            return 1.0
        return 50.0 / float(self.chosen_topics)


def ensure_paths(cfg: PipelineConfig) -> None:
    cfg.input_dir = Path(cfg.input_dir)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.stopwords_path = Path(cfg.stopwords_path)
    if cfg.user_stopwords_path:
        cfg.user_stopwords_path = Path(cfg.user_stopwords_path)
    if cfg.custom_dict_path:
        cfg.custom_dict_path = Path(cfg.custom_dict_path)
    if cfg.wordcloud_font_path:
        cfg.wordcloud_font_path = Path(cfg.wordcloud_font_path)


def validate_config(cfg: PipelineConfig) -> None:
    if not cfg.input_dir.exists() or not cfg.input_dir.is_dir():
        raise ValueError(f"输入目录不存在或不是目录: {cfg.input_dir}")

    if cfg.topic_range[0] < 2 or cfg.topic_range[1] < cfg.topic_range[0]:
        raise ValueError("topic_range 配置非法，应满足 start >= 2 且 end >= start")

    if cfg.chosen_topics is not None and cfg.chosen_topics < 2:
        raise ValueError("topics 必须大于等于 2")

    if cfg.min_token_len < 1:
        raise ValueError("min_token_len 必须大于等于 1")

    if cfg.ingest_workers is not None and cfg.ingest_workers < 1:
        raise ValueError("workers 必须为正整数")


def parse_topic_range(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("topic_range must be two integers separated by a comma, e.g. 5,20")
    start, end = int(parts[0]), int(parts[1])
    if start < 2 or end < start:
        raise ValueError("topic_range must be like 5,20 and end >= start >= 2")
    return start, end


def iter_all_files(root: Path, exts: Iterable[str]) -> list[Path]:
    exts_lower = {e.lower() for e in exts}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower]
