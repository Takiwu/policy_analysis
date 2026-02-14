from __future__ import annotations

import argparse
from pathlib import Path

from src.policy_topic_mining import PipelineConfig, run_pipeline
from src.policy_topic_mining.config import parse_topic_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy topic mining with LDA")
    parser.add_argument("--input", required=True, help="Input directory containing policy files")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--stopwords", default="stopwords/stopwords.txt", help="Stopwords file")
    parser.add_argument("--user-stopwords", default="stopwords/user_stopwords.txt", help="User stopwords file")
    parser.add_argument("--custom-dict", default="stopwords/custom_dict.txt", help="Custom jieba dict")
    parser.add_argument("--topic-range", default="5,20", help="Topic range, e.g. 5,20")
    parser.add_argument("--topics", type=int, default=13, help="Fixed number of topics (default: 13)")
    parser.add_argument(
        "--skip-topic-eval",
        action="store_true",
        help="Skip perplexity/coherence topic-range evaluation",
    )
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR for images")
    parser.add_argument("--tesseract-cmd", default=None, help="Path to tesseract executable")
    parser.add_argument("--wordcloud-font", default=None, help="Font path for Chinese wordcloud")
    parser.add_argument(
        "--year-stages",
        default=None,
        help="Fixed year stages, e.g. 2016-2021,2022-2023,2024-. If omitted, stages are split by each year found in data.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        stopwords_path=Path(args.stopwords),
        user_stopwords_path=Path(args.user_stopwords) if args.user_stopwords else None,
        custom_dict_path=Path(args.custom_dict) if args.custom_dict else None,
        topic_range=parse_topic_range(args.topic_range),
        chosen_topics=args.topics,
        evaluate_topic_range_first=not args.skip_topic_eval,
        enable_ocr=not args.disable_ocr,
        tesseract_cmd=args.tesseract_cmd,
        wordcloud_font_path=Path(args.wordcloud_font) if args.wordcloud_font else None,
        year_stages=args.year_stages,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
