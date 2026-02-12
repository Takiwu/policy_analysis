from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import PipelineConfig, ensure_paths
from .ingest import collect_documents, extract_year
from .lda import (
    choose_best_k,
    doc_topic_distribution,
    evaluate_topic_range,
    extract_topic_words,
    topic_strengths,
    train_lda,
    build_corpus,
)
from .preprocess import add_custom_dict, load_stopwords, preprocess_docs
from .tfidf import compute_tfidf
from .visualize import generate_wordcloud, plot_evaluations, plot_topic_strengths

LOGGER = logging.getLogger(__name__)


def run_pipeline(cfg: PipelineConfig) -> None:
    ensure_paths(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    add_custom_dict(cfg.custom_dict_path)
    stopwords = load_stopwords(cfg.stopwords_path, cfg.user_stopwords_path)

    LOGGER.info("Collecting documents from %s", cfg.input_dir)
    docs, stats = collect_documents(
        cfg.input_dir,
        enable_ocr=cfg.enable_ocr,
        tesseract_cmd=cfg.tesseract_cmd,
    )
    LOGGER.info(
        "Files scanned=%d, supported=%d, skipped=%d, empty_text=%d, by_ext=%s",
        stats.total_files,
        stats.supported_files,
        stats.skipped_files,
        stats.empty_text_files,
        stats.by_extension,
    )
    if stats.empty_text_files:
        LOGGER.warning(
            "发现 %d 个文本为空的文件。若包含扫描件或图片，请启用 OCR 或指定 tesseract 路径。",
            stats.empty_text_files,
        )
    for doc in docs:
        doc["year"] = extract_year(Path(doc["path"]).stem)

    doc_df = pd.DataFrame(docs)
    if not doc_df.empty:
        doc_df["is_empty"] = doc_df["text"].eq("")
    doc_df.to_csv(cfg.output_dir / "documents.csv", index=False, encoding="utf-8-sig")

    non_empty_docs = [d for d in docs if d["text"]]
    if not non_empty_docs:
        raise ValueError("没有可用文本用于建模。请检查 OCR 或输入文件格式。")

    processed = preprocess_docs(non_empty_docs, stopwords=stopwords, min_token_len=cfg.min_token_len)
    usable = [d for d in processed if d["tokens"]]
    dropped = len(processed) - len(usable)
    if dropped:
        LOGGER.warning("有 %d 个文档在分词/停用词处理后为空，已跳过建模。", dropped)
    if not usable:
        raise ValueError("分词后没有可用词条，无法进行 TF-IDF/LDA。请检查停用词或分词参数。")

    processed_df = pd.DataFrame(
        {
            "doc_id": list(range(len(processed))),
            "path": [d["path"] for d in processed],
            "year": [d.get("year") for d in processed],
            "tokens": [" ".join(d["tokens"]) for d in processed],
            "is_empty_after_preprocess": [not bool(d["tokens"]) for d in processed],
        }
    )
    processed_df.to_csv(cfg.output_dir / "tokens.csv", index=False, encoding="utf-8-sig")

    tokens_list = [d["tokens"] for d in usable]

    if cfg.use_tfidf:
        tfidf_df = compute_tfidf(tokens_list, top_n=cfg.tfidf_top_n)
        tfidf_df.to_csv(cfg.output_dir / "tfidf_top_keywords.csv", index=False, encoding="utf-8-sig")
        if cfg.wordcloud_font_path:
            generate_wordcloud(tfidf_df, cfg.output_dir / "tfidf_wordcloud.png", cfg.wordcloud_font_path)
        else:
            LOGGER.warning("未提供中文字体路径，词云可能无法正确显示中文。")

    start, end = cfg.topic_range
    if cfg.chosen_topics:
        evaluations = []
        models = {}
        dictionary, corpus = build_corpus(tokens_list)
        model = train_lda(
            corpus=corpus,
            dictionary=dictionary,
            num_topics=cfg.chosen_topics,
            alpha=cfg.resolved_alpha,
            eta=cfg.lda_eta,
            iterations=cfg.lda_iterations,
            passes=cfg.lda_passes,
            random_state=cfg.random_state,
        )
    else:
        evaluations, models, dictionary, corpus = evaluate_topic_range(
            tokens_list,
            start=start,
            end=end,
            alpha=cfg.lda_alpha,
            eta=cfg.lda_eta,
            iterations=cfg.lda_iterations,
            passes=cfg.lda_passes,
            random_state=cfg.random_state,
        )
        eval_df = pd.DataFrame([e.__dict__ for e in evaluations])
        eval_df.to_csv(cfg.output_dir / "topic_evaluation.csv", index=False, encoding="utf-8-sig")
        plot_evaluations(eval_df, cfg.output_dir)
        best_k = choose_best_k(evaluations)
        cfg.chosen_topics = best_k
        model = models[best_k]

    topic_words_df = extract_topic_words(model, topn=cfg.topn_words)
    topic_words_df.to_csv(cfg.output_dir / "lda_topic_words.csv", index=False, encoding="utf-8-sig")

    doc_topic_df = doc_topic_distribution(model, corpus)
    doc_topic_df.to_csv(cfg.output_dir / "doc_topic_distribution.csv", index=False, encoding="utf-8-sig")

    strength_df = topic_strengths(doc_topic_df)
    strength_df.to_csv(cfg.output_dir / "topic_strengths.csv", index=False, encoding="utf-8-sig")
    plot_topic_strengths(strength_df, cfg.output_dir)

    try:
        import pyLDAvis.gensim_models

        vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, str(cfg.output_dir / "pyldavis.html"))
    except ImportError:
        LOGGER.warning("pyLDAvis 未安装或不可用，已跳过可视化输出。")

    summary = {
        "documents": len(docs),
        "usable_documents": len(usable),
        "total_files": stats.total_files,
        "supported_files": stats.supported_files,
        "skipped_files": stats.skipped_files,
        "empty_text_files": stats.empty_text_files,
        "topics": cfg.chosen_topics,
        "output_dir": str(cfg.output_dir),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
