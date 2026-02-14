from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import Counter
import hashlib

import pandas as pd

from .analysis import (
    build_yearly_stage_assigner,
    build_fixed_stage_assigner,
    base_title_from_dated_stem,
    calc_stage_topic_strength,
    extract_date_from_stem,
    infer_date_from_fabao_content,
    infer_level,
    infer_date_from_pairing,
    infer_year_from_text,
    parse_year_stages,
    plot_stage_strength,
)
from .config import PipelineConfig, ensure_paths, validate_config
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


def _save_and_plot_stage_strength(
    stage_df: pd.DataFrame,
    output_dir: Path,
    csv_name: str,
    png_name: str,
    title: str,
) -> None:
    if stage_df.empty:
        return
    stage_df.to_csv(output_dir / csv_name, index=False, encoding="utf-8-sig")
    plot_stage_strength(stage_df, output_dir / png_name, title=title)


def _check_duplicates(docs: list[dict], output_dir: Path) -> tuple[list[dict], int, int]:
    """检测重复文档（按清洗后全文哈希），并输出报告。

    返回：标注后的 docs、重复文档数量、重复组数量。
    """

    hash_to_first: dict[str, int] = {}
    duplicate_count = 0
    group_counter: Counter[str] = Counter()

    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        if not text:
            doc["is_duplicate"] = False
            doc["duplicate_of"] = None
            doc["text_hash"] = None
            continue
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        doc["text_hash"] = digest
        group_counter[digest] += 1
        if digest in hash_to_first:
            first_idx = hash_to_first[digest]
            doc["is_duplicate"] = True
            doc["duplicate_of"] = docs[first_idx]["path"]
            duplicate_count += 1
        else:
            hash_to_first[digest] = i
            doc["is_duplicate"] = False
            doc["duplicate_of"] = None

    dup_rows = []
    for d in docs:
        if d.get("text_hash") and group_counter[d["text_hash"]] > 1:
            dup_rows.append(
                {
                    "path": d.get("path"),
                    "is_duplicate": d.get("is_duplicate"),
                    "duplicate_of": d.get("duplicate_of"),
                    "text_hash": d.get("text_hash"),
                    "text_len": d.get("text_len"),
                    "group_size": group_counter[d["text_hash"]],
                }
            )

    pd.DataFrame(dup_rows).to_csv(output_dir / "duplicates_report.csv", index=False, encoding="utf-8-sig")
    duplicate_groups = sum(1 for _, c in group_counter.items() if c > 1)
    return docs, duplicate_count, duplicate_groups


def run_pipeline(cfg: PipelineConfig) -> None:
    ensure_paths(cfg)
    validate_config(cfg)
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
    # 1) build mapping (仅网信办): base title -> date from stems like AAA_YYYY-MM-DD
    base_to_date: dict[str, str] = {}
    for doc in docs:
        if "网信办" not in (doc.get("path", "") or ""):
            continue
        stem = Path(doc["path"]).stem
        base = base_title_from_dated_stem(stem)
        date_str = extract_date_from_stem(stem)
        if base and date_str:
            base_to_date.setdefault(base, date_str)

    # 2) assign date/year/level
    years_all: list[int] = []
    for doc in docs:
        path_stem = Path(doc["path"]).stem

        # 法宝数据库优先从内容中 papers 后日期提取
        date_str = infer_date_from_fabao_content(doc.get("path", ""), doc.get("text", ""))
        if not date_str:
            date_str = infer_date_from_pairing(doc.get("path", ""), path_stem, base_to_date)
        if date_str:
            doc["date"] = date_str
            doc["year"] = int(date_str[:4])
        else:
            doc["date"] = None
            year = extract_year(path_stem)
            if year is None:
                year = infer_year_from_text(path_stem, doc.get("text", ""))
            doc["year"] = year

        if isinstance(doc.get("year"), int):
            years_all.append(int(doc["year"]))

        doc["level"] = infer_level(doc.get("path", ""))

    docs, duplicate_count, duplicate_groups = _check_duplicates(docs, cfg.output_dir)
    if duplicate_count:
        LOGGER.warning("检测到重复文档 %d 条（重复组 %d），详见 duplicates_report.csv", duplicate_count, duplicate_groups)

    if cfg.year_stages:
        fixed = parse_year_stages(cfg.year_stages)
        stage_assign, stage_order = build_fixed_stage_assigner(fixed)
        LOGGER.info("Using fixed year stages: %s", cfg.year_stages)
    else:
        stage_assign, stage_order = build_yearly_stage_assigner(years_all)
        LOGGER.info("Using yearly stages inferred from data: %s", ",".join(stage_order))
    for doc in docs:
        doc["stage"] = stage_assign(doc.get("year"))

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
            "date": [d.get("date") for d in processed],
            "year": [d.get("year") for d in processed],
            "level": [d.get("level") for d in processed],
            "stage": [d.get("stage") for d in processed],
            "tokens": [" ".join(d["tokens"]) for d in processed],
            "is_empty_after_preprocess": [not bool(d["tokens"]) for d in processed],
        }
    )
    processed_df.to_csv(cfg.output_dir / "tokens.csv", index=False, encoding="utf-8-sig")

    # 停用词效果诊断：输出过滤后高频词，便于核验停用词是否生效
    token_counter: Counter[str] = Counter()
    for d in processed:
        token_counter.update(d["tokens"])
    top_filtered_terms = token_counter.most_common(100)
    pd.DataFrame(top_filtered_terms, columns=["term", "count"]).to_csv(
        cfg.output_dir / "post_stopwords_top_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tokens_list = [d["tokens"] for d in usable]
    wordcloud_fonts_used: dict[str, str] = {}

    if cfg.use_tfidf:
        tfidf_df = compute_tfidf(tokens_list, top_n=cfg.tfidf_top_n)
        tfidf_df.to_csv(cfg.output_dir / "tfidf_top_keywords.csv", index=False, encoding="utf-8-sig")
        if cfg.wordcloud_font_path and not tfidf_df.empty:
            used_font = generate_wordcloud(tfidf_df, cfg.output_dir / "tfidf_wordcloud.png", cfg.wordcloud_font_path)
            if used_font:
                wordcloud_fonts_used["global"] = used_font
        elif not cfg.wordcloud_font_path:
            LOGGER.warning("未提供中文字体路径，词云可能无法正确显示中文。")
        else:
            LOGGER.warning("TF-IDF 结果为空，已跳过词云生成。")

        usable_df = pd.DataFrame(usable)
        for level_name in ["central", "local"]:
            sub = usable_df[usable_df.get("level") == level_name]
            if sub.empty:
                continue
            level_tokens = [t for t in sub["tokens"].tolist() if t]
            if not level_tokens:
                continue
            tfidf_level_df = compute_tfidf(level_tokens, top_n=cfg.tfidf_top_n)
            tfidf_level_df.to_csv(
                cfg.output_dir / f"tfidf_top_keywords_{level_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if cfg.wordcloud_font_path:
                used_font = generate_wordcloud(
                    tfidf_level_df,
                    cfg.output_dir / f"tfidf_wordcloud_{level_name}.png",
                    cfg.wordcloud_font_path,
                )
                if used_font:
                    wordcloud_fonts_used[level_name] = used_font

    start, end = cfg.topic_range
    evaluations = []
    models = {}
    dictionary, corpus = build_corpus(tokens_list)

    if cfg.evaluate_topic_range_first:
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

    if cfg.chosen_topics:
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
        if not evaluations:
            raise ValueError("未提供固定主题数且跳过了主题评估，无法确定 K。")
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

    meta_df = pd.DataFrame(
        {
            "doc_id": list(range(len(usable))),
            "level": [d.get("level") for d in usable],
            "stage": [d.get("stage") for d in usable],
        }
    )
    stage_all = calc_stage_topic_strength(doc_topic_df, meta_df, level="all", stage_order=stage_order)
    _save_and_plot_stage_strength(
        stage_all,
        cfg.output_dir,
        "topic_strength_by_stage_all.csv",
        "topic_strength_by_stage_all.png",
        "All Policies Topic Strength Trend",
    )

    stage_central = calc_stage_topic_strength(doc_topic_df, meta_df, level="central", stage_order=stage_order)
    _save_and_plot_stage_strength(
        stage_central,
        cfg.output_dir,
        "topic_strength_by_stage_central.csv",
        "topic_strength_by_stage_central.png",
        "Central Policy Topic Strength Trend",
    )

    stage_local = calc_stage_topic_strength(doc_topic_df, meta_df, level="local", stage_order=stage_order)
    _save_and_plot_stage_strength(
        stage_local,
        cfg.output_dir,
        "topic_strength_by_stage_local.csv",
        "topic_strength_by_stage_local.png",
        "Local Policy Topic Strength Trend",
    )

    try:
        import pyLDAvis.gensim_models

        vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, str(cfg.output_dir / "pyldavis.html"))
    except ImportError:
        LOGGER.warning("pyLDAvis 未安装或不可用，已跳过可视化输出。")

    # stage_counts should ignore 'other'/'unknown' in all cases
    stage_counts = pd.Series([d.get("stage") for d in docs]).value_counts(dropna=False).to_dict()
    if stage_order:
        stage_counts = {k: int(stage_counts.get(k, 0)) for k in stage_order}
    else:
        stage_counts = {}

    summary = {
        "documents": len(docs),
        "usable_documents": len(usable),
        "level_counts": pd.Series([d.get("level") for d in docs]).value_counts(dropna=False).to_dict(),
        "stage_counts": stage_counts,
        "total_files": stats.total_files,
        "supported_files": stats.supported_files,
        "skipped_files": stats.skipped_files,
        "empty_text_files": stats.empty_text_files,
        "duplicate_documents": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "topics": cfg.chosen_topics,
        "wordcloud_fonts_used": wordcloud_fonts_used,
        "output_dir": str(cfg.output_dir),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
