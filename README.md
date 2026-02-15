# 政策文本主题挖掘（LDA）

- **数据清洗**：仅保留政策标题与正文文本。
- **中文分词**：使用 `jieba` 精确模式，支持自定义词典。
- **停用词剔除**：内置常用停用词 + 自定义停用词。
- **TF-IDF 关键词**：输出高频关键词与词云图。
- **LDA 主题建模**：计算**困惑度**与**一致性**，自动选择主题数；可输出主题词、文档-主题分布、主题强度。
- **pyLDAvis 可视化**：交互式主题可视化 HTML。

## 目录结构

- `run.py`：命令行入口
- `src/policy_topic_mining/`：核心代码
- `stopwords/`：停用词与自定义词典
- `outputs/`：输出结果

## 运行方式

> 依赖说明：`gensim` 与 `pyLDAvis` 目前对 **Python 3.14** 兼容性不足，建议使用 **Python 3.11/3.12** 运行完整流程。

1. 放置数据：将政策文件（`pdf/docx/doc/md/txt/png/jpg`）放入任意多层目录。
2. 运行分析（示例）：

```bash
D:/Codes/policy_analysis/.venv/Scripts/python.exe run.py --input D:/Codes/get_policy/output --output outputs --wordcloud-font D:/Codes/policy_analysis/fonts/HarmonyOS_Sans_Regular.ttf --disable-ocr --topics 13 --workers 8
```

## 主要输出（`outputs/`）

- `documents.csv`：原始文本与来源路径
- `tokens.csv`：分词结果
- `tfidf_top_keywords.csv`：TF-IDF 高频关键词
- `tfidf_top_keywords_central.csv`：中央层面 TF-IDF 高频关键词（自动识别）
- `tfidf_top_keywords_local.csv`：地方层面 TF-IDF 高频关键词（自动识别）
- `topic_evaluation.csv`：困惑度与一致性
- `topic_eval.png`：主题数评估曲线
- `lda_topic_words.csv`：主题-关键词分布
- `doc_topic_distribution.csv`：文档-主题分布
- `topic_strengths.csv`：主题强度
- `topic_strengths.png`：主题强度图
- `topic_strength_by_stage_central.csv/.png`：中央主题强度时段演进
- `topic_strength_by_stage_local.csv/.png`：地方主题强度时段演进
- `topic_strength_by_stage_all.csv/.png`：总体（不分层级）主题强度时段演进
- `duplicates_report.csv`：重复文档检测报告（按全文哈希）
- `pyldavis.html`：交互式主题可视化
- `summary.json`：运行摘要

## OCR 说明（PNG/JPG）

默认对图片执行 OCR（`pytesseract`）。如果系统未安装 Tesseract，可通过以下方式之一：

- 安装 Tesseract 并指定路径：`--tesseract-cmd`
- 或禁用 OCR：`--disable-ocr`

> `.doc` 文件需安装 LibreOffice（`soffice`）以便转换为 `.docx` 后解析。

> 注意：禁用 OCR 时，图片文件会被扫描到但文本为空，建模阶段会自动跳过这些空文本。

## 参数提示

- `--topic-range 5,20`：主题数搜索范围（支持同值，如 `13,13`）
- `--topics 13`：固定主题数（默认值为 13，对应论文设置）
- `--skip-topic-eval`：跳过困惑度/一致性评估（默认不跳过，建议保持默认）
- `--year-stages`：按年份固定分期（例：`2016-2021,2022-2023,2024-`）；不传时自动按识别到的所有年份逐年分段。若使用开放区间（如 `2024-`），需放在最后。
- `--workers 8`：文档采集并行线程数（默认自动）；在大量 PDF 场景可显著缩短“Collecting documents”阶段时长
- `--wordcloud-font`：中文词云字体路径（例如 `.\fonts\HarmonyOS_Sans_Regular.ttf`）

> 性能提示：采集阶段已支持并行读取，并会周期性输出进度（如 `Collecting progress: 200/889`）与慢文件告警。

> 词云为空通常是以下原因：
>
> - 文本被停用词或规则过滤后为空
> - 未提供/提供的字体不支持中文
> - 输入文件是扫描件但未启用 OCR

## 复现论文参数建议

- 主题数：`K=13`
- 超参数：$\alpha=50/K$, $\beta=0.01$
- 迭代次数：`100`
- 每主题关键词：`15`

这些参数在代码中已设置为默认逻辑，可直接运行或通过命令行覆盖。

> TF-IDF实现说明：采用 `CountVectorizer + TfidfTransformer`，并将词权重输出为归一化占比（`tfidf_score`），数值量级通常为 `0.0x`。

> 分层说明：中央/地方由文件路径与标题关键字自动推断，可在 `summary.json` 查看分层统计。

> 日期识别说明：
>
> - 文件名 `AAA_YYYY-MM-DD` 会直接识别日期；
> - 文件名 `AAA-BBB` 会尝试继承同名 `AAA_YYYY-MM-DD` 的日期；
> - `法宝数据库` 目录文档优先从正文中 `公布日期` 后提取日期（支持 `YYYY.MM.DD / YYYY.MM / YYYY`）。
