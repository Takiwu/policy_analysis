from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import pdfplumber
from docx import Document
from PIL import Image
import pytesseract


LOGGER = logging.getLogger(__name__)

_SPACE_RE = re.compile(r"\s+")
_CN_WS_RE = re.compile(r"[\u3000\t\r\n]")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".md", ".txt", ""}


@dataclass
class CollectionStats:
    total_files: int
    supported_files: int
    skipped_files: int
    empty_text_files: int
    by_extension: dict[str, int]


def extract_text_from_pdf(path: Path) -> str:
    texts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return "\n".join(texts).strip()


def extract_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts).strip()


def extract_text_from_doc(path: Path) -> str:
    soffice = shutil.which("soffice")
    if not soffice:
        LOGGER.warning("未检测到 LibreOffice soffice，无法解析 .doc 文件：%s", path)
        return ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [soffice, "--headless", "--convert-to", "docx", "--outdir", tmpdir, str(path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                LOGGER.warning(".doc 转换失败：%s; %s", path, result.stderr.strip())
                return ""
            converted = next(Path(tmpdir).glob("*.docx"), None)
            if not converted:
                LOGGER.warning(".doc 转换未产生 docx 文件：%s", path)
                return ""
            return extract_text_from_docx(converted)
    except Exception as exc:
        LOGGER.warning(".doc 转换异常：%s; %s", path, exc)
        return ""


def extract_text_from_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
        if b"\x00" in raw[:2048]:
            LOGGER.warning("疑似二进制文件，跳过：%s", path)
            return ""
        for enc in ("utf-8", "gbk", "utf-16", "latin-1"):
            try:
                return raw.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        LOGGER.warning("文本读取失败：%s; %s", path, exc)
        return ""


def extract_text_from_image(path: Path, tesseract_cmd: str | None = None) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img, lang="chi_sim")
    except Exception as exc:
        LOGGER.warning("OCR failed for %s: %s", path, exc)
        return ""


def extract_text(path: Path, enable_ocr: bool = True, tesseract_cmd: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".doc":
        return extract_text_from_doc(path)
    if suffix in {".md", ".txt", ""}:
        return extract_text_from_text(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        if not enable_ocr:
            LOGGER.warning("OCR disabled, skipping image %s", path)
            return ""
        return extract_text_from_image(path, tesseract_cmd=tesseract_cmd)
    return ""


def clean_raw_text(text: str) -> str:
    text = _SPACE_RE.sub(" ", text)
    text = _CN_WS_RE.sub(" ", text)
    return text.strip()


def _collect_one(path: Path, enable_ocr: bool, tesseract_cmd: str | None) -> tuple[str, str, int, str, float]:
    started = time.perf_counter()
    text = extract_text(path, enable_ocr=enable_ocr, tesseract_cmd=tesseract_cmd)
    text = clean_raw_text(text)
    elapsed = time.perf_counter() - started
    return str(path), text, len(text), path.suffix.lower(), elapsed


def _resolve_worker_count(max_workers: int | None) -> int:
    if isinstance(max_workers, int) and max_workers > 0:
        return max_workers
    cpu = os.cpu_count() or 4
    # I/O + 解析混合任务，给一个偏保守且通用的默认值
    return min(8, max(2, cpu))


def collect_documents(
    input_dir: Path,
    exts: Iterable[str] | None = None,
    enable_ocr: bool = True,
    tesseract_cmd: str | None = None,
    max_workers: int | None = None,
) -> tuple[list[dict], CollectionStats]:
    exts = set(exts or SUPPORTED_EXTENSIONS)
    docs: list[dict] = []
    by_extension: dict[str, int] = {}

    all_files = [p for p in sorted(input_dir.rglob("*")) if p.is_file()]
    supported_files = [p for p in all_files if p.suffix.lower() in exts]

    total = len(supported_files)
    if total == 0:
        stats = CollectionStats(
            total_files=len(all_files),
            supported_files=0,
            skipped_files=len(all_files),
            empty_text_files=0,
            by_extension=by_extension,
        )
        return docs, stats

    workers = _resolve_worker_count(max_workers)
    slow_threshold_sec = 5.0
    progress_every = 50
    LOGGER.info("Start collecting %d files with workers=%d", total, workers)

    indexed_docs: dict[int, dict] = {}
    done = 0

    if workers == 1 or total < 20:
        for idx, path in enumerate(supported_files):
            p_str, text, text_len, suffix, elapsed = _collect_one(path, enable_ocr, tesseract_cmd)
            by_extension[suffix] = by_extension.get(suffix, 0) + 1
            indexed_docs[idx] = {"path": p_str, "text": text, "text_len": text_len}
            done += 1
            if elapsed >= slow_threshold_sec:
                LOGGER.warning("Slow file %.2fs: %s", elapsed, p_str)
            if done % progress_every == 0 or done == total:
                LOGGER.info("Collecting progress: %d/%d (%.1f%%)", done, total, done * 100.0 / total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_collect_one, path, enable_ocr, tesseract_cmd): idx
                for idx, path in enumerate(supported_files)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                p_str, text, text_len, suffix, elapsed = future.result()
                by_extension[suffix] = by_extension.get(suffix, 0) + 1
                indexed_docs[idx] = {"path": p_str, "text": text, "text_len": text_len}
                done += 1
                if elapsed >= slow_threshold_sec:
                    LOGGER.warning("Slow file %.2fs: %s", elapsed, p_str)
                if done % progress_every == 0 or done == total:
                    LOGGER.info("Collecting progress: %d/%d (%.1f%%)", done, total, done * 100.0 / total)

    docs = [indexed_docs[i] for i in range(total)]

    empty_text_files = sum(1 for d in docs if not d["text"])
    stats = CollectionStats(
        total_files=len(all_files),
        supported_files=len(supported_files),
        skipped_files=len(all_files) - len(supported_files),
        empty_text_files=empty_text_files,
        by_extension=by_extension,
    )
    return docs, stats


def extract_year(text: str) -> int | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(0)) if match else None
