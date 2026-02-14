from __future__ import annotations

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


CENTRAL_HINTS = [
    "国务院",
    "中央",
    "科技部",
    "财政部",
    "教育部",
    "国家",
    "部委",
]

LOCAL_HINTS = [
    "省",
    "市",
    "自治区",
    "厅",
    "局",
    "县",
    "区",
    "地方",
]


def infer_level(path_text: str) -> str:
    text = path_text or ""
    if any(k in text for k in CENTRAL_HINTS):
        return "central"
    if any(k in text for k in LOCAL_HINTS):
        return "local"
    return "unknown"


def infer_year_from_text(path_stem: str, text: str) -> int | None:
    combined = f"{path_stem} {text[:2000]}"
    match = re.search(r"(19|20)\d{2}", combined)
    if not match:
        return None
    return int(match.group(0))


def extract_date_from_stem(path_stem: str) -> str | None:
    """Extract YYYY-MM-DD from a stem like 'AAA_2021-08-30'.

    Returns the first matched date (most typical is at the end).
    """

    if not path_stem:
        return None
    m = re.search(r"_(\d{4}-\d{2}-\d{2})(?:$|[^\d])", path_stem)
    if not m:
        return None
    return m.group(1)


def base_title_from_dated_stem(path_stem: str) -> str | None:
    date_str = extract_date_from_stem(path_stem)
    if not date_str:
        return None
    # Prefer stripping the exact suffix pattern
    suffix = f"_{date_str}"
    if path_stem.endswith(suffix):
        return path_stem[: -len(suffix)]
    return path_stem.split(suffix, 1)[0]


def infer_date_from_pairing(path_stem: str, base_to_date: dict[str, str]) -> str | None:
    """Infer date for stems like 'AAA-BBB' by matching left part 'AAA' to dated mapping."""

    if not path_stem or not base_to_date:
        return None
    # already dated
    d = extract_date_from_stem(path_stem)
    if d:
        return d

    # try common dash variants
    for sep in ("-", "—", "–", "－"):
        if sep in path_stem:
            left = path_stem.split(sep, 1)[0].strip()
            if left in base_to_date:
                return base_to_date[left]
    return None


def _label_bin(start_year: int, end_year: int) -> str:
    return str(start_year) if start_year == end_year else f"{start_year}-{end_year}"


def parse_year_stages(spec: str) -> list[tuple[int, int | None]]:
    """Parse stages like: '2017-2021,2022-2023,2024-'.

    - Open-ended end year is allowed (e.g. '2024-') meaning 2024 and later.
    """

    items = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not items:
        raise ValueError("year_stages is empty")
    stages: list[tuple[int, int | None]] = []
    for item in items:
        if "-" not in item:
            y = int(item)
            stages.append((y, y))
            continue
        left, right = item.split("-", 1)
        start = int(left.strip())
        right = right.strip()
        end = int(right) if right else None
        stages.append((start, end))

    # sanity: non-decreasing, non-overlapping
    stages_sorted = sorted(stages, key=lambda x: x[0])
    prev_end: int | None = None
    for s, e in stages_sorted:
        if prev_end is not None and s <= prev_end:
            raise ValueError("year_stages contains overlapping or unsorted ranges")
        prev_end = e
    return stages_sorted


def build_fixed_stage_assigner(
    stages: list[tuple[int, int | None]],
) -> tuple[Callable[[int | None], str], list[str]]:
    labels: list[str] = []
    for s, e in stages:
        if e is None:
            labels.append(f"{s}-now")
        else:
            labels.append(_label_bin(s, e))

    def assign(y: int | None) -> str:
        if y is None:
            return "unknown"
        for (s, e), lab in zip(stages, labels, strict=False):
            if e is None:
                if y >= s:
                    return lab
            elif s <= y <= e:
                return lab
        return "other"

    # IMPORTANT: stage_order only contains configured stages;
    # 'other'/'unknown' must be ignored in all outputs.
    stage_order = labels
    return assign, stage_order


def build_time_stage_assigner(
    years: list[int],
    max_bins: int = 4,
) -> tuple[Callable[[int | None], str], list[str]]:
    """Build a stage assigner that matches the dataset's year distribution.

    Strategy:
    - If unique years <= max_bins: each year is a stage.
    - Else: split into max_bins bins with roughly equal document counts (by year order).
    """

    years_valid = [y for y in years if isinstance(y, int)]
    if not years_valid:
        def assign(y: int | None) -> str:
            return "unknown"

        # No valid stages to output
        return assign, []

    years_sorted = sorted(years_valid)
    uniq = sorted(set(years_sorted))
    if len(uniq) <= max_bins:
        bins = [(y, y) for y in uniq]
    else:
        n = max_bins
        # boundaries by quantile positions over the sorted list (keeps counts balanced)
        cut_points = [0]
        for i in range(1, n):
            cut_points.append(round(len(years_sorted) * i / n))
        cut_points.append(len(years_sorted))

        bins: list[tuple[int, int]] = []
        for i in range(n):
            start_i = cut_points[i]
            end_i = max(cut_points[i + 1] - 1, start_i)
            start_y = years_sorted[start_i]
            end_y = years_sorted[end_i]
            bins.append((start_y, end_y))

        # merge any identical / overlapping bins caused by repeated years
        merged: list[tuple[int, int]] = []
        for s, e in bins:
            if not merged:
                merged.append((s, e))
                continue
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        bins = merged

    labels = [_label_bin(s, e) for s, e in bins]

    def assign(y: int | None) -> str:
        if y is None:
            return "unknown"
        for (s, e), lab in zip(bins, labels, strict=False):
            if s <= y <= e:
                return lab
        return "other"

    # IMPORTANT: stage_order only contains computed stages;
    # 'other'/'unknown' must be ignored in all outputs.
    stage_order = labels
    return assign, stage_order


def calc_stage_topic_strength(
    doc_topic_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    level: str | None,
    stage_order: list[str] | None = None,
) -> pd.DataFrame:
    merged = doc_topic_df.merge(meta_df[["doc_id", "level", "stage"]], on="doc_id", how="left")
    if level is None or level == "all":
        subset = merged.copy()
    else:
        subset = merged[merged["level"] == level].copy()
    if subset.empty:
        return pd.DataFrame()
    topic_cols = [c for c in subset.columns if c.startswith("topic_")]
    counts = subset.groupby("stage", dropna=False).size().rename("doc_count")
    means = subset.groupby("stage", dropna=False)[topic_cols].mean()
    grouped = pd.concat([counts, means], axis=1).reset_index()

    if stage_order:
        grouped["stage"] = pd.Categorical(grouped["stage"], categories=stage_order, ordered=True)
        grouped = grouped.set_index("stage").reindex(stage_order).reset_index()
        if "doc_count" in grouped.columns:
            grouped["doc_count"] = grouped["doc_count"].fillna(0).astype(int)
        for c in topic_cols:
            grouped[c] = grouped[c].fillna(0.0)

    grouped = grouped.sort_values("stage")
    return grouped


def plot_stage_strength(stage_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if stage_df.empty:
        return
    topic_cols = [c for c in stage_df.columns if c.startswith("topic_")]
    plt.figure(figsize=(10, 5))
    x = stage_df["stage"].astype(str)
    for c in topic_cols:
        plt.plot(x, stage_df[c], marker="o", linewidth=1.3, label=c)
    plt.title(title)
    plt.xlabel("Time Stage")
    plt.ylabel("Topic Strength")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
