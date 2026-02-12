from __future__ import annotations

from pathlib import Path
import re

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


def assign_time_stage(year: int | None) -> str:
    if year is None:
        return "unknown"
    if 1996 <= year <= 2005:
        return "1996-2005"
    if 2006 <= year <= 2010:
        return "2006-2010"
    if 2011 <= year <= 2015:
        return "2011-2015"
    if 2016 <= year <= 2023:
        return "2016-2023"
    return "other"


def calc_stage_topic_strength(
    doc_topic_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    merged = doc_topic_df.merge(meta_df[["doc_id", "level", "stage"]], on="doc_id", how="left")
    subset = merged[merged["level"] == level].copy()
    if subset.empty:
        return pd.DataFrame()
    topic_cols = [c for c in subset.columns if c.startswith("topic_")]
    grouped = subset.groupby("stage", dropna=False)[topic_cols].mean().reset_index()
    stage_order = ["1996-2005", "2006-2010", "2011-2015", "2016-2023", "other", "unknown"]
    grouped["stage"] = pd.Categorical(grouped["stage"], categories=stage_order, ordered=True)
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
