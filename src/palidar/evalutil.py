"""Shared evaluation helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def split_of(win: str) -> str:
    """Deterministic dev/test split: odd numeric id -> dev, even -> test."""
    n = int(re.findall(r"(\d+)$", win)[0])
    return "dev" if n % 2 else "test"


def windows():
    w = pd.read_csv(ROOT / "data" / "calibration" / "windows.csv")
    w["split"] = w.win.map(split_of)
    return w


def recall_fp_curve(pos_scores: np.ndarray, neg_scores: np.ndarray, neg_area_km2: float, thresholds):
    """pos_scores: best matched score per known site (nan = never detected)."""
    out = []
    for t in thresholds:
        rec = np.mean(np.nan_to_num(pos_scores, nan=-np.inf) >= t)
        fp = np.sum(neg_scores >= t) / neg_area_km2
        out.append(dict(threshold=t, recall=rec, fp_per_km2=fp))
    return pd.DataFrame(out)
