"""Terrain-context features for mound candidates (phase 2 re-tune).

Prehistoric mounds in the upper Ohio valley are mostly 1-3 m high, 10-30 m across, on flat terraces
or bluff tops. The phase-1 dome score favoured tall natural knobs and spoil piles, so these features
describe the *setting* of each dome rather than the dome itself.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from . import viz

CTX_FEATS = [
    "l_height",
    "l_diam",
    "hd",
    "circ",
    "solidity",
    "ecc",
    "ann_slope",
    "ann_rough",
    "tpi50",
    "tpi150",
    "n_nbr60",
    "l_score",
]


def context_features(z: np.ndarray, det: pd.DataFrame, res: float = 1.0) -> pd.DataFrame:
    d = det.copy().reset_index(drop=True)
    if not len(d):
        for c in CTX_FEATS:
            d[c] = []
        return d
    zf = viz.fill_nan(z.astype(np.float32))
    zs = ndimage.gaussian_filter(zf, 2.0 / res)
    slope = viz.slope_deg(zs, res)
    lr5 = viz.lrm(zf, 5.0, res)
    tpi50 = zs - ndimage.uniform_filter(zs, int(101 / res))
    tpi150 = zs - ndimage.uniform_filter(zs, int(301 / res))
    H, W = z.shape
    rows = d.row.to_numpy().astype(int)
    cols = d.col.to_numpy().astype(int)
    ann_s, ann_r = np.zeros(len(d)), np.zeros(len(d))
    for i, (r, c, dm) in enumerate(zip(rows, cols, d.diam.to_numpy(), strict=True)):
        r_in = 0.75 * dm / res
        r_out = r_in + 15 / res
        h = int(np.ceil(r_out))
        r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, H), max(c - h, 0), min(c + h + 1, W)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        dd = np.hypot(yy - r, xx - c)
        ring = (dd >= r_in) & (dd <= r_out)
        if ring.sum() < 10:
            ann_s[i], ann_r[i] = np.nan, np.nan
            continue
        ann_s[i] = np.median(slope[r0:r1, c0:c1][ring])
        ann_r[i] = np.std(lr5[r0:r1, c0:c1][ring])
    d["ann_slope"] = ann_s
    d["ann_rough"] = ann_r
    rr, cc = np.clip(rows, 0, H - 1), np.clip(cols, 0, W - 1)
    d["tpi50"] = tpi50[rr, cc]
    d["tpi150"] = tpi150[rr, cc]
    xy = np.c_[rows, cols] * res
    d["n_nbr60"] = np.array([len(x) - 1 for x in cKDTree(xy).query_ball_point(xy, 60)])
    d["l_height"] = np.log(d.height)
    d["l_diam"] = np.log(d.diam)
    d["l_score"] = np.log(d.score)
    return d


def size_setting_prior(d: pd.DataFrame) -> np.ndarray:
    """Hand-built prior (no fitting): favour 1-3 m high, 10-35 m wide domes on gentle ground."""
    h = np.exp(-0.5 * ((np.log(d.height) - np.log(1.8)) / 0.45) ** 2)
    w = np.exp(-0.5 * ((np.log(d.diam) - np.log(20)) / 0.45) ** 2)
    flat = 1 / (1 + (d.ann_slope.fillna(30) / 8) ** 2)
    smooth = 1 / (1 + (d.ann_rough.fillna(1) / 0.15) ** 2)
    return h * w * flat * smooth
