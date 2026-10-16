"""Detector B: relict charcoal hearths (RCH).

Two stages
1. candidates: local maxima of a "flat disc inside a sloping ring" contrast map (high recall,
   low precision) -> detectors.detect_hearths(..., min_contrast=low)
2. ranking: a small gradient-boosted classifier on rotation-normalised morphometric features
   (ring-wise slope profile, upslope-vs-downslope local relief = the cut/fill signature,
   planarity of the platform, regional slope). Features are rotated so that "down" is the
   regional downslope direction, so the model is aspect-invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import detectors as D
from . import viz

R = 16  # patch half-size (m)
_yy, _xx = np.mgrid[-R : R + 1, -R : R + 1]
_RAD = np.hypot(_yy, _xx)
_ANG = np.arctan2(_yy, _xx)  # rows down = +y
RINGS = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 12), (12, 16)]


def candidates(z, res=1.0):
    return D.detect_hearths(z, res, max_inner_slope=12.0, min_contrast=0.5, min_dist=5)


def features(z, cand: pd.DataFrame, res=1.0) -> pd.DataFrame:
    z = viz.fill_nan(z)
    zs = ndi.gaussian_filter(z, 1.0)
    s = viz.slope_deg(zs, res)
    l = viz.lrm(z, 10, res)
    gy, gx = np.gradient(ndi.gaussian_filter(z, 8.0), res)  # rows, cols
    reg_slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    H, W = z.shape
    rr = np.clip(cand.row.to_numpy().astype(int), R, H - R - 1)
    cc = np.clip(cand.col.to_numpy().astype(int), R, W - R - 1)
    idx_r = rr[:, None, None] + _yy[None]
    idx_c = cc[:, None, None] + _xx[None]
    ps = s[idx_r, idx_c]
    pl = l[idx_r, idx_c]
    pz = zs[idx_r, idx_c]
    down = np.arctan2(-gy[rr, cc], -gx[rr, cc])  # direction of descent in (row, col) frame
    rel = (_ANG[None] - down[:, None, None] + np.pi) % (2 * np.pi) - np.pi  # 0 = downslope
    downhalf = np.abs(rel) < np.pi / 2
    f = {}
    for a, b in RINGS:
        m = (_RAD >= a) & (_RAD < b)
        f[f"slope_r{a}_{b}"] = ps[:, m].mean(1)
        f[f"lrm_r{a}_{b}"] = pl[:, m].mean(1)
    for a, b in [(4, 8), (8, 12)]:
        m = ((_RAD >= a) & (_RAD < b))[None]
        f[f"lrm_down_{a}_{b}"] = np.where(m & downhalf, pl, np.nan).reshape(len(rr), -1)
        f[f"lrm_down_{a}_{b}"] = np.nanmean(f[f"lrm_down_{a}_{b}"], 1)
        up = np.where(m & ~downhalf, pl, np.nan).reshape(len(rr), -1)
        f[f"lrm_up_{a}_{b}"] = np.nanmean(up, 1)
        f[f"cutfill_{a}_{b}"] = f[f"lrm_down_{a}_{b}"] - f[f"lrm_up_{a}_{b}"]
    # planarity of the platform (residual std after plane fit, r <= 5 m)
    m5 = _RAD <= 5
    A = np.c_[_yy[m5], _xx[m5], np.ones(m5.sum())]
    P = pz[:, m5]
    coef, *_ = np.linalg.lstsq(A, P.T, rcond=None)
    resid = P - (A @ coef).T
    f["plane_resid"] = resid.std(1)
    f["plane_slope"] = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
    f["reg_slope"] = reg_slope[rr, cc]
    f["slope_contrast"] = f["slope_r9_12"] - f["slope_r0_3"]
    out = pd.DataFrame(f)
    for k in ("contrast", "inner_slope", "outer_slope", "flat_diam", "flat_ecc", "radius"):
        out[k] = cand[k].to_numpy()
    out["flat_touches"] = cand.flat_touches.astype(int).to_numpy()
    return out


def blocks(shape, block=2000, pad=60):
    H, W = shape
    for i in range(0, H, block):
        for j in range(0, W, block):
            yield (
                max(0, i - pad),
                min(H, i + block + pad),
                max(0, j - pad),
                min(W, j + block + pad),
                i,
                min(H, i + block),
                j,
                min(W, j + block),
            )


def candidates_with_features(z, res=1.0, block=2000):
    """Blocked candidate generation + features for large rasters."""
    out = []
    for r0, r1, c0, c1, i0, i1, j0, j1 in blocks(z.shape, block):
        sub = z[r0:r1, c0:c1]
        c = candidates(sub, res)
        if c.empty:
            continue
        f = features(sub, c, res)
        c = c.reset_index(drop=True)
        c["row"] += r0
        c["col"] += c0
        keep = (c.row >= i0) & (c.row < i1) & (c.col >= j0) & (c.col < j1)
        out.append(pd.concat([c[["row", "col"]], f], axis=1)[keep.to_numpy()])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- CNN inputs
PATCH = 48


def patch_channels(z, res=1.0):
    """3-channel image used by the hearth CNN: LRM(10 m), slope, fine LRM(3 m), each scaled to ~[-1, 1]."""
    z = viz.fill_nan(z)
    zs = ndi.gaussian_filter(z, 0.7)
    c0 = np.clip(viz.lrm(zs, 10, res) / 1.5, -1, 1)
    c1 = np.clip(viz.slope_deg(zs, res) / 40.0, 0, 1.5) - 0.5
    c2 = np.clip(viz.lrm(zs, 3, res) / 0.5, -1, 1)
    return np.stack([c0, c1, c2]).astype("float16")


def extract_patches(ch, rows, cols, size=PATCH):
    h = size // 2
    _, H, W = ch.shape
    rows = np.clip(np.asarray(rows, int), h, H - h)
    cols = np.clip(np.asarray(cols, int), h, W - h)
    yy, xx = np.mgrid[-h:h, -h:h]
    return ch[:, rows[:, None, None] + yy[None], cols[:, None, None] + xx[None]].transpose(1, 0, 2, 3)
