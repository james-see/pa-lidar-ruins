"""Classical (no ML) detectors on 1 m bare-earth DEMs.

All detectors take a DEM array (metres, 1 m pixels, NaNs allowed) and return candidates
in pixel coordinates (row, col) with a score and morphometric attributes.

A. mounds        multi-scale local relief model (LRM) blobs: compact, round, raised
B. hearths       relict charcoal hearth (RCH) platforms: small round FLAT spots cut into a
                 slope (flat disc inside a sloping annulus), checked for roundness
C. linear works  berm+ditch pairs (Hessian ridge next to Hessian valley) summarised per
                 100 m tile with an orientation-diversity term (forts turn corners,
                 roads and streams mostly don't)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, peak_local_max

from . import viz


def _disk(r):
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(float)


def _ring(r0, r1):
    y, x = np.ogrid[-r1 : r1 + 1, -r1 : r1 + 1]
    d = x * x + y * y
    return ((d > r0 * r0) & (d <= r1 * r1)).astype(float)


def _mean_filter(a, k):
    k = k / k.sum()
    from scipy.signal import fftconvolve

    return fftconvolve(a, k, mode="same")


# ----------------------------------------------------------------------------- A. mounds
MOUND_SCALES = ((12, 0.25), (25, 0.35), (50, 0.5), (100, 0.8))  # (LRM radius m, min relief m)


def detect_mounds(z, res=1.0, scales=MOUND_SCALES, min_diam=6, max_diam=110, min_solidity=0.85, max_ecc=0.85):
    z = viz.fill_nan(z)
    rows = []
    for rad, thr in scales:
        l = viz.lrm(z, rad, res)
        lab = measure.label(l > thr)
        for p in measure.regionprops(lab, intensity_image=l):
            d = p.equivalent_diameter_area * res
            if d < min_diam or d > max_diam or d > 1.6 * rad:
                continue
            if p.solidity < min_solidity or p.eccentricity > max_ecc:
                continue
            h = float(p.intensity_max)
            per = max(p.perimeter, 1)
            circ = min(1.0, 4 * np.pi * p.area / per**2)
            r, c = np.unravel_index(np.argmax(np.where(lab == p.label, l, -1e9)[p.slice]), p.image.shape)
            r += p.slice[0].start
            c += p.slice[1].start
            # relief relative to diameter: real mounds are domes, not flat swells
            shape_ok = h / d
            score = h * circ * p.solidity * min(1.0, shape_ok / 0.03)
            rows.append(
                dict(
                    row=r,
                    col=c,
                    scale=rad,
                    diam=d,
                    height=h,
                    circ=circ,
                    solidity=p.solidity,
                    ecc=p.eccentricity,
                    hd=shape_ok,
                    score=score,
                )
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return nms(df, 0.6)


def nms(df, frac=0.6, min_dist=None):
    """Greedy non-max suppression: drop candidates closer than frac*diam (or min_dist) to a stronger one."""
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    keep = []
    pts = df[["row", "col"]].to_numpy(float)
    rad = (df["diam"].to_numpy() * frac) if min_dist is None else np.full(len(df), min_dist)
    taken = np.zeros(len(df), bool)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    for i in range(len(df)):
        if taken[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], max(rad[i], 1)):
            taken[j] = True
    return df.iloc[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- B. hearths
HEARTH_RADII = (4, 5, 6, 7)  # platform radius in m (RCH platforms ~8-15 m across)


def detect_hearths(
    z, res=1.0, radii=HEARTH_RADII, max_inner_slope=7.0, min_contrast=6.0, min_dist=9, return_maps=False
):
    z = viz.fill_nan(z)
    zs = ndi.gaussian_filter(z, 1.0)
    s = viz.slope_deg(zs, res)
    best = np.full(z.shape, -np.inf, dtype="float32")
    best_r = np.zeros(z.shape, dtype="int8")
    inner_at = np.zeros(z.shape, dtype="float32")
    for r in radii:
        sin = _mean_filter(s, _disk(r))
        sout = _mean_filter(s, _ring(int(r * 1.5), int(r * 2.5)))
        # platform also has to be planar: low slope variability inside
        sc = sout - sin
        sc[sin > max_inner_slope] = -np.inf
        upd = sc > best
        best[upd] = sc[upd]
        best_r[upd] = r
        inner_at[upd] = sin[upd]
    pk = peak_local_max(
        np.where(np.isfinite(best), best, -1),
        min_distance=min_dist,
        threshold_abs=min_contrast,
        exclude_border=10,
    )
    rows = []
    for r, c in pk:
        rr = int(best_r[r, c])
        # roundness of the flat patch around the peak
        w = int(rr * 2.5) + 2
        r0, r1, c0, c1 = max(0, r - w), r + w + 1, max(0, c - w), c + w + 1
        flat = s[r0:r1, c0:c1] < (inner_at[r, c] + 4.0)
        lab = measure.label(flat)
        lr, lc = r - r0, c - c0
        if lab[lr, lc] == 0:
            continue
        p = [q for q in measure.regionprops(lab) if q.label == lab[lr, lc]][0]
        touches = p.bbox[0] == 0 or p.bbox[1] == 0 or p.bbox[2] == flat.shape[0] or p.bbox[3] == flat.shape[1]
        d = p.equivalent_diameter_area * res
        rows.append(
            dict(
                row=r,
                col=c,
                radius=rr,
                contrast=float(best[r, c]),
                inner_slope=float(inner_at[r, c]),
                outer_slope=float(best[r, c] + inner_at[r, c]),
                flat_diam=d,
                flat_ecc=p.eccentricity,
                flat_touches=touches,
                diam=2 * rr,
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return (df, best) if return_maps else df
    # roundness: elongated flats (roads, benches) or flats leaking out of the window (fields) are penalised
    round_pen = np.where(df.flat_touches, 0.3, 1.0) * np.clip(1.3 - df.flat_ecc, 0.2, 1.0)
    size_pen = np.where((df.flat_diam >= 6) & (df.flat_diam <= 22), 1.0, 0.4)
    df["score"] = df.contrast * round_pen * size_pen
    return (df, best) if return_maps else df


# ----------------------------------------------------------------------------- C. linear earthworks
def berm_ditch_maps(z, res=1.0, sigmas=(1.5, 3.0), lrm_radius=12, ridge_thr=0.015, pair_dist=7):
    z = viz.fill_nan(z)
    l = viz.lrm(z, lrm_radius, res)
    ridge = np.zeros(z.shape, bool)
    valley = np.zeros(z.shape, bool)
    orient = np.zeros(z.shape, dtype="float32")
    strength = np.zeros(z.shape, dtype="float32")
    for s in sigmas:
        H = hessian_matrix(l, sigma=s, order="rc", use_gaussian_derivatives=True)
        e1, e2 = hessian_matrix_eigvals(H)  # e1 >= e2
        Hrr, Hrc, Hcc = H
        r = (-e2 * s * s) > ridge_thr  # strong negative curvature across = ridge
        v = (e1 * s * s) > ridge_thr
        ridge |= r & (l > 0.15)
        valley |= v & (l < -0.15)
        st = np.maximum(-e2, e1) * s * s
        upd = st > strength
        strength[upd] = st[upd]
        # orientation of the feature (perpendicular to principal curvature direction)
        th = 0.5 * np.arctan2(2 * Hrc, Hrr - Hcc)
        orient[upd] = th[upd]
    k = _disk(pair_dist) > 0
    pair = (ridge & ndi.binary_dilation(valley, k)) | (valley & ndi.binary_dilation(ridge, k))
    return dict(lrm=l, ridge=ridge, valley=valley, pair=pair, orient=orient)


def tile_scores(z, res=1.0, tile=100, maps=None):
    """Per-tile linear-earthwork score. Returns DataFrame (tile_r, tile_c, density, odiv, score)."""
    m = maps or berm_ditch_maps(z, res)
    pair, orient = m["pair"], m["orient"]
    H, W = pair.shape
    rows = []
    for i in range(0, H - tile + 1, tile):
        for j in range(0, W - tile + 1, tile):
            p = pair[i : i + tile, j : j + tile]
            dens = p.mean()
            if p.sum() < 30:
                odiv = 0.0
            else:
                th = orient[i : i + tile, j : j + tile][p]
                # orientation diversity: 1 - |mean resultant of doubled angles|
                odiv = 1 - np.abs(np.mean(np.exp(2j * th)))
            nz = m["lrm"][i : i + tile, j : j + tile]
            rough = float(np.nanstd(nz))
            rows.append(
                dict(
                    tile_r=i // tile,
                    tile_c=j // tile,
                    density=dens,
                    odiv=odiv,
                    rough=rough,
                    score=dens * (0.3 + odiv),
                )
            )
    return pd.DataFrame(rows)
