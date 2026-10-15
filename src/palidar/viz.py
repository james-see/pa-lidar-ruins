"""Terrain visualisations used by the detectors and figures.

* hillshade / multi-directional hillshade (16 azimuths, averaged)
* local relief model (LRM): DEM minus a smoothed DEM, so hills disappear and
  small bumps/hollows (mounds, ditches, hearth platforms) remain
* slope (degrees)
* combo: multi-hillshade modulated by LRM colour (the "look" Nate used)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def fill_nan(dem: np.ndarray) -> np.ndarray:
    m = np.isnan(dem)
    if not m.any():
        return dem
    idx = ndi.distance_transform_edt(m, return_distances=False, return_indices=True)
    return dem[tuple(idx)]


def gradient(dem, res=1.0):
    gy, gx = np.gradient(dem, res)
    return gx, -gy  # x east, y north (rows go south)


def slope_deg(dem, res=1.0):
    gx, gy = gradient(dem, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def hillshade(dem, az=315.0, alt=45.0, res=1.0, z=1.0):
    gx, gy = gradient(dem * z, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)  # direction of steepest descent, from north
    az_r, alt_r = np.radians(az), np.radians(alt)
    hs = np.sin(alt_r) * np.cos(slope) + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect)
    return np.clip(hs, 0, 1)


def multi_hillshade(dem, n=16, alt=35.0, res=1.0, z=2.0):
    return np.mean([hillshade(dem, az, alt, res, z) for az in np.linspace(0, 360, n, endpoint=False)], axis=0)


def smooth(dem, radius_m, res=1.0):
    """Gaussian low-pass; sigma = radius/2 so the kernel spans ~radius each side."""
    return ndi.gaussian_filter(dem, sigma=radius_m / res / 2.0, mode="nearest")


def lrm(dem, radius_m=15.0, res=1.0):
    return dem - smooth(dem, radius_m, res)


def combo_rgb(dem, radius_m=15.0, res=1.0, clip=0.6):
    """RGB: multi-hillshade greyscale tinted by LRM (warm = higher than surroundings)."""
    import matplotlib.cm as cm

    hs = multi_hillshade(dem, res=res)
    l = np.clip(lrm(dem, radius_m, res) / clip, -1, 1) * 0.5 + 0.5
    col = cm.RdYlBu_r(l)[..., :3]
    g = (hs - hs.min()) / (np.ptp(hs) + 1e-9)
    return np.clip(0.55 * col + 0.45 * g[..., None], 0, 1)
