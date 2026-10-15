"""Synthetic-terrain tests: each detector must find a planted feature and ignore flat/tilted ground."""

import numpy as np

from palidar import detectors as D
from palidar import viz


def _plane(n=300, tilt=0.0):
    y, x = np.mgrid[0:n, 0:n].astype("float32")
    return 100 + tilt * y


def test_lrm_removes_plane():
    z = _plane(tilt=0.2)
    l = viz.lrm(z, 15)
    assert np.abs(l[50:-50, 50:-50]).max() < 1e-3


def test_mound_found_at_centre():
    z = _plane()
    y, x = np.mgrid[0:300, 0:300]
    r = np.hypot(y - 150, x - 150)
    z = z + np.clip(3.0 * (1 - (r / 15) ** 2), 0, None)  # 30 m wide, 3 m high dome
    d = D.detect_mounds(z)
    assert len(d) >= 1
    top = d.iloc[0]
    assert np.hypot(top.row - 150, top.col - 150) < 5
    assert 20 < top.diam < 40


def test_no_mound_on_tilted_plane():
    assert D.detect_mounds(_plane(tilt=0.1)).empty


def test_hearth_platform_on_slope():
    z = _plane(tilt=0.25)  # ~14 degree hillside
    y, x = np.mgrid[0:300, 0:300]
    r = np.hypot(y - 150, x - 150)
    z = np.where(r <= 6, z[150, 150], z)  # cut/fill flat disc 12 m across
    d = D.detect_hearths(z)
    assert len(d) >= 1
    top = d.sort_values("score", ascending=False).iloc[0]
    assert np.hypot(top.row - 150, top.col - 150) < 4


def test_berm_ditch_pair():
    z = _plane()
    y, x = np.mgrid[0:300, 0:300].astype(float)
    z = z + 1.2 * np.exp(-(((x - 150) / 2.5) ** 2)) - 1.0 * np.exp(-(((x - 156) / 2.0) ** 2))
    m = D.berm_ditch_maps(z)
    assert m["pair"][50:250, 140:165].mean() > 0.05
    assert m["pair"][50:250, :100].mean() == 0
