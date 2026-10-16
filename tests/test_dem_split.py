import numpy as np

from palidar import dem


def test_export_split_stitches_quadrants(monkeypatch):
    def fake(xmin, ymin, xmax, ymax, epsg, res):
        w, h = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
        if max(w, h) > 600:
            raise RuntimeError("504")
        yy, xx = np.mgrid[0:h, 0:w]
        return (ymax - yy * res) * 1000 + (xmin + xx * res)  # encodes position

    monkeypatch.setattr(dem, "_export", fake)
    a = dem._export_split(0, 0, 1000, 1000, 26918, 1.0)
    assert a.shape == (1000, 1000)
    yy, xx = np.mgrid[0:1000, 0:1000]
    assert np.array_equal(a, (1000 - yy) * 1000 + xx)
