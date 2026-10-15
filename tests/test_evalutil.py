import numpy as np

from palidar import evalutil as E


def test_split_is_deterministic():
    assert E.split_of("site_M01") == "dev"
    assert E.split_of("site_M02") == "test"
    assert E.split_of("bg_10") == "test"


def test_recall_fp_curve():
    pos = np.array([1.0, 2.0, np.nan, 4.0])
    neg = np.array([0.5, 1.5, 3.0])
    c = E.recall_fp_curve(pos, neg, neg_area_km2=2.0, thresholds=[1.0, 3.0])
    assert c.recall.tolist() == [0.75, 0.25]
    assert c.fp_per_km2.tolist() == [1.0, 0.5]
