import numpy as np
import pandas as pd

from palidar import moundctx as MC


def test_context_features_flat_vs_slope():
    yy, xx = np.mgrid[0:300, 0:300].astype(np.float32)
    dome = 2.0 * np.exp(-((yy - 150) ** 2 + (xx - 150) ** 2) / (2 * 6.0**2))
    flat = 100 + dome
    slope = 100 + 0.3 * xx + dome
    det = pd.DataFrame(
        dict(
            row=[150],
            col=[150],
            diam=[20.0],
            height=[2.0],
            score=[3.0],
            circ=[1.0],
            solidity=[1.0],
            ecc=[0.0],
            hd=[0.1],
        )
    )
    f = MC.context_features(flat, det)
    s = MC.context_features(slope, det)
    assert set(MC.CTX_FEATS) <= set(f.columns)
    assert f.ann_slope.iloc[0] < s.ann_slope.iloc[0]
    assert MC.size_setting_prior(f).iloc[0] > MC.size_setting_prior(s).iloc[0]
