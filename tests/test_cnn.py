import numpy as np
import torch

from palidar import cnn
from palidar import hearths as Hh


def test_patches_and_forward_shapes():
    z = np.random.default_rng(0).normal(300, 1, (200, 200)).astype(np.float32)
    ch = Hh.patch_channels(z)
    assert ch.shape == (3, 200, 200)
    X = Hh.extract_patches(ch, np.array([100, 30]), np.array([100, 170]))
    assert X.shape == (2, 3, Hh.PATCH, Hh.PATCH)
    m = cnn.HearthNet()
    p = cnn.predict(m, X, tta=True)
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))
    x = torch.zeros(4, 3, 48, 48)
    assert cnn.augment(x).shape == x.shape
