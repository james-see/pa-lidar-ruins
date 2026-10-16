"""Small CNN that scores relict-charcoal-hearth candidates from 48 x 48 m, 3-channel patches."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class HearthNet(nn.Module):
    def __init__(self, c=3, w=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.f = nn.Sequential(
            blk(c, w),
            nn.MaxPool2d(2),
            blk(w, 2 * w),
            nn.MaxPool2d(2),
            blk(2 * w, 4 * w),
            nn.MaxPool2d(2),
            blk(4 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, 1))

    def forward(self, x):
        return self.head(self.f(x)).squeeze(1)


def augment(x: torch.Tensor) -> torch.Tensor:
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


@torch.no_grad()
def predict(model, X: np.ndarray, batch=4096, tta=True) -> np.ndarray:
    """Probability per patch; test-time augmentation averages the 4 rotations."""
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32)).to(dev)
        if tta:
            logits = torch.stack([model(torch.rot90(xb, k, dims=(2, 3))) for k in range(4)]).mean(0)
        else:
            logits = model(xb)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load(path):
    m = HearthNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m
