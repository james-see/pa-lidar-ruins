"""Render a blind grid of 80 m chips for visual audit of hearth detections in the SGL 43 test half.

Mixes a random sample of UNMATCHED detections (p >= thr) with matched ones (positive controls) in
shuffled order, writes the key separately. Output goes to results/private/ (git-ignored).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from palidar import viz

ROOT = Path(__file__).resolve().parents[1]
PRIV = ROOT / "results" / "private"
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
d = pd.read_csv(PRIV / "sgl43_test_dets.csv")
d = d[d.p >= thr]
rng = np.random.default_rng(1)
un = d[~d.matched].sample(30, random_state=1).assign(kind="unmatched")
ma = d[d.matched].sample(10, random_state=1).assign(kind="matched")
s = pd.concat([un, ma]).sample(frac=1, random_state=2).reset_index(drop=True)
s.index.name = "chip"
s[["kind", "p"]].to_csv(PRIV / "audit_key.csv")
with rasterio.open(ROOT / "data" / "dem" / "rch_sgl43_full.tif") as src:
    z = src.read(1)
fig, axs = plt.subplots(5, 8, figsize=(16, 10))
for ax, (i, r) in zip(axs.flat, s.iterrows(), strict=False):
    h = 40
    sub = viz.fill_nan(z[int(r.row) - h : int(r.row) + h, int(r.col) - h : int(r.col) + h])
    ax.imshow(viz.combo_rgb(sub, 10, clip=0.4))
    ax.set_title(str(i), fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(PRIV / "audit_grid.png", dpi=70)
print(PRIV / "audit_grid.png")
