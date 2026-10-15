"""Render combo (multi-hillshade + LRM) chips for window centres, for visual QA.

usage: python scripts/03_chips.py [glob] [--size 400] [--radius 15]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from palidar import dem, viz

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("pattern", default="site_*", nargs="?")
ap.add_argument("--size", type=int, default=400)
ap.add_argument("--radius", type=float, default=15)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--out", default="results/chips/qa.png")
a = ap.parse_args()
files = sorted((ROOT / "data" / "dem").glob(a.pattern + ".tif"))
n = len(files)
cols = min(a.cols, n)
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
for ax in axs.flat:
    ax.axis("off")
for ax, f in zip(axs.flat, files):
    z, _, _ = dem.read(f)
    z = viz.fill_nan(z)
    c = z.shape[0] // 2, z.shape[1] // 2
    h = a.size // 2
    sub = z[c[0] - h : c[0] + h, c[1] - h : c[1] + h]
    ax.imshow(viz.combo_rgb(sub, a.radius))
    ax.plot([h], [h], "c+", ms=10)
    ax.set_title(f.stem, fontsize=8)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(a.out, dpi=70)
print(a.out, n)
