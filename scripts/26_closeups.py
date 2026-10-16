"""Lidar close-ups of reviewed candidates (outputs private: they carry locations)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from palidar import dem, viz

OUT = Path("results/private/closeups")
OUT.mkdir(parents=True, exist_ok=True)
SITES = [
    ("01_Dauphin_zigzag", 40.592697, -76.622682, 150),
    ("02_McKean_knobs", 41.801368, -78.934686, 150),
    ("03_Warren_dome", 41.811868, -79.110304, 100),
    ("04_Cumberland_hearths_1", 40.037364, -77.399829, 200),
    ("05_Cumberland_hearths_2", 40.050724, -77.227523, 200),
    ("06_Cumberland_hearths_3", 40.048515, -77.302839, 200),
    ("07_Cumberland_hearths_4", 40.020875, -77.282893, 200),
    ("08_Franklin_OldForge_hearths", 39.786708, -77.479705, 200),
    ("09_Franklin_hearths_2", 39.800286, -77.499592, 200),
    ("10_Adams_hearths", 39.827833, -77.44493, 200),
    ("11_Huntingdon_hearths_1", 40.617624, -78.06751, 200),
    ("12_Huntingdon_hearths_2", 40.664867, -77.755854, 200),
]
for name, lat, lon, half in SITES:
    p = dem.fetch_window(f"closeup_{name}", lon, lat, half=half)
    z = dem.read(p)[0]
    z = viz.fill_nan(np.asarray(z, dtype=float))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.3))
    ax[0].imshow(viz.multi_hillshade(z), cmap="gray")
    ax[0].set_title("16-direction hillshade")
    ax[1].imshow(viz.combo_rgb(z))
    ax[1].set_title("local relief (red = higher than surroundings)")
    for a in ax:
        a.plot(z.shape[1] / 2, z.shape[0] / 2, "+", color="cyan", ms=16, mew=1.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{name.replace('_', ' ')}  |  {2 * half} m across, north up, USGS 3DEP 1 m")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=110)
    plt.close(fig)
    print("ok", name, z.shape)
