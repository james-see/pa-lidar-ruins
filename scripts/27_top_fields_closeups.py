"""Lidar close-ups + KML for the top new hearth fields (outputs private: they carry locations)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palidar import dem, viz

OUT = Path("results/private/closeups_top24")
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_csv("results/private/hearth_fields_statewide.csv")
top = f[(f.frac_known < 0.2) & (f.nearest_gnis_furnace_km > 5) & (f.n >= 50)].sort_values("n", ascending=False)
h = pd.read_parquet("data/private/search/cand_hearths_statewide.parquet")
pms = []
for rank, r in enumerate(top.itertuples(), 1):
    pts = h[h.field == r.field]
    half = 450.0
    # centre on the densest spot: the hearth with the most neighbours within 200 m
    xy = pts[["x", "y"]].to_numpy()
    nb = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1) < 200**2
    cx, cy = (round(v) for v in xy[nb.sum(1).argmax()])
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(int(r.epsg), 4326, always_xy=True).transform(cx, cy)
    p = dem.fetch_bbox(f"top24_{r.field}", cx - half, cy - half, cx + half, cy + half, int(r.epsg), 1.0, overwrite=True)
    z, tr, _ = dem.read(p)
    z = viz.fill_nan(np.asarray(z, dtype=float))
    col = (pts.x.values - tr.c) / tr.a
    row = (pts.y.values - tr.f) / tr.e
    name = f"{rank:02d}_{r.county}_{r.unit.replace('rest ', '').replace(' ', '')}_{r.n}hearths"
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.3))
    ax[0].imshow(viz.multi_hillshade(z), cmap="gray")
    ax[0].set_title("16-direction hillshade")
    ax[1].imshow(viz.combo_rgb(z))
    inview = (col >= 0) & (col < z.shape[1]) & (row >= 0) & (row < z.shape[0])
    ax[1].scatter(col[inview], row[inview], s=40, facecolors="none", edgecolors="cyan", linewidths=0.8)
    ax[1].set_xlim(0, z.shape[1]); ax[1].set_ylim(z.shape[0], 0)
    ax[1].set_title(f"local relief + CNN hearths (cyan): {inview.sum()} of {len(pts)} in view")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        f"#{rank} {r.county} ({r.unit})  |  {r.n} hearths, {r.frac_known:.0%} in Carter et al., "
        f"nearest GNIS furnace {r.nearest_gnis_furnace_km:.1f} km ({r.nearest_gnis_furnace})  |  {2 * half:.0f} m across, north up"
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=100)
    plt.close(fig)
    pms.append(
        f"<Placemark><name>#{rank} {r.county} ({r.unit}), {r.n} hearths</name>"
        f"<LookAt><longitude>{lon}</longitude><latitude>{lat}</latitude><range>{3 * half:.0f}</range></LookAt>"
        f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
    )
    print("ok", name)
(Path("results/private") / "pa_top24_hearth_fields.kml").write_text(
    '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
    "<name>PA top 24 new hearth fields (private)</name>" + "".join(pms) + "</Document></kml>"
)
