"""Blinded skeptic-review sheets: top search candidates mixed with hidden known-site controls and
negative controls (random-PA and strip-mine windows), shuffled under opaque ids.

Writes (all git-ignored): results/private/review/<group>_sheet_XX.png (6 chips per sheet, id only),
results/private/review/key.csv (id -> source, never shown to the reviewer before scoring).
usage: uv run python scripts/19_review_chips.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from palidar import evalutil as E  # noqa: E402
from palidar import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
OUT = ROOT / "results" / "private" / "review"
DEM = ROOT / "data" / "dem"
rng = np.random.default_rng(42)
GROUPS = {  # chip size m, LRM radius m, LRM clip m
    "mounds": (240, 25, 1.2),
    "works": (500, 25, 0.8),
    "hearths": (200, 10, 0.5),
}


def chip(path, size, r, x=None, y=None):
    m = 2 * r
    with rasterio.open(path) as src:
        if x is None:
            row, col = src.height // 2, src.width // 2
        else:
            row, col = src.index(x, y)
        h = size // 2 + m
        z = src.read(1, window=Window(col - h, row - h, 2 * h, 2 * h), boundless=True, fill_value=np.nan)
    z = viz.fill_nan(z.astype(np.float32))
    lr = viz.lrm(z, r)[m:-m, m:-m]
    hs = viz.multi_hillshade(z)[m:-m, m:-m]
    return lr, hs


def items():
    it = []
    w = E.windows()
    ks = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv")
    mo = pd.read_parquet(S / "cand_mounds.parquet")
    mo = mo.groupby("county").head(5).sort_values("score", ascending=False).head(20)
    for r in mo.itertuples():
        it.append(
            dict(
                group="mounds", src="candidate", ref=f"m{r.Index}", path=DEM / "search" / f"{r.tile}.tif", x=r.x, y=r.y
            )
        )
    wk = pd.read_parquet(S / "cand_works.parquet").head(20)
    for r in wk.itertuples():
        it.append(
            dict(group="works", src="candidate", ref=f"w{r.Index}", path=DEM / "search" / f"{r.tile}.tif", x=r.x, y=r.y)
        )
    F = pd.read_csv(ROOT / "results" / "private" / "hearth_fields.csv")
    F = F[F.frac_known < 0.2].sort_values("n", ascending=False).head(10)
    H = pd.read_parquet(S / "cand_hearths.parquet")
    for r in F.itertuples():
        h = H[H.field == r.field]
        k = np.argmin(np.hypot(h.x - r.x, h.y - r.y))
        hh = h.iloc[k]
        it.append(
            dict(
                group="hearths",
                src="candidate",
                ref=f"f{r.field}",
                path=DEM / "search" / f"{hh.tile}.tif",
                x=hh.x,
                y=hh.y,
            )
        )
    # hidden positive controls
    mc = ks[(ks.cls == "mound") & ks.visual_audit.isin(["clear_dome", "faint"])].sample(8, random_state=1)
    for r in mc.itertuples():
        it.append(dict(group="mounds", src="control_known", ref=r.site_id, path=DEM / f"site_{r.site_id}.tif"))
    wc = pd.concat(
        [
            ks[(ks.cls == "military") & (ks.visual_audit == "clear")].sample(6, random_state=1),
            ks[(ks.cls == "military") & (ks.visual_audit == "faint")].sample(3, random_state=1),
            ks[(ks.cls == "enclosure") & (ks.visual_audit == "clear")],
        ]
    )
    for r in wc.itertuples():
        it.append(dict(group="works", src="control_known", ref=r.site_id, path=DEM / f"site_{r.site_id}.tif"))
    src = rasterio.open(DEM / "rch_sgl43_full.tif")
    m = gpd.read_file(ROOT / "data" / "raw" / "rch" / "rch_master_final.geojson")
    m = m[(m.SGL.astype(str) == "43") & (m.Confirm == "Y")].to_crs(src.crs).sample(5, random_state=1)
    for i, g in enumerate(m.geometry):
        it.append(
            dict(group="hearths", src="control_known", ref=f"sgl43_Y{i}", path=DEM / "rch_sgl43_full.tif", x=g.x, y=g.y)
        )
    # negative controls: window centres of random-PA and strip-mine windows
    bg = w[w.kind == "bg"].sample(18, random_state=2).win.tolist()
    aml = w[w.kind == "aml"].sample(12, random_state=2).win.tolist()
    for gi, grp in enumerate(GROUPS):
        for wn in bg[gi * 6 : gi * 6 + 6] + aml[gi * 4 : gi * 4 + 4]:
            it.append(dict(group=grp, src="control_negative", ref=wn, path=DEM / f"{wn}.tif"))
    return pd.DataFrame(it)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I = items()
    I = I.sample(frac=1, random_state=7).reset_index(drop=True)
    I["id"] = [f"R{i + 1:03d}" for i in range(len(I))]
    I.drop(columns="path").to_csv(OUT / "key.csv", index=False)
    for grp, (size, r, clip) in GROUPS.items():
        g = I[I.group == grp]
        for s in range(0, len(g), 6):
            sub = g.iloc[s : s + 6]
            fig, axs = plt.subplots(len(sub), 2, figsize=(7, 3.4 * len(sub)))
            axs = np.atleast_2d(axs)
            for ax, rr in zip(axs, sub.itertuples()):
                lr, hs = chip(rr.path, size, r, None if pd.isna(rr.x) else rr.x, None if pd.isna(rr.y) else rr.y)
                ax[0].imshow(lr, cmap="gray", vmin=-clip, vmax=clip)
                ax[1].imshow(hs, cmap="gray")
                ax[0].set_title(f"{rr.id}  LRM r={r} m (+-{clip} m)", fontsize=9)
                ax[1].set_title(f"{rr.id}  multi-hillshade, {size} m wide", fontsize=9)
                for a in ax:
                    a.set_xticks([])
                    a.set_yticks([])
            fig.tight_layout()
            fig.savefig(OUT / f"{grp}_sheet_{s // 6 + 1:02d}.png", dpi=80)
            plt.close(fig)
    print(I.groupby(["group", "src"]).size())


if __name__ == "__main__":
    main()
