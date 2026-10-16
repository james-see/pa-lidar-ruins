"""Cross-check reviewed candidates against OSM, GNIS, historic USGS topos (TNM / topoView
GeoTIFFs, oldest available + a mid-century 1:24k) and NAIP imagery.

Input: results/private/review/key.csv + scores.csv (candidates only). Outputs (git-ignored):
results/private/crosscheck/<id>.png (oldest topo | mid-century topo | NAIP | hillshade) and
results/private/crosscheck/crosscheck.csv (nearby OSM/GNIS features, topo editions used).
"""

import io
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from palidar import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
REV = ROOT / "results" / "private" / "review"
OUT = ROOT / "results" / "private" / "crosscheck"
TOPO = ROOT / "data" / "raw" / "topo"
PBF = ROOT / "data" / "raw" / "osm" / "pennsylvania-latest.osm.pbf"
OSMC = ROOT / "data" / "cache" / "osm_context.gpkg"
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
HALF = {"mounds": 300, "works": 500, "hearths": 300}


def osm_context():
    if OSMC.exists():
        return
    pts = pyogrio.read_dataframe(
        PBF, layer="points", where="other_tags LIKE '%historic%' OR man_made IS NOT NULL OR other_tags LIKE '%ruins%'"
    )
    mp = pyogrio.read_dataframe(
        PBF,
        layer="multipolygons",
        where="historic IS NOT NULL OR landuse IS NOT NULL OR leisure IS NOT NULL OR man_made IS NOT NULL "
        "OR military IS NOT NULL OR amenity IS NOT NULL OR tourism IS NOT NULL",
    )
    ln = pyogrio.read_dataframe(
        PBF,
        layer="lines",
        where="highway IN ('track','path','footway','bridleway') OR barrier IS NOT NULL OR man_made IS NOT NULL",
    )
    for name, g in [("points", pts), ("multipolygons", mp), ("lines", ln)]:
        g.to_crs(5070).to_file(OSMC, layer=name)
        print("osm", name, len(g), flush=True)


def describe_osm(layer, g):
    cols = [
        c
        for c in [
            "name",
            "historic",
            "landuse",
            "leisure",
            "man_made",
            "military",
            "amenity",
            "tourism",
            "highway",
            "barrier",
        ]
        if c in g
    ]
    out = []
    for r in g.itertuples():
        tags = {c: getattr(r, c) for c in cols if isinstance(getattr(r, c), str)}
        ot = getattr(r, "other_tags", None)
        if isinstance(ot, str) and "historic" in ot:
            tags["other"] = ot[:80]
        if tags:
            out.append(f"{layer}:{json.dumps(tags)}")
    return out


def topo_maps(lon, lat):
    d = 0.002
    p = dict(
        datasets="Historical Topographic Maps",
        bbox=f"{lon - d},{lat - d},{lon + d},{lat + d}",
        max=200,
        outputFormat="JSON",
    )
    for _ in range(4):
        try:
            items = requests.get(TNM, params=p, timeout=60).json()["items"]
            break
        except Exception:
            time.sleep(5)
    else:
        return None, None
    rows = []
    for it in items:
        u = it.get("urls", {}).get("GeoTIFF")
        if not u:
            continue
        scale = int(u.split("_")[-2]) if u.split("_")[-2].isdigit() else 0
        rows.append(
            dict(
                year=int(it["publicationDate"][:4]),
                scale=scale,
                url=u,
                size=it.get("sizeInBytes", 0),
                title=it["title"],
            )
        )
    m = pd.DataFrame(rows)
    if m.empty:
        return None, None
    m = m[m.scale <= 62500]
    old = m.sort_values(["year", "scale"]).iloc[0] if len(m) else None
    mid = m[(m.scale == 24000) & m.year.between(1940, 1975)].sort_values("year")
    mid = mid.iloc[0] if len(mid) else None
    return old, mid


def get_topo(url):
    f = TOPO / url.split("/")[-1]
    if f.exists() and f.stat().st_size > 0:
        return f
    TOPO.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            r = requests.get(url, timeout=(20, 600))
            r.raise_for_status()
            exp = int(r.headers.get("Content-Length", len(r.content)))
            if len(r.content) != exp:
                raise OSError("truncated")
            f.write_bytes(r.content)
            return f
        except Exception as e:  # noqa: PERF203
            print("retry topo", url, e, flush=True)
            time.sleep(5)
    return None


def topo_chip(f, lon, lat, half):
    with rasterio.open(f) as src:
        x, y = Transformer.from_crs(4326, src.crs, always_xy=True).transform(lon, lat)
        res = abs(src.transform.a)
        h = half * (1.0 if src.crs.is_projected else 1 / 111000)
        win = from_bounds(x - h, y - h, x + h, y + h, src.transform)
        a = src.read(window=win, boundless=True, fill_value=255)
        del res
    if a.shape[0] == 1 and src.colorinterp[0].name == "palette":
        cmap = src.colormap(1)
        lut = np.array([cmap.get(i, (255, 255, 255, 255))[:3] for i in range(256)], np.uint8)
        return lut[a[0]]
    return np.moveaxis(a[:3], 0, -1) if a.shape[0] >= 3 else a[0]


def naip(x, y, epsg, half):
    p = dict(
        bbox=f"{x - half},{y - half},{x + half},{y + half}",
        bboxSR=epsg,
        imageSR=epsg,
        size="500,500",
        format="jpg",
        f="image",
    )
    for _ in range(2):
        try:
            r = requests.get(NAIP, params=p, timeout=(15, 45))
            if r.ok and r.headers.get("Content-Type", "").startswith("image"):
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            time.sleep(3)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    osm_context()
    key = pd.read_csv(REV / "key.csv")
    sc = pd.read_csv(REV / "scores.csv")
    k = key.merge(sc, on="id")
    k = k[(k.src == "candidate") & (k.score >= 1)]  # survivors of the blind review only
    cand = {
        "mounds": pd.read_parquet(S / "cand_mounds.parquet"),
        "works": pd.read_parquet(S / "cand_works.parquet"),
        "hearths": pd.read_csv(ROOT / "results" / "private" / "hearth_fields.csv"),
    }
    H = pd.read_parquet(S / "cand_hearths.parquet")
    gn = pd.read_csv(ROOT / "data" / "raw" / "gnis" / "Text" / "DomesticNames_PA.txt", sep="|", encoding="utf-8-sig")
    gn = gpd.GeoDataFrame(gn, geometry=gpd.points_from_xy(gn.prim_long_dec, gn.prim_lat_dec), crs=4326).to_crs(5070)
    rows = []
    for r in k.itertuples():
        idx = int(r.ref[1:])
        c = cand[r.group]
        if r.group == "hearths":
            c = c[c.field == idx].iloc[0].copy()
            h = H[H.field == idx]
            hh = h.iloc[np.argmin(np.hypot(h.x - c.x, h.y - c.y))]
            c["tile"], c["x"], c["y"] = hh.tile, hh.x, hh.y
        else:
            c = c.loc[idx]
        lon, lat = float(c.lon), float(c.lat)
        x5, y5 = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
        bb = (x5 - 200, y5 - 200, x5 + 200, y5 + 200)
        osm = []
        for layer in ("points", "multipolygons", "lines"):
            g = pyogrio.read_dataframe(OSMC, layer=layer, bbox=bb)
            osm += describe_osm(layer, g)
        near = gn[(gn.geometry.x - x5).abs().lt(1000) & (gn.geometry.y - y5).abs().lt(1000)]
        gnis = [f"{n} ({fc})" for n, fc in zip(near.feature_name, near.feature_class, strict=True)]
        old, mid = topo_maps(lon, lat)
        half = HALF[r.group]
        panels, titles = [], []
        for lab, m in (("oldest", old), ("mid-century 24k", mid)):
            if m is None:
                continue
            f = get_topo(m.url)
            if f is not None:
                panels.append(topo_chip(f, lon, lat, half))
                titles.append(f"{lab}: {m.year} 1:{m.scale}")
        epsg = int(c.epsg)
        x, y = float(c.x), float(c.y)
        im = naip(x, y, epsg, half)
        if im is not None:
            panels.append(im)
            titles.append("NAIP / HRO (current)")
        with rasterio.open(ROOT / "data" / "dem" / "search" / f"{c.tile}.tif") as src:
            row, col = src.index(x, y)
            z = src.read(
                1,
                window=rasterio.windows.Window(col - half, row - half, 2 * half, 2 * half),
                boundless=True,
                fill_value=np.nan,
            )
        panels.append(viz.multi_hillshade(viz.fill_nan(z.astype(np.float32))))
        titles.append("lidar hillshade")
        fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
        for ax, p, t in zip(np.atleast_1d(axs), panels, titles, strict=True):
            ax.imshow(p, cmap="gray" if p.ndim == 2 else None)
            ax.plot(p.shape[1] / 2, p.shape[0] / 2, "r+", ms=14)
            ax.set_title(f"{r.id} {t}", fontsize=9)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUT / f"{r.id}.png", dpi=80)
        plt.close(fig)
        rows.append(
            dict(
                id=r.id,
                group=r.group,
                ref=r.ref,
                county=c.county,
                review_score=r.score,
                osm_within_200m="; ".join(osm)[:600],
                gnis_within_1km="; ".join(gnis)[:400],
                topo_oldest=None if old is None else f"{old.year} 1:{old.scale}",
                topo_mid=None if mid is None else f"{mid.year} 1:{mid.scale}",
                naip=im is not None,
            )
        )
        print(r.id, r.group, c.county, len(osm), "osm", len(gnis), "gnis", titles, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "crosscheck.csv", index=False)


if __name__ == "__main__":
    main()
