"""Tiling and masking helpers for the wide-area search."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box

from . import dem

ROOT = Path(__file__).resolve().parents[2]
AOI_DIR = ROOT / "data" / "cache" / "aoi"
CRS = 5070
FIXED_BUF = {"mined": 0, "quarry": 0, "rail": 15, "water": 10, "buildings": 10}  # mined/quarry pre-buffered


def make_tiles(aoi: gpd.GeoDataFrame, size=4000) -> pd.DataFrame:
    """Square UTM tiles (size m) covering each AOI polygon. Returns one row per (aoi, tile)."""
    rows = []
    for a in aoi.itertuples():
        g = gpd.GeoSeries([a.geometry], crs=aoi.crs)
        lon = g.to_crs(4326).iloc[0].centroid.x
        epsg = dem.utm_epsg(lon)
        gu = g.to_crs(epsg).iloc[0]
        x0, y0, x1, y1 = gu.bounds
        for x in np.arange(np.floor(x0 / size) * size, x1, size):
            for y in np.arange(np.floor(y0 / size) * size, y1, size):
                b = box(x, y, x + size, y + size)
                inter = gu.intersection(b).area
                if inter > 5000:
                    rows.append(
                        dict(
                            aoi=a.aoi,
                            unit=a.unit,
                            epsg=epsg,
                            x0=int(x),
                            y0=int(y),
                            size=size,
                            aoi_km2=inter / 1e6,
                            tile=f"{epsg}_{int(x)}_{int(y)}",
                        )
                    )
    return pd.DataFrame(rows)


def _read(layer, bbox5070):
    return pyogrio.read_dataframe(AOI_DIR / "excl.gpkg", layer=layer, bbox=bbox5070)


def tile_mask(t, aoi_geom_5070, pad=60, res=1.0):
    """Boolean raster (True = searchable) for a tile incl. padding: inside AOI public land and
    outside every exclusion layer."""
    epsg = t["epsg"]
    x0, y0, size = t["x0"] - pad, t["y0"] - pad, t["size"] + 2 * pad
    n = int(size / res)
    tr = from_origin(x0, y0 + size, res, res)
    tb = gpd.GeoSeries([box(x0, y0, x0 + size, y0 + size)], crs=epsg)
    bb = tuple(tb.to_crs(CRS).total_bounds)
    aoi_u = gpd.GeoSeries([aoi_geom_5070], crs=CRS).clip(box(*bb)).to_crs(epsg)
    allowed = features.rasterize(
        [(g, 1) for g in aoi_u if g is not None and not g.is_empty],
        out_shape=(n, n),
        transform=tr,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    shapes = []
    for layer in ("mined", "quarry", "rail", "water", "buildings", "roads", "waterways"):
        g = _read(layer, bb)
        if g.empty:
            continue
        g = g.to_crs(epsg)
        if layer in ("roads", "waterways"):
            geoms = g.geometry.buffer(g["w"].to_numpy())
        else:
            b = FIXED_BUF[layer]
            geoms = g.geometry.buffer(b) if b else g.geometry
        shapes += [(gg, 1) for gg in geoms if gg is not None and not gg.is_empty]
    if shapes:
        ex = features.rasterize(shapes, out_shape=(n, n), transform=tr, fill=0, dtype="uint8").astype(bool)
        allowed &= ~ex
    return allowed, tr
