"""Fetch bare-earth DEM windows from the USGS 3DEP ImageServer (best-available, 1 m where flown).

We request a square window in the local UTM zone (NAD83) at 1 m pixels, so every
raster has metric pixels and north-up geometry. Results are cached as compressed GeoTIFFs.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer

IMAGE_SERVER = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
ROOT = Path(__file__).resolve().parents[2]
DEM_DIR = ROOT / "data" / "dem"
MAX_PX = 2000  # server rejects (HTTP 500) some 4000 px requests; tile at 2000


def utm_epsg(lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 26900 + zone  # NAD83 / UTM zone N


def to_utm(lon: float, lat: float, epsg: int | None = None):
    epsg = epsg or utm_epsg(lon)
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = t.transform(lon, lat)
    return x, y, epsg


def _export(xmin, ymin, xmax, ymax, epsg, res, retries=5):
    w = int(round((xmax - xmin) / res))
    h = int(round((ymax - ymin) / res))
    params = dict(
        bbox=f"{xmin},{ymin},{xmax},{ymax}",
        bboxSR=epsg,
        imageSR=epsg,
        size=f"{w},{h}",
        format="tiff",
        pixelType="F32",
        noDataInterpretation="esriNoDataMatchAny",
        interpolation="RSP_BilinearInterpolation",
        f="image",
    )
    last = None
    for i in range(retries):
        try:
            r = requests.get(IMAGE_SERVER, params=params, timeout=(20, 120))
            r.raise_for_status()
            if r.content[:2] not in (b"II", b"MM"):
                raise RuntimeError(f"not a tiff: {r.content[:200]!r}")
            with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
                a = src.read(1).astype("float32")
                nd = src.nodata
            if a.shape != (h, w):
                raise RuntimeError(f"shape {a.shape} != {(h, w)}")
            if nd is not None:
                a[a == nd] = np.nan
            a[a < -1000] = np.nan
            return a
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"export failed after {retries} tries: {last}")


def fetch_window(
    name: str,
    lon: float,
    lat: float,
    half: float = 750.0,
    res: float = 1.0,
    epsg: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch a (2*half) m square DEM centred on lon/lat. Returns path to cached GeoTIFF."""
    cx, cy, epsg = to_utm(lon, lat, epsg)
    cx, cy = round(cx), round(cy)
    return fetch_bbox(name, cx - half, cy - half, cx + half, cy + half, epsg, res, overwrite)


def fetch_bbox(
    name: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    epsg: int,
    res: float = 1.0,
    overwrite: bool = False,
    out_dir: Path | None = None,
) -> Path:
    """Fetch a DEM for a UTM bbox (metres), tiling requests at MAX_PX."""
    d = Path(out_dir) if out_dir else DEM_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.tif"
    if out.exists() and not overwrite:
        return out
    nx = int(round((xmax - xmin) / res))
    ny = int(round((ymax - ymin) / res))
    arr = np.full((ny, nx), np.nan, dtype="float32")
    step = MAX_PX * res
    jobs = []
    for r0 in range(0, ny, MAX_PX):
        for c0 in range(0, nx, MAX_PX):
            x0, y1 = xmin + c0 * res, ymax - r0 * res
            x1, y0 = min(xmax, x0 + step), max(ymin, y1 - step)
            jobs.append((r0, c0, x0, y0, x1, y1))
    from concurrent.futures import ThreadPoolExecutor

    def one(j):
        r0, c0, x0, y0, x1, y1 = j
        return r0, c0, _export(x0, y0, x1, y1, epsg, res)

    with ThreadPoolExecutor(min(8, len(jobs))) as ex:
        for k, (r0, c0, a) in enumerate(ex.map(one, jobs), 1):
            arr[r0 : r0 + a.shape[0], c0 : c0 + a.shape[1]] = a
            if len(jobs) > 4:
                print(f"{name}: tile {k}/{len(jobs)}", flush=True)
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        width=nx,
        height=ny,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as dst:
        dst.write(arr, 1)
    tmp.rename(out)
    return out


def has_1m(lon: float, lat: float, d: float = 0.001) -> list[str]:
    """Return titles of USGS 1 m DEM products overlapping a point (provenance check)."""
    params = {
        "datasets": "Digital Elevation Model (DEM) 1 meter",
        "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}",
        "prodFormats": "GeoTIFF",
        "max": 10,
    }
    for i in range(4):
        try:
            r = requests.get(TNM_API, params=params, timeout=60)
            r.raise_for_status()
            return [it["title"] for it in r.json().get("items", [])]
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return ["<lookup failed>"]


def read(path: Path):
    with rasterio.open(path) as s:
        return s.read(1), s.transform, s.crs
