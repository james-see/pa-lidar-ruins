"""Phase 2 wide-area search, pass A (CPU, parallel): DEM fetch, masking, detectors.

usage: uv run python scripts/15_search.py hearths|mounds|works [--units "Michaux SF,..."] [--workers 8]

Per 4 km UTM tile (60 m padding): fetch 1 m 3DEP DEM, rasterise the searchable mask (AOI public
land minus exclusions), then
  hearths -> phase-1 candidates + features + GBM probability (CNN is applied in pass B, 16_*.py)
  mounds  -> detect_mounds, candidates with score >= 1
  works   -> berm+ditch 100 m tile scores (multi scales) with masked pixels removed
Outputs (with coordinates) go to data/private/search/<class>/ -- git-ignored, never published.
Coverage (searchable km2 per unit) -> data/private/search/<class>/coverage.csv
"""

import argparse
import re
import time
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio

from palidar import dem
from palidar import detectors as D
from palidar import hearths as Hh
from palidar import search as S

ROOT = Path(__file__).resolve().parents[1]
PRIV = ROOT / "data" / "private" / "search"
DEMD = ROOT / "data" / "dem" / "search"
PAD = 60
WORKS_CFG = dict(sigmas=(1.5, 3.0, 6.0), lrm_radius=25)
_G = {}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def init(cls):
    _G["cls"] = cls
    A = gpd.read_file(S.AOI_DIR / "aois.gpkg")
    _G["aoi"] = {r.unit: r.geometry for r in A.itertuples()}
    if cls == "hearths":
        _G["clf"], _G["feats"] = joblib.load(ROOT / "data" / "cache" / "hearth_model.joblib")


def run_tile(t):
    cls = _G["cls"]
    out = PRIV / cls / f"{t['tile']}__{slug(t['unit'])}.parquet"
    if out.exists():
        return t["tile"], "cached", None
    t0 = time.time()
    p = dem.fetch_bbox(
        t["tile"],
        t["x0"] - PAD,
        t["y0"] - PAD,
        t["x0"] + t["size"] + PAD,
        t["y0"] + t["size"] + PAD,
        int(t["epsg"]),
        out_dir=DEMD,
    )
    with rasterio.open(p) as src:
        z = src.read(1)
        tr = src.transform
    nan_frac = float(np.isnan(z).mean())
    allowed, _ = S.tile_mask(t, _G["aoi"][t["unit"]], pad=PAD)
    allowed &= ~np.isnan(z)
    inner = np.zeros_like(allowed)
    inner[PAD:-PAD, PAD:-PAD] = True
    cov_km2 = float((allowed & inner).sum()) / 1e6
    res = None
    if cov_km2 > 0.01:
        if cls == "hearths":
            c = Hh.candidates_with_features(z)
            if not c.empty:
                r, cc = c.row.to_numpy().astype(int), c.col.to_numpy().astype(int)
                keep = allowed[r, cc] & inner[r, cc]
                c = c[keep].reset_index(drop=True)
                c["p_gbm"] = _G["clf"].predict_proba(c[_G["feats"]])[:, 1] if len(c) else []
                res = c
        elif cls == "mounds":
            d = D.detect_mounds(z)
            if not d.empty:
                r, cc = d.row.to_numpy().astype(int), d.col.to_numpy().astype(int)
                d = d[allowed[r, cc] & inner[r, cc] & (d.score >= 1.0)].reset_index(drop=True)
                res = d
        elif cls == "works":
            ts = D.tile_scores(z, tile=100, allowed=allowed, **WORKS_CFG)
            ts["row"] = ts.tile_r * 100 + 50
            ts["col"] = ts.tile_c * 100 + 50
            r, cc = ts.row.to_numpy(), ts.col.to_numpy()
            ok = (r >= PAD) & (r < z.shape[0] - PAD) & (cc >= PAD) & (cc < z.shape[1] - PAD) & (ts.allowed >= 0.5)
            res = ts[ok].reset_index(drop=True)
    if res is None:
        res = pd.DataFrame(dict(row=[], col=[]))
    xs, ys = rasterio.transform.xy(tr, res.row.to_numpy(), res.col.to_numpy()) if len(res) else ([], [])
    res["x"], res["y"], res["epsg"] = np.asarray(xs, float), np.asarray(ys, float), int(t["epsg"])
    res["tile"], res["unit"] = t["tile"], t["unit"]
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out)
    out.with_suffix(".cov").write_text(f"{cov_km2:.6f}")
    return t["tile"], f"{len(res)} rows, cov {cov_km2:.2f} km2, nan {nan_frac:.2f}, {time.time() - t0:.0f}s", cov_km2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cls", choices=["hearths", "mounds", "works"])
    ap.add_argument("--units", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    T = pd.read_csv(S.AOI_DIR / "tiles.csv")
    T = T[T.aoi == a.cls]
    if a.units:
        T = T[T.unit.isin([u.strip() for u in a.units.split(",")])]
    print(a.cls, len(T), "tiles,", round(T.aoi_km2.sum(), 1), "km2 of AOI", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.cls,)) as pool:
        for i, (tile, msg, _) in enumerate(pool.imap_unordered(run_tile, T.to_dict("records")), 1):
            print(f"[{time.time() - t0:6.0f}s] {i}/{len(T)} {tile}: {msg}", flush=True)


if __name__ == "__main__":
    main()
