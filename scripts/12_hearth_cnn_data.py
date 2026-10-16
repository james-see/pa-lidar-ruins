"""Build training/test patches for the hearth CNN from the 4,361 manual RCH points (Carter et al.)
plus the SGL 43 closed-world reference.

Spatial holdout: points are grouped into 20 km blocks; every block that touches one of the phase-1
recall windows (rch_rec_*) is forced into TEST, plus a seeded random 20% of the rest. SGL 43 west
half = train, east half = test (same split as phase 1), so the GBM and CNN can be compared like for like.

For every 2 km cell containing manual points we fetch a 1 m DEM (cached in data/dem/rchtrain/), run
the phase-1 candidate generator, and label candidates: positive if within 6 m of a reference point,
negative if > 20 m from every reference point (manual labels are not exhaustive -> some label noise).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from palidar import dem
from palidar import evalutil as E
from palidar import hearths as Hh

ROOT = Path(__file__).resolve().parents[1]
RCH = ROOT / "data" / "raw" / "rch"
OUT = ROOT / "data" / "cache" / "cnn"
OUT.mkdir(parents=True, exist_ok=True)
DEMD = ROOT / "data" / "dem" / "rchtrain"
CELL, BLOCK = 2000, 20000
rng = np.random.default_rng(0)
NEG_PER_CELL = 600


def main():
    man = gpd.read_file(RCH / "rch_manual.geojson")
    man["zone"] = man.geometry.x.map(dem.utm_epsg)
    w = E.windows()
    rec = w[w.kind == "rch_rec"]
    cells = []
    for epsg, g in man.groupby("zone"):
        gg = g.to_crs(epsg)
        x, y = gg.geometry.x.to_numpy(), gg.geometry.y.to_numpy()
        cx, cy = np.floor(x / CELL).astype(int), np.floor(y / CELL).astype(int)
        for a, b in set(zip(cx, cy, strict=True)):
            cells.append(dict(epsg=epsg, cx=a, cy=b, bx=a * CELL // BLOCK, by=b * CELL // BLOCK))
    C = pd.DataFrame(cells)
    # spatial holdout: (a) cells within 3 km of a phase-1 recall window centre -> test;
    # (b) a seeded random 20% of 20 km blocks -> test; (c) train cells within 2.5 km of any test
    # cell are dropped ("gap") so no hearth field straddles the split.
    C["x"] = (C.cx + 0.5) * CELL
    C["y"] = (C.cy + 0.5) * CELL
    test = np.zeros(len(C), bool)
    for r in rec.itertuples():
        epsg = dem.utm_epsg(r.lon)
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(r.lon, r.lat)
        test |= (C.epsg == epsg).to_numpy() & (np.hypot(C.x - x, C.y - y) < 3000).to_numpy()
    blocks = sorted(set(zip(C.epsg, C.bx, C.by, strict=True)))
    pick = {blocks[i] for i in rng.choice(len(blocks), size=int(0.2 * len(blocks)), replace=False)}
    test |= np.array([(e, a, b) in pick for e, a, b in zip(C.epsg, C.bx, C.by, strict=True)])
    split = np.where(test, "test", "train").astype(object)
    for i in np.where(~test)[0]:
        same = (C.epsg == C.epsg.iloc[i]).to_numpy() & test
        if same.any() and np.hypot(C.x[same] - C.x.iloc[i], C.y[same] - C.y.iloc[i]).min() < 2500:
            split[i] = "gap"
    C["split"] = split
    C = C[C.split != "gap"].reset_index(drop=True)
    C["name"] = [f"c{e}_{a}_{b}" for e, a, b in zip(C.epsg, C.cx, C.cy, strict=True)]
    C.to_csv(OUT / "cells.csv", index=False)
    print(len(C), "cells;", C.split.value_counts().to_dict(), flush=True)

    def fetch(r):
        x0, y0 = r.cx * CELL - 30, r.cy * CELL - 30
        return dem.fetch_bbox(r.name, x0, y0, x0 + CELL + 60, y0 + CELL + 60, r.epsg, out_dir=DEMD)

    DEMD.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(fetch, C.itertuples()), 1):
            if i % 25 == 0:
                print("dem", i, len(C), flush=True)

    X, Y, S, G, F = [], [], [], [], []
    for i, r in enumerate(C.itertuples(), 1):
        f = DEMD / f"{r.name}.tif"
        with rasterio.open(f) as src:
            z = src.read(1)
            T = ~src.transform
            gm = man.to_crs(src.crs)
        if np.isnan(z).mean() > 0.2:
            continue
        cand = Hh.candidates_with_features(z)
        if cand.empty:
            continue
        ref = np.array([T * (x, y) for x, y in zip(gm.geometry.x, gm.geometry.y, strict=True)])[:, ::-1]
        ref = ref[(ref[:, 0] > -50) & (ref[:, 0] < z.shape[0] + 50) & (ref[:, 1] > -50) & (ref[:, 1] < z.shape[1] + 50)]
        rc = cand[["row", "col"]].to_numpy()
        inner = (rc[:, 0] >= 30) & (rc[:, 0] < z.shape[0] - 30) & (rc[:, 1] >= 30) & (rc[:, 1] < z.shape[1] - 30)
        d, _ = cKDTree(ref).query(rc) if len(ref) else (np.full(len(rc), 1e9), None)
        y = np.full(len(rc), -1)
        y[d > 20] = 0
        if len(ref):
            dd, ii = cKDTree(rc).query(ref)
            y[ii[dd <= 6]] = 1
        y[~inner] = -1
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        neg = rng.choice(neg, size=min(NEG_PER_CELL, len(neg)), replace=False)
        sel = np.r_[pos, neg]
        ch = Hh.patch_channels(z)
        X.append(Hh.extract_patches(ch, rc[sel, 0], rc[sel, 1]))
        Y.append(y[sel])
        S.append(np.full(len(sel), r.split))
        G.append(np.full(len(sel), r.name))
        F.append(cand.iloc[sel].drop(columns=["row", "col"]).to_numpy("float32"))
        if i % 25 == 0:
            print("patches", i, len(C), flush=True)
    np.savez_compressed(
        OUT / "manual_patches.npz",
        X=np.concatenate(X),
        y=np.concatenate(Y),
        split=np.concatenate(S),
        cell=np.concatenate(G),
        F=np.concatenate(F),
    )
    print("saved", sum(len(y) for y in Y), "patches; positives", sum((y == 1).sum() for y in Y), flush=True)


if __name__ == "__main__":
    sys.exit(main())
