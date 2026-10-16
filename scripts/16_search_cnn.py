"""Hearth search pass B: apply the hearth CNN (MPS) to every pass-A candidate.

Reads data/private/search/hearths/*.parquet + the cached search DEMs; writes p_cnn into
data/private/search/hearths_cnn/ (git-ignored). Re-run safe: finished tiles are skipped.
usage: uv run python scripts/16_search_cnn.py [--wait]   (--wait keeps polling until pass A is done)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from palidar import cnn
from palidar import hearths as Hh

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "private" / "search" / "hearths"
OUT = ROOT / "data" / "private" / "search" / "hearths_cnn"
DEMD = ROOT / "data" / "dem" / "search"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = cnn.load(ROOT / "models" / "hearth_cnn.pt")
    n_tiles = len(pd.read_csv(ROOT / "data" / "cache" / "aoi" / "tiles.csv").query("aoi == 'hearths'"))
    while True:
        todo = [f for f in sorted(IN.glob("*.parquet")) if not (OUT / f.name).exists()]
        for f in todo:
            t0 = time.time()
            c = pd.read_parquet(f)
            if len(c):
                with rasterio.open(DEMD / f"{c.tile.iloc[0]}.tif") as src:
                    z = src.read(1)
                X = Hh.extract_patches(Hh.patch_channels(z), c.row.to_numpy().astype(int), c.col.to_numpy().astype(int))
                c["p_cnn"] = cnn.predict(model, X)
            else:
                c["p_cnn"] = np.array([], float)
            c.to_parquet(OUT / f.name)
            n9 = int((c.p_cnn >= 0.9).sum())
            print(f"{f.stem}: {len(c)} cand, {n9} p_cnn>=0.9, {time.time() - t0:.0f}s", flush=True)
        done = len(list(OUT.glob("*.parquet")))
        if "--wait" not in sys.argv or done >= n_tiles:
            break
        time.sleep(30)
    print("done", done, "tiles", flush=True)


if __name__ == "__main__":
    main()
