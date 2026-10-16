"""Hearth search pass B: apply the hearth CNN (MPS) to every pass-A candidate.

Reads data/private/search/hearths/*.parquet + the cached search DEMs; writes p_cnn into
data/private/search/hearths_cnn/ (git-ignored). Re-run safe: finished tiles are skipped.
usage: uv run python scripts/16_search_cnn.py [--cls=hearths_all] [--wait] [--delete-dem]
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
CLS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cls=")), "hearths")
IN = ROOT / "data" / "private" / "search" / CLS
OUT = ROOT / "data" / "private" / "search" / f"{CLS}_cnn"
DEMD = ROOT / "data" / "dem" / "search"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = cnn.load(ROOT / "models" / "hearth_cnn.pt")
    TT = pd.read_csv(ROOT / "data" / "cache" / "aoi" / "tiles.csv")
    n_tiles = len(TT.query("aoi == @CLS"))
    keep_dem = set(TT[TT.aoi != CLS].tile)  # never delete DEMs used by the phase-2 searches
    n_units = TT[TT.aoi == CLS].groupby("tile").size().to_dict()  # tiles shared by county units
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
            tl = f.stem.split("__")[0]
            all_done = len(list(OUT.glob(f"{tl}__*.parquet"))) >= n_units.get(tl, 1)
            if "--delete-dem" in sys.argv and tl not in keep_dem and all_done:
                # statewide run: DEMs are re-downloadable; keep disk use bounded (~68 MB per tile)
                (DEMD / f"{tl}.tif").unlink(missing_ok=True)
            n9 = int((c.p_cnn >= 0.9).sum())
            nd = len(list(OUT.glob("*.parquet")))
            msg = f"{len(c)} cand, {n9} p_cnn>=0.9, {time.time() - t0:.0f}s"
            print(f"{time.strftime('%H:%M')} {nd}/{n_tiles} {f.stem}: {msg}", flush=True)
        done = len(list(OUT.glob("*.parquet")))
        a_done = (IN / "_PASS_A_DONE").exists() and not [f for f in IN.glob("*.parquet") if not (OUT / f.name).exists()]
        if "--wait" not in sys.argv or done >= n_tiles or a_done:
            break
        time.sleep(30)
    print("done", done, "tiles", flush=True)


if __name__ == "__main__":
    main()
