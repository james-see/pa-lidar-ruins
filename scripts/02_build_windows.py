"""Build the list of calibration DEM windows and download them (1 m, 3DEP ImageServer).

Window kinds
  site      1.2 km square around every known mound / enclosure / military earthwork
  rch_sgl43 the whole of State Game Land 43 (Berks/Chester Co.), where Carter et al. (2021)
            visually assessed every Mask R-CNN prediction and logged false negatives
            -> closed-world reference for hearth recall AND precision
  rch_rec   2 km squares on the densest clusters of manually mapped hearths in other SGLs
            (recall only; labels there are not exhaustive)
  bg        random 1.2 km squares inside Pennsylvania (background rate / rank percentile)
  aml       1.2 km squares on DEP abandoned-mine-land polygons (strip mines, spoil piles):
            the expected main false-positive source
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from palidar import dem

ROOT = Path(__file__).resolve().parents[1]

CAL = ROOT / "data" / "calibration"
REF = ROOT / "data" / "raw" / "ref"
RCH = ROOT / "data" / "raw" / "rch"
rng = np.random.default_rng(42)
HALF = 600


def windows():
    rows = []
    sites = pd.read_csv(CAL / "known_sites.csv")
    for r in sites.itertuples():
        rows.append(dict(win=f"site_{r.site_id}", kind="site", cls=r.cls, lat=r.lat, lon=r.lon, half=HALF))

    pa = gpd.read_file(REF / "physio_provinces.geojson").to_crs(4326).union_all()
    minx, miny, maxx, maxy = pa.bounds
    n = 0
    while n < 60:
        lon, lat = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if pa.buffer(-0.01).contains(Point(lon, lat)):
            n += 1
            rows.append(dict(win=f"bg_{n:02d}", kind="bg", cls="background", lat=lat, lon=lon, half=HALF))

    aml = gpd.read_file(REF / "aml_polys.geojson")
    aml = aml[aml.geometry.notna() & aml.is_valid]
    aml = aml.to_crs(26917)
    aml = aml[(aml.area > 20000) & (aml.area < 400000)]
    pick = aml.sample(40, random_state=7)
    pts = pick.representative_point().to_crs(4326)
    for i, (p, t) in enumerate(zip(pts, pick.sf_probl_1), 1):
        rows.append(dict(win=f"aml_{i:02d}", kind="aml", cls=t, lat=p.y, lon=p.x, half=HALF))

    man = gpd.read_file(RCH / "rch_manual.geojson").to_crs(26918)
    sgl = gpd.read_file(REF / "state_game_lands.geojson").to_crs(26918)
    s43 = sgl[sgl.SGL == 43].union_all()
    man = man[man.distance(s43) > 5000]
    xy = np.c_[man.geometry.x, man.geometry.y]
    chosen = []
    # greedy: densest 2 km cells, at least 10 km apart
    cell = np.floor(xy / 2000).astype(int)
    cnt = pd.Series(list(map(tuple, cell))).value_counts()
    for (cx, cy), _n in cnt.items():
        ctr = np.array([cx * 2000 + 1000, cy * 2000 + 1000])
        if all(np.hypot(*(ctr - q)) > 10000 for q in chosen):
            chosen.append(ctr)
        if len(chosen) == 12:
            break
    from pyproj import Transformer

    t = Transformer.from_crs(26918, 4326, always_xy=True)
    for i, c in enumerate(chosen, 1):
        lon, lat = t.transform(*c)
        rows.append(dict(win=f"rch_rec_{i:02d}", kind="rch_rec", cls="hearth", lat=lat, lon=lon, half=1000))
    return pd.DataFrame(rows), s43


def main():
    w, _ = windows()
    w.to_csv(CAL / "windows.csv", index=False)
    print(w.kind.value_counts())
    jobs = []
    with ThreadPoolExecutor(6) as ex:
        # full extent of the SGL 43 lidar image assessed by Carter et al. (all predictions + FNs)
        jobs.append(ex.submit(dem.fetch_bbox, "rch_sgl43_full", 427700, 4443300, 440200, 4452860, 26918))
        for r in w.itertuples():
            jobs.append(ex.submit(dem.fetch_window, r.win, r.lon, r.lat, r.half))
        done = 0
        for f in as_completed(jobs):
            try:
                p = f.result()
                done += 1
                print("ok", done, len(jobs), p.name, flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAIL", e, flush=True)


if __name__ == "__main__":
    main()
