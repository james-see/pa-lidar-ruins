"""Statewide hearth wrap-up: phase-2 hearth areas + the rest of PA public land.

Reads CNN outputs (data/private/search/hearths_cnn + hearths_all_cnn), keeps p_cnn >= 0.9,
dedupes, matches Carter et al. (all 52,884 SGL predictions + 4,361 manual points) within 30 m,
groups hearth fields (DBSCAN 250 m, >= 5) and measures the distance to GNIS "Furnace" names.
Public (no coordinates): results/statewide_hearths_stage_counts.csv,
results/statewide_hearths_by_county.csv, results/statewide_hearths_carter_agreement.json.
Private: data/private/search/cand_hearths_statewide.parquet, results/private/hearth_fields_statewide.csv
"""

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
P = ROOT / "results" / "private"
spec = importlib.util.spec_from_file_location("c18", ROOT / "scripts" / "18_candidates.py")
c18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c18)
COLS = ["x", "y", "epsg", "tile", "unit", "p_cnn", "p_gbm"]
THR = 0.9


def load(dirname):
    n_cand, n_tiles, parts = 0, 0, []
    for f in sorted((S / dirname).glob("*.parquet")):
        d = pd.read_parquet(f, columns=None)
        n_tiles += 1
        n_cand += len(d)
        if len(d):
            parts.append(d[d.p_cnn >= 0.5][COLS])
    return pd.concat(parts, ignore_index=True), n_cand, n_tiles


def coverage():
    rows = []
    for dname in ("hearths", "hearths_all"):
        for f in (S / dname).glob("*.cov"):
            tile, unit = f.stem.split("__", 1)
            rows.append(dict(run=dname, tile=tile, unit=unit, km2=float(f.read_text())))
    return pd.DataFrame(rows)


def main():
    st = []
    A, na, ta = load("hearths_cnn")
    B, nb, tb = load("hearths_all_cnn")
    cov = coverage()
    n_expected = len(
        pd.read_csv(ROOT / "data" / "cache" / "aoi" / "tiles.csv").query("aoi in ['hearths','hearths_all']")
    )
    st += [
        dict(stage="tile_jobs_expected", n=n_expected),
        dict(stage="tile_jobs_cnn_scored", n=ta + tb),
        dict(stage="searchable_km2", n=round(cov.km2.sum(), 1)),
        dict(stage="geometric_candidates", n=na + nb),
    ]
    d = pd.concat([A, B], ignore_index=True)
    st.append(dict(stage="cnn_p>=0.5", n=len(d)))
    d = d[d.p_cnn >= THR].sort_values("p_cnn", ascending=False).reset_index(drop=True)
    st.append(dict(stage=f"cnn_p>={THR}", n=len(d)))
    d = c18.dedupe(d)
    st.append(dict(stage="deduplicated_8m", n=len(d)))
    d["known_carter"] = False
    rch = ROOT / "data" / "raw" / "rch"
    master = gpd.read_file(rch / "rch_master_final.geojson")
    manual = gpd.read_file(rch / "rch_manual.geojson")
    agree = {}
    for epsg, g in d.groupby("epsg"):
        kn = c18.known_hearths_utm(int(epsg))
        dist, _ = cKDTree(kn).query(g[["x", "y"]].to_numpy())
        d.loc[g.index, "known_carter"] = dist <= 30
    st.append(dict(stage="matching_Carter_et_al_30m", n=int(d.known_carter.sum())))
    # agreement the other way: Carter points that fall inside a searched tile
    tiles = set(cov[cov.km2 > 0].tile)
    for name, ref in (
        ("master_all", master),
        ("master_confirmed_Y", master[master.Confirm == "Y"]),
        ("manual", manual),
    ):
        tot = hit = 0
        for epsg, g in d.groupby("epsg"):
            r = ref.to_crs(int(epsg))
            xy = np.c_[r.geometry.x, r.geometry.y]
            tid = [f"{epsg}_{int(x // 4000 * 4000)}_{int(y // 4000 * 4000)}" for x, y in xy]
            inside = np.array([t in tiles for t in tid])
            if not inside.any():
                continue
            dist, _ = cKDTree(g[["x", "y"]].to_numpy()).query(xy[inside])
            tot += int(inside.sum())
            hit += int((dist <= 30).sum())
        agree[name] = dict(
            points_in_searched_tiles=tot, within_30m_of_cnn_detection=hit, fraction=round(hit / tot, 3) if tot else None
        )
    g = c18.add_county(c18.to_wgs(d))
    g["field"] = -1
    off = 0
    for _, gg in g.groupby("epsg"):
        lab = DBSCAN(eps=250, min_samples=5).fit_predict(gg[["x", "y"]].to_numpy())
        g.loc[gg.index, "field"] = np.where(lab >= 0, lab + off, -1)
        off += lab.max() + 1
    F = (
        g[g.field >= 0]
        .groupby("field")
        .agg(
            n=("p_cnn", "size"),
            n_known=("known_carter", "sum"),
            x=("x", "mean"),
            y=("y", "mean"),
            epsg=("epsg", "first"),
            county=("county", lambda s: s.mode().iloc[0]),
            unit=("unit", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    F["frac_known"] = F.n_known / F.n
    fg = c18.to_wgs(F)
    fur = c18.gnis_furnaces().to_crs(5070)
    fd, fi = cKDTree(np.c_[fur.geometry.x, fur.geometry.y]).query(
        np.c_[fg.to_crs(5070).geometry.x, fg.to_crs(5070).geometry.y]
    )
    F["nearest_gnis_furnace_km"] = fd / 1000
    F["nearest_gnis_furnace"] = fur.feature_name.to_numpy()[fi]
    F["lon"], F["lat"] = fg.geometry.x.to_numpy(), fg.geometry.y.to_numpy()
    new = F.frac_known < 0.2
    far = F.nearest_gnis_furnace_km > 5
    st += [
        dict(stage="hearth_fields_dbscan250m_min5", n=len(F)),
        dict(stage="hearths_in_fields", n=int(F.n.sum())),
        dict(stage="fields_mostly_new(<20%_Carter)", n=int(new.sum())),
        dict(stage="new_fields_>5km_from_GNIS_furnace", n=int((new & far).sum())),
        dict(stage="new_fields_>5km_from_GNIS_furnace_and_>=50_hearths", n=int((new & far & (F.n >= 50)).sum())),
    ]
    fur_wgs = c18.gnis_furnaces()
    fur_tiles_hit = 0
    fx = fur.copy()
    for epsg, gd in d.groupby("epsg"):
        r = fur_wgs.to_crs(int(epsg))
        dist, _ = cKDTree(gd[["x", "y"]].to_numpy()).query(np.c_[r.geometry.x, r.geometry.y])
        fx[f"d{epsg}"] = dist
    dmin = fx[[c for c in fx.columns if c.startswith("d2")]].min(axis=1)
    fur_tiles_hit = int((dmin <= 2000).sum())
    agree["gnis_furnace_names"] = dict(n=len(fur_wgs), with_cnn_hearth_within_2km=fur_tiles_hit)
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    pd.DataFrame(g.drop(columns="geometry")).to_parquet(S / "cand_hearths_statewide.parquet")
    F.to_csv(P / "hearth_fields_statewide.csv", index=False)
    S_ = pd.DataFrame(st)
    S_.to_csv(ROOT / "results" / "statewide_hearths_stage_counts.csv", index=False)
    by = (
        g.groupby("county")
        .size()
        .rename("hearths_cnn_p>=0.9")
        .to_frame()
        .join(F.groupby("county").size().rename("fields"))
        .join(F[new].groupby("county").size().rename("fields_mostly_new"))
        .join(F[new & far].groupby("county").size().rename("new_fields_far_from_GNIS_furnace"))
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    by.to_csv(ROOT / "results" / "statewide_hearths_by_county.csv", index=False)
    (ROOT / "results" / "statewide_hearths_carter_agreement.json").write_text(json.dumps(agree, indent=2))
    print(S_.to_string(index=False))
    print(json.dumps(agree, indent=2))
    print(by.sort_values("hearths_cnn_p>=0.9", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()
