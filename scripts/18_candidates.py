"""Aggregate the search outputs into filtered candidate lists with stage counts.

Private (coordinates): data/private/search/cand_<class>.parquet, results/private/candidates_<class>.csv
Public (no coordinates): results/search_coverage.csv (searchable km2 per AOI unit),
results/search_stage_counts.csv (counts per class and filtering stage),
results/search_counts_by_county.csv (final-stage counts per class and county).
"""

import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
PRIV = ROOT / "results" / "private"
RES = ROOT / "results"
H_THR = 0.9  # CNN threshold (SGL43 east: recall 0.959, precision 0.577 at 0.9)
M_THR = 2.0  # mound score (phase 1: clear-dome recall 80% at >= 2)


def load(cls):
    fs = sorted((S / cls).glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return d, fs


def coverage(cls):
    rows = []
    for f in (S / cls).glob("*.cov"):
        tile, unit = f.stem.split("__", 1)
        rows.append(dict(cls=cls, tile=tile, unit_slug=unit, km2=float(f.read_text())))
    return pd.DataFrame(rows)


def to_wgs(d):
    if not len(d):
        return gpd.GeoDataFrame(d.assign(geometry=None), geometry="geometry", crs=4326)
    parts = []
    for epsg, g in d.groupby("epsg"):
        gg = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.x, g.y), crs=int(epsg)).to_crs(4326)
        parts.append(gg)
    g = pd.concat(parts)
    return gpd.GeoDataFrame(g, geometry="geometry", crs=4326)


def dedupe(d, r=8.0):
    """Drop duplicate detections from overlapping AOI units (same class) keeping the highest key."""
    keep = np.ones(len(d), bool)
    for _, g in d.groupby("epsg"):
        xy = g[["x", "y"]].to_numpy()
        pairs = cKDTree(xy).query_pairs(r, output_type="ndarray")
        idx = g.index.to_numpy()
        keep[idx[pairs[:, 1]]] = False
    return d[keep].reset_index(drop=True)


def add_county(g):
    c = gpd.read_file(ROOT / "data" / "raw" / "ref" / "counties.geojson")[["COUNTY_NAM", "geometry"]].to_crs(4326)
    j = gpd.sjoin(g, c, how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]
    j["county"] = j.COUNTY_NAM.fillna("outside_PA").str.title()
    return j.drop(columns="COUNTY_NAM")


def known_hearths_utm(epsg):
    rch = ROOT / "data" / "raw" / "rch"
    a = gpd.read_file(rch / "rch_master_final.geojson").to_crs(epsg)
    b = gpd.read_file(rch / "rch_manual.geojson").to_crs(epsg)
    return np.c_[np.r_[a.geometry.x, b.geometry.x], np.r_[a.geometry.y, b.geometry.y]]


def gnis_furnaces():
    g = pd.read_csv(ROOT / "data" / "raw" / "gnis" / "Text" / "DomesticNames_PA.txt", sep="|", encoding="utf-8-sig")
    g = g[g.feature_name.str.contains("Furnace", case=False, na=False)]
    return gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.prim_long_dec, g.prim_lat_dec), crs=4326)


def hearths(stages):
    d, fs = load("hearths_cnn")
    stages.append(dict(cls="hearths", stage="tiles_processed", n=len(fs)))
    stages.append(dict(cls="hearths", stage="geometric_candidates_in_searchable_area", n=len(d)))
    stages.append(dict(cls="hearths", stage="phase1_gbm_p>=0.9 (for comparison)", n=int((d.p_gbm >= 0.9).sum())))
    stages.append(dict(cls="hearths", stage="cnn_p>=0.5", n=int((d.p_cnn >= 0.5).sum())))
    d = d[d.p_cnn >= H_THR].reset_index(drop=True)
    stages.append(dict(cls="hearths", stage=f"cnn_p>={H_THR}", n=len(d)))
    d = dedupe(d.sort_values("p_cnn", ascending=False).reset_index(drop=True))
    stages.append(dict(cls="hearths", stage="deduplicated_8m", n=len(d)))
    d["known_carter"] = False
    for epsg, g in d.groupby("epsg"):
        dist, _ = cKDTree(known_hearths_utm(int(epsg))).query(g[["x", "y"]].to_numpy())
        d.loc[g.index, "known_carter"] = dist <= 30
    stages.append(dict(cls="hearths", stage="matching_Carter_et_al_points_30m", n=int(d.known_carter.sum())))
    g = add_county(to_wgs(d))
    # hearth fields: DBSCAN 250 m / 5 hearths, per UTM zone
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
            p_mean=("p_cnn", "mean"),
        )
        .reset_index()
    )
    F["frac_known"] = F.n_known / F.n
    fg = to_wgs(F)
    fur = gnis_furnaces()
    fu = fur.to_crs(5070)
    fd, fi = cKDTree(np.c_[fu.geometry.x, fu.geometry.y]).query(
        np.c_[fg.to_crs(5070).geometry.x, fg.to_crs(5070).geometry.y]
    )
    F["nearest_gnis_furnace_km"] = fd / 1000
    F["nearest_gnis_furnace"] = fur.feature_name.to_numpy()[fi]
    F["lon"], F["lat"] = fg.geometry.x.to_numpy(), fg.geometry.y.to_numpy()
    stages.append(dict(cls="hearths", stage="hearth_fields_dbscan250m_min5", n=len(F)))
    stages.append(dict(cls="hearths", stage="fields_mostly_new(<20%_known)", n=int((F.frac_known < 0.2).sum())))
    stages.append(
        dict(
            cls="hearths",
            stage="new_fields_>5km_from_GNIS_furnace",
            n=int(((F.frac_known < 0.2) & (F.nearest_gnis_furnace_km > 5)).sum()),
        )
    )
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    g = pd.DataFrame(g.drop(columns="geometry"))
    g.to_parquet(S / "cand_hearths.parquet")
    F.to_csv(PRIV / "hearth_fields.csv", index=False)
    return g, F


def mounds(stages):
    d, fs = load("mounds")
    stages.append(dict(cls="mounds", stage="tiles_processed", n=len(fs)))
    stages.append(dict(cls="mounds", stage="domes_score>=1_in_searchable_area", n=len(d)))
    d = d[d.score >= M_THR]
    stages.append(dict(cls="mounds", stage=f"score>={M_THR}", n=len(d)))
    d = d[(d.height.between(0.5, 8)) & (d.diam.between(8, 50))]
    stages.append(dict(cls="mounds", stage="mound-sized (h 0.5-8 m, d 8-50 m)", n=len(d)))
    d = dedupe(d.sort_values("score", ascending=False).reset_index(drop=True), r=15)
    stages.append(dict(cls="mounds", stage="deduplicated_15m", n=len(d)))
    g = add_county(to_wgs(d))
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    g = pd.DataFrame(g.drop(columns="geometry")).sort_values("score", ascending=False)
    g.to_parquet(S / "cand_mounds.parquet")
    return g


def works(stages):
    d, fs = load("works")
    stages.append(dict(cls="works", stage="tiles_processed", n=len(fs)))
    stages.append(dict(cls="works", stage="100m_tiles_scored (>=50% searchable)", n=len(d)))
    summ = json.loads((RES / "earthworks_v2_summary.json").read_text())
    thr = summ["bg_quantiles"]["v1"]["0.995"]
    m, feats = joblib.load(ROOT / "data" / "cache" / "works_v2_model.joblib")
    d["l_density"] = np.log(d.density + 1e-3)
    d["l_amp95"] = np.log(d.amp95 + 1e-3)
    d["l_rough"] = np.log(d["rough"] + 1e-3)
    d["l_score"] = np.log(d.score + 1e-3)
    d["p_lr"] = m.predict_proba(d[feats].to_numpy())[:, 1] if len(d) else []
    d = d[d.score >= thr].reset_index(drop=True)
    d["tile_r"] = d.tile_r.astype(int)
    d["tile_c"] = d.tile_c.astype(int)
    stages.append(dict(cls="works", stage="score>=bg_99.5th_pct", n=len(d)))
    # group 8-connected hot tiles within each raster tile
    d["grp"] = -1
    off = 0
    for _, g in d.groupby(["tile", "unit"]):
        grid = np.zeros((g.tile_r.max() + 2, g.tile_c.max() + 2), int)
        grid[g.tile_r, g.tile_c] = 1
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        d.loc[g.index, "grp"] = lab[g.tile_r, g.tile_c] - 1 + off
        off += n
    G = (
        d.groupby("grp")
        .agg(
            n_tiles=("score", "size"),
            score_max=("score", "max"),
            p_lr_max=("p_lr", "max"),
            parallel=("parallel", "mean"),
            x=("x", "mean"),
            y=("y", "mean"),
            epsg=("epsg", "first"),
            unit=("unit", "first"),
            tile=("tile", "first"),
        )
        .reset_index()
    )
    stages.append(dict(cls="works", stage="hot_tile_groups", n=len(G)))
    G = dedupe(G.sort_values("score_max", ascending=False).reset_index(drop=True), r=150)
    stages.append(dict(cls="works", stage="groups_deduplicated", n=len(G)))
    g = add_county(to_wgs(G))
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    g = pd.DataFrame(g.drop(columns="geometry")).sort_values("score_max", ascending=False)
    g.to_parquet(S / "cand_works.parquet")
    return g


def main():
    PRIV.mkdir(parents=True, exist_ok=True)
    stages = []
    cov = pd.concat([coverage(c) for c in ("hearths", "mounds", "works")])
    cov.groupby(["cls", "unit_slug"]).km2.agg(["sum", "size"]).rename(columns={"sum": "km2", "size": "tiles"}).round(
        1
    ).to_csv(RES / "search_coverage.csv")
    for c, g in cov.groupby("cls"):
        stages.append(dict(cls=c, stage="searchable_km2", n=round(g.km2.sum(), 1)))
    h, F = hearths(stages)
    mo = mounds(stages)
    w = works(stages)
    st = pd.DataFrame(stages)
    st.to_csv(RES / "search_stage_counts.csv", index=False)
    by = pd.concat(
        [
            h.groupby("county").size().rename("n").reset_index().assign(cls="hearths_cnn>=0.9"),
            F[F.frac_known < 0.2].groupby("county").size().rename("n").reset_index().assign(cls="new_hearth_fields"),
            mo.groupby("county").size().rename("n").reset_index().assign(cls="mound_candidates"),
            w.groupby("county").size().rename("n").reset_index().assign(cls="earthwork_tile_groups"),
        ]
    )[["cls", "county", "n"]]
    by.to_csv(RES / "search_counts_by_county.csv", index=False)
    mo.to_csv(PRIV / "candidates_mounds.csv", index=False)
    w.to_csv(PRIV / "candidates_works.csv", index=False)
    print(st.to_string(index=False))
    print(by.to_string(index=False))


if __name__ == "__main__":
    main()
