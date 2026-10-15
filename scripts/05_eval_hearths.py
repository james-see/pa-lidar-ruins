"""Calibrate detector B (relict charcoal hearths).

Reference (closed world): the full lidar image around State Game Land 43 (Berks/Chester Co.,
next to Hopewell Furnace NHS) from Carter, Blackadar & Conner (2021): every Mask R-CNN
prediction there was visually assessed (Confirm = Y/N) and missed hearths were logged
(False_Neg_SGL43). Reference hearths = Confirm Y + false negatives.

Split: west half of the image -> train the ranking model; east half -> test (recall AND
precision). Independent recall check: 12 x 4 km2 windows in other game lands with the
authors' *manually* mapped hearths (not exhaustive -> recall only). Background / strip-mine
windows give detection density where few or no hearths are expected.

Matching: a reference hearth is found if a detection lies within 6 m (platforms are 8-15 m wide).
Outputs contain no coordinates for bg/aml windows.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

from palidar import evalutil as E
from palidar import hearths as Hh

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "hearths"
CACHE.mkdir(parents=True, exist_ok=True)
RCH = ROOT / "data" / "raw" / "rch"
R_POS, R_NEG = 6.0, 15.0


def load_cands(name):
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    with rasterio.open(ROOT / "data" / "dem" / f"{name}.tif") as s:
        z = s.read(1)
    c = Hh.candidates_with_features(z)
    c.to_parquet(f)
    return c


def to_rc(gdf, src):
    T = ~src.transform
    g = gdf.to_crs(src.crs)
    return np.array([T * (x, y) for x, y in zip(g.geometry.x, g.geometry.y, strict=True)])[:, ::-1]


def label(c, ref_rc, neg_rc=None):
    y = np.full(len(c), -1)
    tree = cKDTree(c[["row", "col"]].to_numpy())
    d_ref, _ = cKDTree(ref_rc).query(c[["row", "col"]].to_numpy())
    y[d_ref > R_NEG] = 0
    # positives: nearest candidate to each reference within R_POS
    dd, ii = tree.query(ref_rc)
    y[ii[dd <= R_POS]] = 1
    if neg_rc is not None and len(neg_rc):
        dn, inn = tree.query(neg_rc)
        y[inn[dn <= R_POS]] = 0
    return y


FEATS = None


def main():
    global FEATS
    src = rasterio.open(ROOT / "data" / "dem" / "rch_sgl43_full.tif")
    H, W = src.shape
    m = gpd.read_file(RCH / "rch_master_final.geojson")
    m = m[m.SGL.astype(str) == "43"]
    fn = gpd.read_file(RCH / "rch_falseneg_sgl43.geojson")
    ref = np.vstack([to_rc(m[m.Confirm == "Y"], src), to_rc(fn, src)])
    negref = to_rc(m[m.Confirm == "N"], src)
    inb = (ref[:, 0] > 20) & (ref[:, 0] < H - 20) & (ref[:, 1] > 20) & (ref[:, 1] < W - 20)
    ref = ref[inb]
    c = load_cands("rch_sgl43_full")
    c = c[(c.row > 20) & (c.row < H - 20) & (c.col > 20) & (c.col < W - 20)].reset_index(drop=True)
    FEATS = [k for k in c.columns if k not in ("row", "col")]
    y = label(c, ref, negref)
    mid = W / 2
    tr = (c.col < mid).to_numpy()
    te = ~tr
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced", random_state=0
    )
    k = tr & (y >= 0)
    clf.fit(c.loc[k, FEATS], y[k])
    c["p"] = clf.predict_proba(c[FEATS])[:, 1]

    # ---- closed-world test on the east half
    ref_te = ref[ref[:, 1] >= mid]
    area_te = H * (W - mid) / 1e6
    rows = []
    ct = c[te]
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        det = ct[ct.p >= t][["row", "col"]].to_numpy()
        if len(det) == 0:
            rows.append(dict(threshold=t, recall=0, precision=np.nan, n_det=0))
            continue
        d_ref, _ = cKDTree(det).query(ref_te)
        rec = (d_ref <= R_POS).mean()
        d_det, _ = cKDTree(ref_te).query(det)
        prec = (d_det <= R_POS).mean()
        rows.append(
            dict(threshold=t, recall=rec, precision=prec, n_det=len(det), det_per_km2=len(det) / area_te)
        )
    cw = pd.DataFrame(rows)
    cw.to_csv(ROOT / "results" / "hearths_sgl43_test_curve.csv", index=False)

    # ---- independent recall windows + density windows
    man = gpd.read_file(RCH / "rch_manual.geojson")
    w = E.windows()
    rec_rows, dens_rows = [], []
    for r in w[w.kind.isin(["rch_rec", "bg", "aml"])].itertuples():
        cc = load_cands(r.win)
        if cc.empty:
            continue
        cc["p"] = clf.predict_proba(cc[FEATS])[:, 1]
        s2 = rasterio.open(ROOT / "data" / "dem" / f"{r.win}.tif")
        if r.kind == "rch_rec":
            g = man.to_crs(s2.crs).cx[
                s2.bounds.left + 30 : s2.bounds.right - 30, s2.bounds.bottom + 30 : s2.bounds.top - 30
            ]
            rc = to_rc(g, s2)
            for t in (0.3, 0.5, 0.7, 0.9):
                det = cc[cc.p >= t][["row", "col"]].to_numpy()
                d, _ = cKDTree(det).query(rc) if len(det) else (np.full(len(rc), 1e9), None)
                rec_rows.append(
                    dict(
                        win=r.win,
                        threshold=t,
                        n_ref=len(rc),
                        recall=(d <= R_POS).mean(),
                        det_per_km2=len(det) / (s2.width * s2.height / 1e6),
                    )
                )
        else:
            for t in (0.3, 0.5, 0.7, 0.9):
                dens_rows.append(
                    dict(
                        win=r.win,
                        kind=r.kind,
                        threshold=t,
                        det_per_km2=(cc.p >= t).sum() / (s2.width * s2.height / 1e6),
                    )
                )
    RR = pd.DataFrame(rec_rows)
    DD = pd.DataFrame(dens_rows)
    RR.to_csv(ROOT / "results" / "hearths_recall_windows.csv", index=False)
    DD.to_csv(ROOT / "results" / "hearths_density_windows.csv", index=False)
    agg_rec = RR.groupby("threshold").apply(
        lambda g: pd.Series(
            dict(
                recall=np.average(g.recall, weights=g.n_ref),
                n_ref=g.n_ref.sum(),
                det_per_km2=g.det_per_km2.mean(),
            )
        ),
        include_groups=False,
    )
    agg_den = DD.groupby(["kind", "threshold"]).det_per_km2.agg(["mean", "median"])
    summary = dict(
        sgl43_reference_hearths=int(len(ref)),
        sgl43_test_reference=int(len(ref_te)),
        sgl43_candidates=int(len(c)),
        candidate_recall_test=float(
            (cKDTree(ct[["row", "col"]].to_numpy()).query(ref_te)[0] <= R_POS).mean()
        ),
        train_pos=int(((y == 1) & tr).sum()),
        train_neg=int(((y == 0) & tr).sum()),
    )
    (ROOT / "results" / "hearths_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(cw.round(3).to_string())
    print(agg_rec.round(3).to_string())
    print(agg_den.round(1).to_string())
    # keep the model's test-half detections for a private visual audit (not committed)
    priv = ROOT / "results" / "private"
    priv.mkdir(parents=True, exist_ok=True)
    ct2 = c[te].copy()
    d_det, _ = cKDTree(ref).query(ct2[["row", "col"]].to_numpy())
    ct2["matched"] = d_det <= R_POS
    ct2[["row", "col", "p", "matched"]].to_csv(priv / "sgl43_test_dets.csv", index=False)
    import joblib

    joblib.dump((clf, FEATS), ROOT / "data" / "cache" / "hearth_model.joblib")


if __name__ == "__main__":
    main()
