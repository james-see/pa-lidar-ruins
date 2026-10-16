"""Train the hearth CNN and compare it with the phase-1 GBM on exactly the phase-1 tests.

Train: manual-label cells in the train split + SGL 43 west half (+ strip-mine dev windows as hard
negatives). Test (never seen in training): SGL 43 east half (closed-world precision + recall),
the 12 phase-1 recall windows (inside held-out cells), and density on random-PA / strip-mine windows.
"""

import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.spatial import cKDTree
from torch import nn

from palidar import cnn
from palidar import evalutil as E
from palidar import hearths as Hh

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RCH = ROOT / "data" / "raw" / "rch"
torch.manual_seed(0)
rng = np.random.default_rng(0)
R_POS = 6.0


def to_rc(g, src):
    T = ~src.transform
    g = g.to_crs(src.crs)
    return np.array([T * (x, y) for x, y in zip(g.geometry.x, g.geometry.y, strict=True)])[:, ::-1]


def sgl43():
    src = rasterio.open(ROOT / "data" / "dem" / "rch_sgl43_full.tif")
    z = src.read(1)
    H, W = z.shape
    m = gpd.read_file(RCH / "rch_master_final.geojson")
    m = m[m.SGL.astype(str) == "43"]
    fn = gpd.read_file(RCH / "rch_falseneg_sgl43.geojson")
    ref = np.vstack([to_rc(m[m.Confirm == "Y"], src), to_rc(fn, src)])
    negref = to_rc(m[m.Confirm == "N"], src)
    ref = ref[(ref[:, 0] > 20) & (ref[:, 0] < H - 20) & (ref[:, 1] > 20) & (ref[:, 1] < W - 20)]
    c = pd.read_parquet(CACHE / "hearths" / "rch_sgl43_full.parquet")
    c = c[(c.row > 20) & (c.row < H - 20) & (c.col > 20) & (c.col < W - 20)].reset_index(drop=True)
    rc = c[["row", "col"]].to_numpy()
    y = np.full(len(c), -1)
    d, _ = cKDTree(ref).query(rc)
    y[d > 15] = 0
    dd, ii = cKDTree(rc).query(ref)
    y[ii[dd <= R_POS]] = 1
    dn, inn = cKDTree(rc).query(negref)
    y[inn[dn <= R_POS]] = 0
    ch = Hh.patch_channels(z)
    return c, y, ref, ch, (H, W)


def train(Xtr, ytr, Xva, yva, epochs=14):
    dev = cnn.device()
    model = cnn.HearthNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * 200)
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(200):  # balanced batches of 256
            idx = np.r_[rng.choice(pos, 96), rng.choice(neg, 160)]
            xb = torch.from_numpy(Xtr[idx].astype(np.float32)).to(dev)
            yb = torch.from_numpy(ytr[idx].astype(np.float32)).to(dev)
            xb = cnn.augment(xb)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        p = cnn.predict(model, Xva, tta=False)
        from sklearn.metrics import average_precision_score

        print(f"epoch {ep} loss {tot / 200:.3f} val AP {average_precision_score(yva, p):.3f}", flush=True)
    return model.cpu()


def curve(p, rc, ref, area):
    rows = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        det = rc[p >= t]
        if not len(det):
            continue
        rec = (cKDTree(det).query(ref)[0] <= R_POS).mean()
        prec = (cKDTree(ref).query(det)[0] <= R_POS).mean()
        rows.append(dict(threshold=t, recall=rec, precision=prec, n_det=len(det), det_per_km2=len(det) / area))
    return pd.DataFrame(rows)


def main():
    d = np.load(CACHE / "cnn" / "manual_patches.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    c43, y43, ref43, ch43, (H, W) = sgl43()
    mid = W / 2
    rc43 = c43[["row", "col"]].to_numpy()
    west = (rc43[:, 1] < mid) & (y43 >= 0)
    wi = np.where(west)[0]
    wi = np.r_[wi[y43[wi] == 1], rng.choice(wi[y43[wi] == 0], 30000, replace=False)]
    X43w = Hh.extract_patches(ch43, rc43[wi, 0], rc43[wi, 1])
    # strip-mine dev windows: hard negatives
    w = E.windows()
    Xa = []
    for r in w[(w.kind == "aml") & (w.split == "dev")].itertuples():
        cc = pd.read_parquet(CACHE / "hearths" / f"{r.win}.parquet")
        z = rasterio.open(ROOT / "data" / "dem" / f"{r.win}.tif").read(1)
        k = rng.choice(len(cc), min(800, len(cc)), replace=False)
        Xa.append(Hh.extract_patches(Hh.patch_channels(z), cc.row.to_numpy()[k], cc.col.to_numpy()[k]))
    Xa = np.concatenate(Xa)
    tr = split == "train"
    Xtr = np.concatenate([X[tr], X43w, Xa])
    ytr = np.concatenate([y[tr], y43[wi], np.zeros(len(Xa), int)])
    va = split == "test"
    print("train", len(ytr), "pos", int((ytr == 1).sum()), "| manual-test patches", int(va.sum()), flush=True)
    model = train(Xtr, ytr, X[va], y[va])
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "hearth_cnn.pt")

    # ---- SGL 43 east (closed world)
    east = rc43[:, 1] >= mid
    pe = cnn.predict(model, Hh.extract_patches(ch43, rc43[east, 0], rc43[east, 1]))
    ref_e = ref43[ref43[:, 1] >= mid]
    area = H * (W - mid) / 1e6
    clf, feats = joblib.load(CACHE / "hearth_model.joblib")
    pg = clf.predict_proba(c43.loc[east, feats])[:, 1]
    out = {}
    for name, p in [("gbm_phase1", pg), ("cnn", pe), ("mean_gbm_cnn", (pe + pg) / 2)]:
        cv = curve(p, rc43[east], ref_e, area)
        cv.to_csv(ROOT / "results" / f"hearths_sgl43_test_curve_{name}.csv", index=False)
        out[name] = cv
    # ---- recall windows and density windows
    man = gpd.read_file(RCH / "rch_manual.geojson")
    rows = []
    for r in w[w.kind.isin(["rch_rec", "bg", "aml"])].itertuples():
        cc = pd.read_parquet(CACHE / "hearths" / f"{r.win}.parquet")
        src = rasterio.open(ROOT / "data" / "dem" / f"{r.win}.tif")
        z = src.read(1)
        p_c = cnn.predict(model, Hh.extract_patches(Hh.patch_channels(z), cc.row.to_numpy(), cc.col.to_numpy()))
        p_g = clf.predict_proba(cc[feats])[:, 1]
        km2 = z.size / 1e6
        rc = cc[["row", "col"]].to_numpy()
        ref = None
        if r.kind == "rch_rec":
            b = src.bounds
            g = man.to_crs(src.crs).cx[b.left + 30 : b.right - 30, b.bottom + 30 : b.top - 30]
            ref = to_rc(g, src)
        for name, p in [("gbm_phase1", p_g), ("cnn", p_c), ("mean_gbm_cnn", (p_c + p_g) / 2)]:
            for t in (0.5, 0.7, 0.9, 0.95):
                det = rc[p >= t]
                rec = np.nan
                if ref is not None:
                    rec = (cKDTree(det).query(ref)[0] <= R_POS).mean() if len(det) else 0.0
                rows.append(
                    dict(
                        win=r.win,
                        kind=r.kind,
                        split=r.split,
                        model=name,
                        threshold=t,
                        n_ref=0 if ref is None else len(ref),
                        recall=rec,
                        det_per_km2=len(det) / km2,
                    )
                )
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / "hearths_windows_compare.csv", index=False)
    rec = (
        R[R.kind == "rch_rec"]
        .groupby(["model", "threshold"])
        .apply(lambda g: np.average(g.recall, weights=g.n_ref), include_groups=False)
        .unstack()
    )
    # density only on TEST-split bg/aml windows (dev aml windows were used as CNN negatives)
    den = (
        R[(R.kind != "rch_rec") & (R.split == "test")]
        .groupby(["model", "kind", "threshold"])
        .det_per_km2.median()
        .unstack()
    )
    summ = {
        k: v.set_index("threshold")[["recall", "precision", "det_per_km2"]].round(3).to_dict("index")
        for k, v in out.items()
    }
    (ROOT / "results" / "hearths_cnn_compare.json").write_text(
        json.dumps(
            dict(
                sgl43_east=summ,
                recall_windows=rec.round(3).to_dict(),
                density_test_windows={f"{m}|{k}": v for (m, k), v in den.round(2).T.to_dict().items()},
            ),
            indent=1,
            default=str,
        )
    )
    for k, v in out.items():
        print(k)
        print(v[v.threshold.isin([0.5, 0.7, 0.8, 0.9, 0.95])].round(3).to_string(index=False))
    print(rec.round(3))
    print(den.round(2))


if __name__ == "__main__":
    main()
