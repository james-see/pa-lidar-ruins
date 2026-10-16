"""Detector C v2: strip-mine-vs-earthwork features + a small dev-trained logistic model.

Phase 1 ranked 100 m tiles by score = density * (0.3 + odiv) and found strip-mine spoil ridges
were the main false positives. v2 adds per-tile features (orientation parallelism, LRM amplitude
and roughness, ridge/valley balance) and fits a heavily regularised logistic regression on the
DEV split only (positives: 2 best phase-1 tiles near each dev military/enclosure site;
negatives: every tile of dev bg + aml windows). Evaluation is on the TEST split only: for each
test site, best tile within NEAR of the published point; rank percentile vs test-split negative
tiles. The phase-1 score is re-measured on exactly the same negatives for a fair before/after.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from palidar import dem
from palidar import detectors as D
from palidar import evalutil as E

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "works_v2"
CACHE.mkdir(parents=True, exist_ok=True)
CFG = dict(sigmas=(1.5, 3.0, 6.0), lrm_radius=25)
TILE = 100
NEAR = {"military": 150, "enclosure": 250}
FEATS = ["l_density", "odiv", "parallel", "l_amp95", "l_rough", "ridge", "valley", "l_score"]
# Variants are all fitted/defined on DEV; the one used in the search is picked by DEV median
# pct-above (bg+aml mean). TEST numbers for every variant are reported, not just the winner.
VARIANTS = {
    "lr_full": FEATS,
    "lr_small": ["l_score", "l_rough", "parallel"],
    "lr_shape": ["l_score", "parallel", "odiv", "l_amp95"],
}


def tiles(win):
    f = CACHE / f"{win}.csv"
    if not f.exists():
        z, _, _ = dem.read(ROOT / "data" / "dem" / f"{win}.tif")
        D.tile_scores(z, tile=TILE, **CFG).to_csv(f, index=False)
    return win, pd.read_csv(f)


def add_feats(t):
    t = t.copy()
    t["l_density"] = np.log(t.density + 1e-3)
    t["l_amp95"] = np.log(t.amp95 + 1e-3)
    t["l_rough"] = np.log(t["rough"] + 1e-3)
    t["l_score"] = np.log(t.score + 1e-3)
    return t


def main():
    w = E.windows()
    w = w[w.kind.isin(["bg", "aml"]) | ((w.kind == "site") & w.cls.isin(["military", "enclosure"]))]
    with Pool(6) as pool:
        T = dict(pool.map(tiles, w.win.tolist()))
    T = {k: add_feats(v) for k, v in T.items()}
    neg = pd.concat([T[r.win].assign(kind=r.kind, split=r.split, win=r.win) for r in w[w.kind != "site"].itertuples()])
    sites = []
    for r in w[w.kind == "site"].itertuples():
        t = T[r.win]
        n = int(np.sqrt(len(t)))
        ctr = n * TILE / 2
        d = np.hypot((t.tile_r + 0.5) * TILE - ctr, (t.tile_c + 0.5) * TILE - ctr)
        sites.append((r, t[d <= NEAR[r.cls]]))
    pos = pd.concat([near.nlargest(2, "score") for r, near in sites if r.split == "dev"])
    ndev = neg[neg.split == "dev"]
    y = np.r_[np.ones(len(pos)), np.zeros(len(ndev))]
    models, out = {}, dict(n_pos_dev=int(len(pos)), n_neg_dev=int(len(ndev)), coefs={})
    for name, fs in VARIANTS.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        m.fit(pd.concat([pos[fs], ndev[fs]]).to_numpy(), y)
        models[name] = (m, fs)
        out["coefs"][name] = dict(zip(fs, m[-1].coef_[0].round(3).tolist(), strict=True))
        neg[name] = m.predict_proba(neg[fs].to_numpy())[:, 1]
    neg["v1"] = neg.score
    neg["v1_nonpar"] = neg.score * (1 - neg.parallel)
    scorers = ["v1", "v1_nonpar", *VARIANTS]
    aud = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv").set_index("site_id")
    rows = []
    for r, near in sites:
        sid = r.win.replace("site_", "")
        near = near.copy()
        near["v1"] = near.score
        near["v1_nonpar"] = near.score * (1 - near.parallel)
        for name, (m, fs) in models.items():
            near[name] = m.predict_proba(near[fs].to_numpy())[:, 1]
        N = neg[neg.split == r.split]
        row = dict(site=sid, cls=r.cls, split=r.split, audit=aud.loc[sid, "visual_audit"])
        for sc in scorers:
            b = near[sc].max()
            for bk in ("bg", "aml"):
                row[f"{sc}_pct_{bk}"] = float((N[N.kind == bk][sc] > b).mean())
        rows.append(row)
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "results" / "earthworks_v2_known.csv", index=False)
    tab = []
    for split in ("dev", "test"):
        for cls in ("military", "enclosure", "both"):
            for lab in ("all", "clear"):
                g = P[P.split == split] if cls == "both" else P[(P.split == split) & (P.cls == cls)]
                if lab == "clear":
                    g = g[g.audit == "clear"]
                if not len(g):
                    continue
                for sc in scorers:
                    tab.append(
                        dict(
                            split=split,
                            cls=cls,
                            subset=lab,
                            n=len(g),
                            scorer=sc,
                            median_pct_bg_above=float(g[f"{sc}_pct_bg"].median()),
                            median_pct_aml_above=float(g[f"{sc}_pct_aml"].median()),
                        )
                    )
    tab = pd.DataFrame(tab)
    tab.to_csv(ROOT / "results" / "earthworks_v2_table.csv", index=False)
    d = tab[(tab.split == "dev") & (tab.cls == "both") & (tab.subset == "all")].copy()
    d["crit"] = (d.median_pct_bg_above + d.median_pct_aml_above) / 2
    best = d.sort_values("crit").scorer.iloc[0]
    out["picked_on_dev"] = best
    if best in models:
        joblib.dump(models[best], ROOT / "data" / "cache" / "works_v2_model.joblib")
    bg = neg[neg.kind == "bg"]
    out["bg_quantiles"] = {sc: {q: float(bg[sc].quantile(q)) for q in (0.95, 0.99, 0.995, 0.999)} for sc in scorers}
    print(tab.round(4).to_string())
    (ROOT / "results" / "earthworks_v2_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
