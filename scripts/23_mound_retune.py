"""Mound detector re-tune: re-rank phase-1 dome detections toward 1-3 m domes on flat ground,
using terrain-context features and known false positives as negatives, with spatial holdouts.

Data (all detections with phase-1 score >= 1, the same universe the search used):
  * calibration windows (phase 1): known mounds = best detection within 30 m of the published
    point; negatives = every detection in random-PA (bg) and strip-mine (aml) windows.
    Windows keep their phase-1 dev/test split.
  * search detections (natural knobs, spoil, rail/road bumps; 19/20 of the phase-2 top-20 were
    rejected in blind review): negatives, split by 20 km blocks into train/test halves (seeded).
Scorers: v1 (phase-1 dome score), prior (hand-built size/setting prior x log score, no fitting),
lr (logistic regression on context features; fitted on dev windows + train blocks only).
The scorer for the search is picked on DEV; TEST numbers are reported for every scorer.
Operating points: thresholds that give the phase-1 dev bg densities (16 and 3 per km2) on dev.
Search re-ranking uses a 2-fold block cross-fit so no detection is scored by a model that saw it.
Outputs (public, no coordinates): results/mounds_v2_*.csv/json. Private: data/private/search/mounds_v2.parquet
"""

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from palidar import dem
from palidar import evalutil as E
from palidar import moundctx as MC

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "data" / "cache" / "mounds_ctx"
S = ROOT / "data" / "private" / "search"
SC = S / "mounds_ctx"
R_MATCH = 30
WIN_KM2 = 1.44
TARGET_BG = (3.0, 1.0, 0.3)  # 16/km2 (phase-1 low point) exceeds the >=1 universe
rng = np.random.default_rng(11)


def win_ctx(win):
    f = CC / f"{win}.parquet"
    if not f.exists():
        z, _, _ = dem.read(ROOT / "data" / "dem" / f"{win}.tif")
        d = pd.read_csv(ROOT / "data" / "cache" / "mounds" / f"{win}.csv")
        d = d[d.score >= 1.0]
        MC.context_features(z, d).to_parquet(f)
    return win, pd.read_parquet(f)


def tile_ctx(path):
    f = SC / path.name
    if not f.exists():
        d = pd.read_parquet(path)
        if len(d):
            with rasterio.open(ROOT / "data" / "dem" / "search" / f"{d.tile.iloc[0]}.tif") as src:
                z = src.read(1)
            d = MC.context_features(z, d)
        else:
            d = MC.context_features(np.zeros((10, 10), np.float32), d)
        d.to_parquet(f)
    return pd.read_parquet(f)


def scorers(fit_df):
    X = fit_df[MC.CTX_FEATS].fillna(fit_df[MC.CTX_FEATS].median()).to_numpy()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
    m.fit(X, fit_df.lab.to_numpy())
    med = fit_df[MC.CTX_FEATS].median()

    def lr(d):
        return m.predict_proba(d[MC.CTX_FEATS].fillna(med).to_numpy())[:, 1]

    return {
        "v1": lambda d: d.score.to_numpy(),
        "prior": lambda d: (MC.size_setting_prior(d) * np.log1p(d.score)).to_numpy(),
        "lr": lr,
    }, m


def thr_for_density(neg_scores, km2, target):
    s = np.sort(neg_scores)[::-1]
    k = int(np.floor(target * km2))
    return s[k] if k < len(s) else -np.inf


def main():
    CC.mkdir(parents=True, exist_ok=True)
    SC.mkdir(parents=True, exist_ok=True)
    w = E.windows()
    w = w[((w.kind == "site") & (w.cls == "mound")) | w.kind.isin(["bg", "aml"])]
    aud = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv").set_index("site_id").visual_audit
    with Pool(8) as p:
        WC = dict(p.map(win_ctx, w.win.tolist()))
        tiles = sorted((S / "mounds").glob("*.parquet"))
        SD = pd.concat(p.map(tile_ctx, tiles), ignore_index=True)
    # ---- calibration table
    rows = []
    for r in w.itertuples():
        d = WC[r.win]
        if r.kind == "site":
            with rasterio.open(ROOT / "data" / "dem" / f"{r.win}.tif") as src:
                c = src.height / 2
            dist = np.hypot(d.row - c, d.col - c) if len(d) else np.array([])
            near = d[dist <= R_MATCH] if len(d) else d
            if len(near):
                b = near.loc[[near.score.idxmax()]].assign(lab=1, win=r.win, kind="site", split=r.split)
                b["audit"] = aud.get(r.win.replace("site_", ""))
                b["others"] = len(d) - 1
                rows.append(b)
            else:
                rows.append(
                    pd.DataFrame(
                        [
                            dict(
                                lab=1,
                                win=r.win,
                                kind="site",
                                split=r.split,
                                missed=True,
                                audit=aud.get(r.win.replace("site_", "")),
                            )
                        ]
                    )
                )
        else:
            rows.append(d.assign(lab=0, win=r.win, kind=r.kind, split=r.split))
    C = pd.concat(rows, ignore_index=True)
    # ---- search negatives with 20 km block split
    SD["block"] = (
        SD.epsg.astype(str)
        + "_"
        + (SD.x // 20000).astype(int).astype(str)
        + "_"
        + (SD.y // 20000).astype(int).astype(str)
    )
    blocks = np.array(sorted(SD.block.unique()))
    tb = set(rng.choice(blocks, len(blocks) // 2, replace=False))
    SD["split"] = np.where(SD.block.isin(tb), "test", "dev")
    SD["lab"] = 0
    SD["kind"] = "search"
    cov = {f.stem: float(f.read_text()) for f in (S / "mounds").glob("*.cov")}
    tile_split = SD.drop_duplicates("tile").set_index("tile").split.to_dict()
    km2_search = {s: sum(v for k, v in cov.items() if tile_split.get(k.split("__")[0]) == s) for s in ("dev", "test")}
    fit = pd.concat([C[(C.split == "dev") & ~C.get("missed", False).fillna(False).astype(bool)], SD[SD.split == "dev"]])
    fns, model = scorers(fit)
    coefs = dict(zip(MC.CTX_FEATS, model[-1].coef_[0].round(3).tolist(), strict=True))
    det = C[~C.get("missed", False).fillna(False).astype(bool)]
    km2 = {(k, s): WIN_KM2 * len(w[(w.kind == k) & (w.split == s)]) for k in ("bg", "aml") for s in ("dev", "test")}
    out = dict(
        coefs_lr=coefs,
        km2=dict(search_dev=km2_search["dev"], search_test=km2_search["test"]),
        n_pos_dev=int(((det.lab == 1) & (det.split == "dev")).sum()),
        n_pos_test=int(((det.lab == 1) & (det.split == "test")).sum()),
        n_missed_any=int(C.get("missed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        n_neg_search_dev=int((SD.split == "dev").sum()),
        n_neg_search_test=int((SD.split == "test").sum()),
    )
    table = []
    for name, f in fns.items():
        det_s = det.assign(s=f(det))
        sd_s = SD.assign(s=f(SD))
        for tgt in TARGET_BG:
            dev_bg = det_s[(det_s.kind == "bg") & (det_s.split == "dev")].s.to_numpy()
            t = thr_for_density(dev_bg, km2[("bg", "dev")], tgt)
            out.setdefault("thresholds", {})[f"{name}@{tgt}"] = float(t)
            for split in ("dev", "test"):
                pos = det_s[(det_s.lab == 1) & (det_s.split == split)]
                n_sites = int(((C.lab == 1) & (C.split == split)).sum())
                clear = pos[pos.audit == "clear_dome"]
                n_clear = int(((C.lab == 1) & (C.split == split) & (C.audit == "clear_dome")).sum())
                table.append(
                    dict(
                        scorer=name,
                        target_dev_bg_per_km2=tgt,
                        split=split,
                        recall_all=round(float((pos.s >= t).sum() / n_sites), 3),
                        recall_clear_dome=round(float((clear.s >= t).sum() / max(n_clear, 1)), 3),
                        n_sites=n_sites,
                        n_clear=n_clear,
                        bg_per_km2=round(
                            float(
                                (det_s[(det_s.kind == "bg") & (det_s.split == split)].s >= t).sum() / km2[("bg", split)]
                            ),
                            2,
                        ),
                        aml_per_km2=round(
                            float(
                                (det_s[(det_s.kind == "aml") & (det_s.split == split)].s >= t).sum()
                                / km2[("aml", split)]
                            ),
                            2,
                        ),
                        search_per_km2=round(float((sd_s[sd_s.split == split].s >= t).sum() / km2_search[split]), 2),
                    )
                )
    T = pd.DataFrame(table)
    T.to_csv(ROOT / "results" / "mounds_v2_table.csv", index=False)
    dev3 = T[(T.split == "dev") & (T.target_dev_bg_per_km2 == 3.0)].sort_values(
        ["recall_all", "search_per_km2"], ascending=[False, True]
    )
    out["picked_on_dev"] = dev3.scorer.iloc[0]
    # ---- re-rank the search with a 2-fold block cross-fit for the picked scorer
    pick = out["picked_on_dev"]
    sc = np.zeros(len(SD))
    for fold in ("dev", "test"):
        fit_f = pd.concat(
            [C[(C.split == "dev") & ~C.get("missed", False).fillna(False).astype(bool)], SD[SD.split != fold]]
        )
        f2, _ = scorers(fit_f)
        idx = SD.split == fold
        sc[idx.to_numpy()] = f2[pick](SD[idx])
    SD["v2"] = sc
    SD["prior"] = fns["prior"](SD)
    SD.to_parquet(S / "mounds_v2.parquet")
    (ROOT / "results" / "mounds_v2_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
