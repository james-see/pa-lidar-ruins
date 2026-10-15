"""Calibrate detector A (mounds) on 47 known public mounds vs PA background + strip-mine windows.

Match rule: a known mound counts as found if a candidate lies within 30 m of the published point.
Negatives: every candidate in random PA background windows (bg) and DEP abandoned-mine windows (aml).
(bg candidates may include real unrecorded features, so 'FP/km2' is an upper bound.)
Thresholds are chosen on the dev split (odd ids) and reported on the test split (even ids).
Only scores (no coordinates) are written for bg/aml windows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from palidar import dem
from palidar import detectors as D
from palidar import evalutil as E

ROOT = Path(__file__).resolve().parents[1]

R_MATCH = 30
w = E.windows()
pos_rows, neg_rows = [], []
for r in w.itertuples():
    if not ((r.kind == "site" and r.cls == "mound") or r.kind in ("bg", "aml")):
        continue
    z, _, _ = dem.read(ROOT / "data" / "dem" / f"{r.win}.tif")
    cache = ROOT / "data" / "cache" / "mounds" / f"{r.win}.csv"
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        d = D.detect_mounds(z)
        cache.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(cache, index=False)
    c = z.shape[0] / 2
    if r.kind == "site":
        d["dist"] = np.hypot(d.row - c, d.col - c)
        near = d[d.dist <= R_MATCH]
        best = near.score.max() if len(near) else np.nan
        rank = int((d.score > best).sum() + 1) if len(near) else None
        b = near.loc[near.score.idxmax()] if len(near) else None
        pos_rows.append(
            dict(
                win=r.win,
                split=r.split,
                score=best,
                rank_in_window=rank,
                n_window=len(d),
                diam=None if b is None else round(b.diam, 1),
                height=None if b is None else round(b.height, 2),
            )
        )
    else:
        area = (z.shape[0] * z.shape[1]) / 1e6
        for s in d.score:
            neg_rows.append(dict(win=r.win, kind=r.kind, split=r.split, score=s))
        if not len(d):
            neg_rows.append(dict(win=r.win, kind=r.kind, split=r.split, score=np.nan))
P = pd.DataFrame(pos_rows)
N = pd.DataFrame(neg_rows)
area = {
    k: w[(w.kind == k[0]) & (w.split == k[1])].shape[0] * 1.44
    for k in [("bg", "dev"), ("bg", "test"), ("aml", "dev"), ("aml", "test")]
}
P.to_csv(ROOT / "results" / "mounds_known.csv", index=False)
N.to_csv(ROOT / "results" / "mounds_negative_scores.csv", index=False)

res = {}
ths = np.round(np.arange(0.5, 15.01, 0.25), 2)
for split in ("dev", "test"):
    p = P[P.split == split].score.to_numpy()
    nb = N[(N.split == split) & (N.kind == "bg")].score.dropna().to_numpy()
    na = N[(N.split == split) & (N.kind == "aml")].score.dropna().to_numpy()
    cb = E.recall_fp_curve(p, nb, area[("bg", split)], ths)
    ca = E.recall_fp_curve(p, na, area[("aml", split)], ths)
    cb["fp_aml_per_km2"] = ca.fp_per_km2
    res[split] = cb
# operating points from dev: highest threshold reaching >= target dev recall
ops = []
for target in (0.9, 0.8, 0.7, 0.5):
    dv = res["dev"]
    ok = dv[dv.recall >= target]
    t = ok.threshold.max() if len(ok) else ths.min()
    te = res["test"].set_index("threshold").loc[t]
    dd = dv.set_index("threshold").loc[t]
    ops.append(
        dict(
            target_dev_recall=target,
            threshold=float(t),
            dev_recall=dd.recall,
            test_recall=te.recall,
            test_bg_fp_per_km2=te.fp_per_km2,
            test_aml_fp_per_km2=te.fp_aml_per_km2,
            dev_bg_fp_per_km2=dd.fp_per_km2,
        )
    )
O = pd.DataFrame(ops)
O.to_csv(ROOT / "results" / "mounds_operating_points.csv", index=False)
pd.concat([res["dev"].assign(split="dev"), res["test"].assign(split="test")]).to_csv(
    ROOT / "results" / "mounds_curve.csv", index=False
)
summary = dict(
    n_known=len(P),
    n_detected_any=int(P.score.notna().sum()),
    recall_any_candidate=float(P.score.notna().mean()),
    median_rank_in_window=float(P.rank_in_window.median()),
    top1_in_window=int((P.rank_in_window == 1).sum()),
    top5_in_window=int((P.rank_in_window <= 5).sum()),
    median_window_candidates=float(P.n_window.median()),
)
(ROOT / "results" / "mounds_summary.json").write_text(json.dumps(summary, indent=2))
aud = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv").set_index("site_id").visual_audit
P["audit"] = P.win.str.replace("site_", "").map(aud)
by = []
for o in ops:
    t = o["threshold"]
    for a, g in P.groupby("audit"):
        by.append(dict(threshold=t, audit=a, n=len(g), recall=float((g.score >= t).mean())))
    g = P
    by.append(dict(threshold=t, audit="ALL", n=len(g), recall=float((g.score >= t).mean())))
B = pd.DataFrame(by).drop_duplicates()
B.to_csv(ROOT / "results" / "mounds_recall_by_visibility.csv", index=False)
P.to_csv(ROOT / "results" / "mounds_known.csv", index=False)
summary["recall_any_candidate_clear_domes"] = float(P[P.audit == "clear_dome"].score.notna().mean())
(ROOT / "results" / "mounds_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(B.to_string())
print(O.round(3).to_string())
print(P.sort_values("score").to_string())
