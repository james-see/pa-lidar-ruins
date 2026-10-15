"""Calibrate detector C (linear earthworks: berm + ditch) as a tile-ranking problem, like Nate's
pueblo test: every 1.2 km window is cut into 100 m tiles and scored. For each known site we take
the best tile within 150 m of the published point and ask what fraction of background tiles
(random PA + strip-mine windows) score higher (= its rank percentile; lower is better).

Run on military earthworks (class C) and on prehistoric enclosures (class A, linear part).
Outputs: no coordinates for background tiles.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from palidar import dem
from palidar import detectors as D
from palidar import evalutil as E

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "small": dict(sigmas=(1.5, 3.0), lrm_radius=12),  # parapet+ditch 5-15 m wide
    "multi": dict(sigmas=(1.5, 3.0, 6.0), lrm_radius=25),  # adds broad Hopewell-scale walls
}
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "small"
CACHE = ROOT / "data" / "cache" / f"works_{CONFIG}"
CACHE.mkdir(parents=True, exist_ok=True)
TILE = 100
NEAR = {"military": 150, "enclosure": 250}  # enclosure points are centroids of 200-400 m circles


def tiles(win):
    f = CACHE / f"{win}.csv"
    if f.exists():
        return pd.read_csv(f)
    z, _, _ = dem.read(ROOT / "data" / "dem" / f"{win}.tif")
    t = D.tile_scores(z, tile=TILE, **CONFIGS[CONFIG])
    t.to_csv(f, index=False)
    return t


def main():
    w = E.windows()
    aud = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv").set_index("site_id")
    neg = []
    for r in w[w.kind.isin(["bg", "aml"])].itertuples():
        t = tiles(r.win)
        neg.append(t.assign(kind=r.kind, win=r.win))
    N = pd.concat(neg)
    rows = []
    for r in w[(w.kind == "site") & w.cls.isin(["military", "enclosure"])].itertuples():
        t = tiles(r.win)
        n = int(np.sqrt(len(t)))
        ctr = n * TILE / 2
        cy = (t.tile_r + 0.5) * TILE
        cx = (t.tile_c + 0.5) * TILE
        near = t[np.hypot(cy - ctr, cx - ctr) <= NEAR[r.cls]]
        sid = r.win.replace("site_", "")
        best = near.score.max()
        rows.append(
            dict(
                site=sid,
                cls=r.cls,
                audit=aud.loc[sid, "visual_audit"],
                split=r.split,
                score=best,
                pct_bg_above=float((N[N.kind == "bg"].score > best).mean()),
                pct_aml_above=float((N[N.kind == "aml"].score > best).mean()),
                pct_all_above=float((N.score > best).mean()),
            )
        )
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "results" / f"earthworks_known_{CONFIG}.csv", index=False)
    N[["kind", "density", "odiv", "score"]].to_csv(
        ROOT / "results" / f"earthworks_negative_tiles_{CONFIG}.csv", index=False
    )
    out = {}
    for split in ("dev", "test"):
        for cls in ("military", "enclosure"):
            g = P[(P.cls == cls) & (P.split == split)]
            out[f"{cls}_{split}_median_pct_all_above"] = float(g.pct_all_above.median())
    for cls, sub in [("military", P[P.cls == "military"]), ("enclosure", P[P.cls == "enclosure"])]:
        for lab, g in [("all", sub), ("clear", sub[sub.audit == "clear"])]:
            out[f"{cls}_{lab}"] = dict(
                n=len(g),
                median_pct_all_above=float(g.pct_all_above.median()),
                in_top1pct=int((g.pct_all_above <= 0.01).sum()),
                in_top5pct=int((g.pct_all_above <= 0.05).sum()),
                in_top10pct=int((g.pct_all_above <= 0.10).sum()),
            )
    out["n_negative_tiles"] = dict(bg=int((N.kind == "bg").sum()), aml=int((N.kind == "aml").sum()))
    (ROOT / "results" / f"earthworks_summary_{CONFIG}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(P.sort_values("pct_all_above").round(4).to_string())


if __name__ == "__main__":
    main()
