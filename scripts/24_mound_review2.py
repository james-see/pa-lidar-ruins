"""Round-2 blind review for the re-tuned mound ranking.

Top 20 v2 mound candidates (<= 5 per county, 15 m dedupe) mixed with 8 hidden known mounds NOT
used as controls in round 1 and 10 negative windows NOT used in round 1, shuffled under new ids.
Private outputs: results/private/review2/ (sheets, key.csv); candidates -> results/private/candidates_mounds_v2.csv
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from palidar import evalutil as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
OUT = ROOT / "results" / "private" / "review2"
DEM = ROOT / "data" / "dem"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rc = load("rc", "19_review_chips.py")
cand = load("cand", "18_candidates.py")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(S / "mounds_v2.parquet").sort_values("v2", ascending=False).reset_index(drop=True)
    d = cand.dedupe(d, r=15)
    top = d.head(3000).copy()
    g = cand.add_county(cand.to_wgs(top))
    g["lon"], g["lat"] = g.geometry.x, g.geometry.y
    g = pd.DataFrame(g.drop(columns="geometry")).sort_values("v2", ascending=False).reset_index(drop=True)
    g.to_csv(ROOT / "results" / "private" / "candidates_mounds_v2.csv", index=False)
    pick = g.groupby("county").head(5).head(20)
    summ = __import__("json").loads((ROOT / "results" / "mounds_v2_summary.json").read_text())
    thr = summ["thresholds"]["lr@1.0"]  # dev operating point: 1 bg detection per km2
    full = cand.add_county(cand.to_wgs(d[d.v2 >= thr]))
    pd.DataFrame(full.drop(columns="geometry")).groupby("county").size().rename("n").reset_index().assign(
        cls="mound_candidates_v2 (lr, dev 1/km2 op)"
    )[["cls", "county", "n"]].to_csv(ROOT / "results" / "search_counts_by_county_mounds_v2.csv", index=False)
    print("v2 candidates at 1/km2 op:", len(full), "threshold", round(thr, 4))
    r1 = pd.read_csv(ROOT / "results" / "private" / "review" / "key.csv")
    used = set(r1.ref)
    ks = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv")
    ctl = ks[(ks.cls == "mound") & ks.visual_audit.isin(["clear_dome", "faint"]) & ~ks.site_id.isin(used)]
    ctl = ctl.sample(min(8, len(ctl)), random_state=5)
    w = E.windows()
    negs = w[w.kind.isin(["bg", "aml"]) & ~w.win.isin(used)]
    negs = pd.concat(
        [negs[negs.kind == "bg"].sample(6, random_state=5), negs[negs.kind == "aml"].sample(4, random_state=5)]
    )
    it = [
        dict(src="candidate", ref=f"v{i}", path=DEM / "search" / f"{r.tile}.tif", x=r.x, y=r.y)
        for i, r in pick.iterrows()
    ]
    it += [dict(src="control_known", ref=r.site_id, path=DEM / f"site_{r.site_id}.tif") for r in ctl.itertuples()]
    it += [dict(src="control_negative", ref=r.win, path=DEM / f"{r.win}.tif") for r in negs.itertuples()]
    I = pd.DataFrame(it).sample(frac=1, random_state=17).reset_index(drop=True)
    I["id"] = [f"M2-{i + 1:02d}" for i in range(len(I))]
    I["group"] = "mounds"
    I.drop(columns="path").to_csv(OUT / "key.csv", index=False)
    size, r, clip = rc.GROUPS["mounds"]
    for s in range(0, len(I), 6):
        sub = I.iloc[s : s + 6]
        fig, axs = plt.subplots(len(sub), 2, figsize=(7, 3.4 * len(sub)))
        axs = np.atleast_2d(axs)
        for ax, rr in zip(axs, sub.itertuples(), strict=True):
            lr, hs = rc.chip(rr.path, size, r, None if pd.isna(rr.x) else rr.x, None if pd.isna(rr.y) else rr.y)
            ax[0].imshow(lr, cmap="gray", vmin=-clip, vmax=clip)
            ax[1].imshow(hs, cmap="gray")
            ax[0].set_title(f"{rr.id}  LRM r={r} m (+-{clip} m)", fontsize=9)
            ax[1].set_title(f"{rr.id}  multi-hillshade, {size} m wide", fontsize=9)
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"mounds2_sheet_{s // 6 + 1:02d}.png", dpi=80)
        plt.close(fig)
    print(I.groupby("src").size())
    print("v2 candidates after dedupe:", len(d))


if __name__ == "__main__":
    main()
