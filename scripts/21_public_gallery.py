"""Public figure gallery: lidar-only chips of review survivors and typical rejects, labelled by
class and county only. De-georeferenced on purpose: plain PNG (no world file / GeoTIFF tags),
metadata stripped, no coordinates in names or text, no roads/towns labels, no scale bar, and each
chip gets a random 90-degree rotation / flip so it cannot be matched by orientation.
Writes figures/gallery/*.png (committed) -- everything with coordinates stays in results/private/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import importlib.util  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
REV = ROOT / "results" / "private" / "review"
OUT = ROOT / "figures" / "gallery"
spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts" / "19_review_chips.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
rng = np.random.default_rng(2026)
LABEL = {"hearths": "Charcoal-hearth field", "mounds": "Mound candidate", "works": "Linear-earthwork candidate"}


def strip_save(fig, path):
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=90, metadata={"Software": None})
    plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    clean = Image.frombytes("RGB", im.size, im.tobytes())  # drops every ancillary chunk
    clean.save(path, format="PNG", optimize=True)
    tmp.unlink()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    k = pd.read_csv(REV / "key.csv").merge(pd.read_csv(REV / "scores.csv"), on="id")
    k = k[k.src == "candidate"]
    vd = pd.read_csv(ROOT / "results" / "private" / "crosscheck" / "verdicts.csv").set_index("id").verdict.to_dict()
    cand = {
        "mounds": pd.read_parquet(S / "cand_mounds.parquet"),
        "works": pd.read_parquet(S / "cand_works.parquet"),
        "hearths": pd.read_csv(ROOT / "results" / "private" / "hearth_fields.csv"),
    }
    H = pd.read_parquet(S / "cand_hearths.parquet")
    rows = []
    for grp in ("hearths", "mounds", "works"):
        g = k[k.group == grp].sort_values("score", ascending=False)
        keep = pd.concat(
            [g[g.score >= 2].head(6), g[g.score == 0].sample(min(3, (g.score == 0).sum()), random_state=3)]
        )
        for n, r in enumerate(keep.itertuples(), 1):
            idx = int(r.ref[1:])
            c = cand[grp]
            if grp == "hearths":
                c = c[c.field == idx].iloc[0]
                h = H[H.field == idx]
                hh = h.iloc[np.argmin(np.hypot(h.x - c.x, h.y - c.y))]
                tile, x, y = hh.tile, hh.x, hh.y
            else:
                c = c.loc[idx]
                tile, x, y = c.tile, c.x, c.y
            size, rad, clip = rc.GROUPS[grp]
            lr, hs = rc.chip(ROOT / "data" / "dem" / "search" / f"{tile}.tif", size, rad, x, y)
            kk, fl = int(rng.integers(4)), bool(rng.integers(2))
            lr, hs = np.rot90(lr, kk), np.rot90(hs, kk)
            if fl:
                lr, hs = lr[:, ::-1], hs[:, ::-1]
            verdict = "survived blind review" if r.score >= 2 else "rejected in blind review"
            if r.score >= 2 and r.id in vd:
                verdict += f"; cross-check: {vd[r.id]}"
            fig, ax = plt.subplots(1, 2, figsize=(8, 4.6))
            ax[0].imshow(lr, cmap="gray", vmin=-clip, vmax=clip)
            ax[1].imshow(hs, cmap="gray")
            for a in ax:
                a.set_axis_off()
            fig.suptitle(
                f"{LABEL[grp]} - {c.county} County - blind-review score {r.score}/3\n{verdict}\n"
                f"left: local relief, right: multi-direction hillshade; ~{size} m across, orientation randomised",
                fontsize=9,
            )
            fig.tight_layout()
            name = f"{grp}_{n:02d}_{c.county.replace(' ', '')}.png"
            strip_save(fig, OUT / name)
            rows.append(
                dict(file=f"figures/gallery/{name}", cls=grp, county=c.county, review_score=r.score, verdict=verdict)
            )
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
