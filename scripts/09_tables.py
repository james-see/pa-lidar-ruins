"""Print the README tables from results/ (fixed threshold grid, not tuned)."""

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
P = pd.read_csv(R / "mounds_known.csv")
N = pd.read_csv(R / "mounds_negative_scores.csv")
nb, na = N[N.kind == "bg"].win.nunique() * 1.44, N[N.kind == "aml"].win.nunique() * 1.44
rows = []
for t in (0.5, 1.0, 2.0, 3.0, 5.0):
    rows.append(
        dict(
            threshold=t,
            recall_clear_domes=(P[P.audit == "clear_dome"].score >= t).mean(),
            recall_all=(P.score >= t).mean(),
            recall_clear_dev=(P[(P.audit == "clear_dome") & (P.split == "dev")].score >= t).mean(),
            recall_clear_test=(P[(P.audit == "clear_dome") & (P.split == "test")].score >= t).mean(),
            bg_per_km2=(N[N.kind == "bg"].score >= t).sum() / nb,
            aml_per_km2=(N[N.kind == "aml"].score >= t).sum() / na,
        )
    )
T = pd.DataFrame(rows)
T.to_csv(R / "mounds_table.csv", index=False)
print(T.round(2).to_markdown(index=False))
E = pd.read_csv(R / "earthworks_known_multi.csv")
out = []
for (cls, lab), g in E.assign(vis=np.where(E.audit == "clear", "clear", "faint/none")).groupby(
    ["cls", "vis"]
):
    out.append(
        dict(
            cls=cls,
            visibility=lab,
            n=len(g),
            median_pct_bg_tiles_above=g.pct_bg_above.median(),
            median_pct_mine_tiles_above=g.pct_aml_above.median(),
            top5pct_vs_bg=int((g.pct_bg_above <= 0.05).sum()),
            top5pct_vs_mines=int((g.pct_aml_above <= 0.05).sum()),
        )
    )
for cls, g in E.groupby("cls"):
    for sp, gg in g.groupby("split"):
        out.append(
            dict(
                cls=cls,
                visibility=f"all ({sp})",
                n=len(gg),
                median_pct_bg_tiles_above=gg.pct_bg_above.median(),
                median_pct_mine_tiles_above=gg.pct_aml_above.median(),
                top5pct_vs_bg=int((gg.pct_bg_above <= 0.05).sum()),
                top5pct_vs_mines=int((gg.pct_aml_above <= 0.05).sum()),
            )
        )
O = pd.DataFrame(out)
O.to_csv(R / "earthworks_table.csv", index=False)
print(O.round(3).to_markdown(index=False))
H = pd.read_csv(R / "hearths_sgl43_test_curve.csv")
print(H[H.threshold.isin([0.5, 0.7, 0.8, 0.9, 0.95])].round(3).to_markdown(index=False))
W = pd.read_csv(R / "hearths_recall_windows.csv")
print(
    W.groupby("threshold")
    .apply(
        lambda g: pd.Series(
            dict(
                recall=np.average(g.recall, weights=g.n_ref),
                n=g.n_ref.sum(),
                per_win_min=g.recall.min(),
                per_win_max=g.recall.max(),
            )
        ),
        include_groups=False,
    )
    .round(3)
    .to_markdown()
)
