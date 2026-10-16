"""Assemble the PRIVATE candidate table + report draft (results/private/, git-ignored) and the
PUBLIC control-metric summary (results/review_controls.csv, no coordinates)."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "private" / "search"
P = ROOT / "results" / "private"


def main():
    k = pd.read_csv(P / "review" / "key.csv").merge(pd.read_csv(P / "review" / "scores.csv"), on="id")
    ctl = (
        k.groupby(["group", "src"])
        .score.agg(
            n="size", mean="mean", n_score_ge2=lambda x: int((x >= 2).sum()), n_score3=lambda x: int((x == 3).sum())
        )
        .round(2)
        .reset_index()
    )
    ctl.to_csv(ROOT / "results" / "review_controls.csv", index=False)
    v = pd.read_csv(P / "crosscheck" / "verdicts.csv")
    x = pd.read_csv(P / "crosscheck" / "crosscheck.csv")
    F = pd.read_csv(P / "hearth_fields.csv")
    M = pd.read_parquet(S / "cand_mounds.parquet")
    W = pd.read_parquet(S / "cand_works.parquet")
    rows = []
    for r in k[k.src == "candidate"].itertuples():
        i = int(r.ref[1:])
        if r.group == "hearths":
            c = F[F.field == i].iloc[0]
            extra = (
                f"{int(c.n)} CNN hearths (p>=0.9) in field; {c.frac_known:.0%} in Carter et al.; "
                f"nearest GNIS furnace {c.nearest_gnis_furnace_km:.1f} km"
            )
        elif r.group == "mounds":
            c = M.loc[i]
            extra = f"dome score {c.score:.2f}, diam {c.diam:.1f} m, height {c.height:.2f} m"
        else:
            c = W.loc[i]
            extra = f"berm+ditch score {c.score_max:.3f} ({int(c.n_tiles)} tiles), LR p {c.p_lr_max:.2f}"
        rows.append(
            dict(
                id=r.id,
                cls=r.group,
                county=c.county,
                unit=c.unit,
                lat=round(c.lat, 6),
                lon=round(c.lon, 6),
                review_score=r.score,
                review_note=r.note,
                detector=extra,
            )
        )
    T = (
        pd.DataFrame(rows)
        .merge(v, on="id", how="left")
        .merge(x[["id", "osm_within_200m", "gnis_within_1km", "topo_oldest", "topo_mid"]], on="id", how="left")
    )
    T = T.sort_values(["review_score", "cls"], ascending=[False, True])
    T.to_csv(P / "reviewed_candidates.csv", index=False)
    print(ctl.to_string(index=False))
    print(T[["id", "cls", "county", "review_score", "verdict"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
