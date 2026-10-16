"""Rebuild results/hearths_cnn_compare.json from the CSVs written by 14_train_hearth_cnn.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    out = {}
    for name in ("gbm_phase1", "cnn", "mean_gbm_cnn"):
        cv = pd.read_csv(RES / f"hearths_sgl43_test_curve_{name}.csv")
        out[name] = cv.set_index("threshold")[["recall", "precision", "det_per_km2"]].round(3).to_dict("index")
    R = pd.read_csv(RES / "hearths_windows_compare.csv")
    rec = (
        R[R.kind == "rch_rec"]
        .groupby(["model", "threshold"])
        .apply(lambda g: np.average(g.recall, weights=g.n_ref), include_groups=False)
        .unstack()
    )
    den = R[(R.kind != "rch_rec") & (R.split == "test")].groupby(["model", "kind", "threshold"]).det_per_km2.median()
    summ = dict(
        sgl43_east=out,
        recall_windows=rec.round(3).to_dict("index"),
        density_test_windows={f"{m}|{k}|{t}": round(v, 2) for (m, k, t), v in den.items()},
    )
    (RES / "hearths_cnn_compare.json").write_text(json.dumps(summ, indent=1))
    for k, v in out.items():
        print(k)
        print(pd.DataFrame(v).T.loc[[0.5, 0.7, 0.8, 0.9, 0.95]].to_string())
    print(rec.round(3))
    print(den.unstack().round(2))


if __name__ == "__main__":
    main()
