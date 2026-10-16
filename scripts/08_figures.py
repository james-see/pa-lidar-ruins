"""Figures for the README (all from already-public known sites; no candidate locations)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from palidar import dem, viz

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"
K = pd.read_csv(ROOT / "data" / "calibration" / "known_sites.csv").set_index("site_id")


def chip(win, size=400):
    z, _, _ = dem.read(ROOT / "data" / "dem" / f"{win}.tif")
    z = viz.fill_nan(z)
    c, h = z.shape[0] // 2, size // 2
    return z[c - h : c + h, c - h : c + h]


def fig_methods():
    rows = [
        ("site_M05", "Miamisburg Mound (OH)"),
        ("site_F01", "Fort Foote (MD)"),
        ("site_E01", "Newark Great Circle (OH)"),
    ]
    fig, axs = plt.subplots(3, 4, figsize=(14, 10.5))
    for i, (w, name) in enumerate(rows):
        z = chip(w, 500)
        axs[i, 0].imshow(viz.hillshade(z, 315, 45, z=2), cmap="gray")
        axs[i, 1].imshow(viz.multi_hillshade(z), cmap="gray")
        l = viz.lrm(z, 15)
        axs[i, 2].imshow(np.clip(l, -0.8, 0.8), cmap="RdYlBu_r")
        axs[i, 3].imshow(viz.combo_rgb(z, 15))
        axs[i, 0].set_ylabel(name, fontsize=11)
    for j, t in enumerate(["hillshade (NW sun)", "16-direction hillshade", "local relief (DEM - smoothed)", "combo"]):
        axs[0, j].set_title(t)
    for a in axs.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("USGS 3DEP 1 m bare-earth DEM, 500 m windows: four ways to look at it", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "01_visualisations.png", dpi=80)


def fig_gallery():
    ids = [
        "M01",
        "M02",
        "M06",
        "M16",
        "M26",
        "M46",
        "E01",
        "E03",
        "E11",
        "F01",
        "F02",
        "F03",
        "F04",
        "F06",
        "F10",
        "F17",
        "F20",
        "F21",
    ]
    fig, axs = plt.subplots(3, 6, figsize=(16, 8.5))
    for a, s in zip(axs.flat, ids, strict=True):
        a.imshow(viz.combo_rgb(chip(f"site_{s}", 300), 15))
        a.set_title(f"{K.loc[s, 'name'][:26]} ({K.loc[s, 'state']})", fontsize=8)
        a.axis("off")
    fig.suptitle("Calibration: known, public sites in the same lidar (300 m windows, combo view)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "02_known_sites_gallery.png", dpi=80)


def fig_mounds():
    P = pd.read_csv(RES / "mounds_known.csv")
    N = pd.read_csv(RES / "mounds_negative_scores.csv")
    n_bg = N[N.kind == "bg"].win.nunique()
    n_aml = N[N.kind == "aml"].win.nunique()
    ths = np.round(np.arange(0.5, 12.01, 0.25), 2)
    rec_all = [(P.score >= t).mean() for t in ths]
    rec_clear = [(P[P.audit == "clear_dome"].score >= t).mean() for t in ths]
    fp_bg = [(N[N.kind == "bg"].score >= t).sum() / (n_bg * 1.44) for t in ths]
    fp_aml = [(N[N.kind == "aml"].score >= t).sum() / (n_aml * 1.44) for t in ths]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(fp_bg, rec_clear, "-o", ms=3, label="clearly visible domes (n=25) vs random PA")
    ax[0].plot(fp_bg, rec_all, "-o", ms=3, label="all 47 published mounds vs random PA")
    ax[0].plot(fp_aml, rec_clear, "--", label="clear domes vs strip-mine windows")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("candidates per km² above threshold (upper bound on false positives)")
    ax[0].set_ylabel("recall (candidate within 30 m)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Detector A (mounds): recall vs background rate")
    ax[0].grid(alpha=0.3)
    # hit map on one window
    w = "site_M47"
    z, _, _ = dem.read(ROOT / "data" / "dem" / f"{w}.tif")
    d = pd.read_csv(ROOT / "data" / "cache" / "mounds" / f"{w}.csv")
    h = 200
    cc = z.shape[0] // 2
    sub = viz.fill_nan(z)[cc - h : cc + h, cc - h : cc + h]
    ax[1].imshow(viz.combo_rgb(sub, 15))
    dd = d[(abs(d.row - cc) < h) & (abs(d.col - cc) < h)]
    sc = ax[1].scatter(
        dd.col - cc + h,
        dd.row - cc + h,
        c=dd.score,
        s=18 + 6 * dd.score,
        cmap="viridis",
        edgecolors="w",
        linewidths=0.5,
    )
    ax[1].set_title("Ashland Central Park mound group (KY), 400 m: candidates coloured by score")
    plt.colorbar(sc, ax=ax[1], fraction=0.04)
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "03_mound_calibration.png", dpi=80)


def fig_hearths():
    cw = pd.read_csv(RES / "hearths_sgl43_test_curve.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(cw.recall, cw.precision, "-o", ms=3, label="measured vs reference")
    a = json.loads((RES / "hearths_visual_audit.json").read_text())
    r9 = cw.set_index("threshold").loc[0.9]
    ax[0].errorbar(
        [r9.recall],
        [np.mean(a["audit_adjusted_precision_at_0.9_range"])],
        yerr=[[0.02], [0.02]],
        fmt="s",
        color="C3",
        label="p≥0.9 after blind visual audit",
    )
    for t in (0.5, 0.8, 0.9, 0.95):
        rr = cw.set_index("threshold").loc[t]
        ax[0].annotate(f"p≥{t}", (rr.recall, rr.precision), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_title("Detector B (charcoal hearths): SGL 43 held-out east half")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    src = rasterio.open(ROOT / "data" / "dem" / "rch_sgl43_full.tif")
    m = gpd.read_file(ROOT / "data" / "raw" / "rch" / "rch_master_final.geojson")
    m = m[(m.SGL.astype(str) == "43") & (m.Confirm == "Y")].to_crs(src.crs)
    T = ~src.transform
    ref = np.array([T * (x, y) for x, y in zip(m.geometry.x, m.geometry.y, strict=True)])[:, ::-1]
    det = pd.read_csv(RES / "private" / "sgl43_test_dets.csv")
    det = det[det.p >= 0.9]
    # densest 800 m box of the test half
    rr = ref[ref[:, 1] > src.width / 2]
    H = np.histogram2d(rr[:, 0], rr[:, 1], bins=[np.arange(0, src.height, 800), np.arange(0, src.width, 800)])
    i, j = np.unravel_index(H[0].argmax(), H[0].shape)
    r0, c0 = int(H[1][i]), int(H[2][j])
    z = viz.fill_nan(src.read(1, window=((r0, r0 + 800), (c0, c0 + 800))))
    ax[1].imshow(viz.combo_rgb(z, 10, clip=0.4))
    k = (rr[:, 0] >= r0) & (rr[:, 0] < r0 + 800) & (rr[:, 1] >= c0) & (rr[:, 1] < c0 + 800)
    ax[1].scatter(rr[k, 1] - c0, rr[k, 0] - r0, s=60, facecolors="none", edgecolors="k", label="reference hearth")
    dk = det[(det.row >= r0) & (det.row < r0 + 800) & (det.col >= c0) & (det.col < c0 + 800)]
    ax[1].scatter(dk.col - c0, dk.row - r0, s=10, c="b", label="detection p≥0.9")
    found = (cKDTree(dk[["row", "col"]].to_numpy()).query(rr[k])[0] <= 6).mean() if len(dk) else 0
    ax[1].set_title(f"800 m test block: ring with dot = found ({found:.0%} here)")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "04_hearth_calibration.png", dpi=80)


def fig_works():
    P = pd.read_csv(RES / "earthworks_known_multi.csv")
    N = pd.read_csv(RES / "earthworks_negative_tiles_multi.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(N.score.max(), P.score.max()), 60)
    ax.hist(N[N.kind == "bg"].score, bins, alpha=0.5, density=True, label="random PA 100 m tiles")
    ax.hist(N[N.kind == "aml"].score, bins, alpha=0.5, density=True, label="strip-mine 100 m tiles")
    for cls, col, yy in (("military", "C3", 0.9), ("enclosure", "C2", 0.8)):
        g = P[P.cls == cls]
        ax.plot(
            g.score,
            np.full(len(g), yy) * ax.get_ylim()[1],
            "|",
            color=col,
            ms=18,
            mew=2,
            label=f"known {cls} (best tile)",
        )
    ax.set_xlabel("berm+ditch tile score")
    ax.set_ylabel("density")
    ax.set_title("Detector C (linear earthworks): known sites vs background tiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "05_earthwork_calibration.png", dpi=80)


if __name__ == "__main__":
    fig_methods()
    fig_gallery()
    fig_mounds()
    fig_hearths()
    fig_works()
    print(sorted(p.name for p in FIG.glob("*.png")))
