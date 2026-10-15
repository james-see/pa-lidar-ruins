"""Record which USGS 1 m DEM project(s) underlie each calibration window centre (TNM API)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from palidar import dem
from palidar import evalutil as E

ROOT = Path(__file__).resolve().parents[1]
w = E.windows()
with ThreadPoolExecutor(8) as ex:
    t = list(ex.map(lambda r: dem.has_1m(r[1], r[0]), zip(w.lat, w.lon, strict=True)))
w["usgs_1m_products"] = [" | ".join(x) for x in t]
w["has_1m"] = [bool(x) and x != ["<lookup failed>"] for x in t]
w[["win", "kind", "cls", "has_1m", "usgs_1m_products"]].to_csv(
    ROOT / "results" / "dem_provenance.csv", index=False
)
print(w.groupby("kind").has_1m.agg(["sum", "count"]))
print(w[~w.has_1m][["win", "usgs_1m_products"]].to_string())
