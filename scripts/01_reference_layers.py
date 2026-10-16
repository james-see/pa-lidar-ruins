"""Download public reference layers (PASDA ArcGIS REST) used for calibration and search planning.

- PGC State Game Lands (for the SGL 43 closed-world hearth reference + phase-2 search areas)
- DCNR State Forests and State Parks (phase-2 search areas)
- DCNR Physiographic Provinces (PA outline; sampling random background windows)
- DEP Abandoned Mine Land inventory polygons (strip mines / spoil / pits = hard negatives)
"""

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ref"
OUT.mkdir(parents=True, exist_ok=True)
B = "https://mapservices.pasda.psu.edu/server/rest/services/pasda"
LAYERS = {
    "state_game_lands": (f"{B}/PennsylvaniaGameCommission/MapServer/4", "1=1"),
    "state_forests": (f"{B}/DCNR/MapServer/6", "1=1"),
    "state_parks": (f"{B}/DCNR/MapServer/8", "1=1"),
    "physio_provinces": (f"{B}/DCNR/MapServer/16", "1=1"),
    "counties": (f"{B}/PennDOT/MapServer/7", "1=1"),
    "anf_compartments": (f"{B}/AlleghenyNationalForest/MapServer/0", "1=1"),
    "aml_polys": (
        f"{B}/DEP/MapServer/1",
        "sf_probl_1 LIKE '%Spoil%' OR sf_probl_1 LIKE '%Strip%' OR sf_probl_1 LIKE '%Gob%' "
        "OR sf_probl_1 LIKE '%Pit%' OR sf_probl_1 LIKE '%Highwall%' OR sf_probl_1 LIKE '%Embankment%'",
    ),
}


def get(url, params):
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print("retry", e, file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def fetch(name, url, where, page=1000):
    out = OUT / f"{name}.geojson"
    if out.exists():
        print("have", out)
        return
    n = get(f"{url}/query", {"where": where, "returnCountOnly": "true", "f": "json"})["count"]
    feats = []
    for off in range(0, n, page):
        d = get(
            f"{url}/query",
            {
                "where": where,
                "outFields": "*",
                "f": "geojson",
                "outSR": 4326,
                "resultOffset": off,
                "resultRecordCount": page,
                "maxAllowableOffset": 0.00005,
            },
        )
        feats += d["features"]
        print(name, len(feats), "/", n)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


if __name__ == "__main__":
    for k, (u, w) in LAYERS.items():
        fetch(k, u, w)
