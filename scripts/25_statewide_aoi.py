"""Statewide hearth AOI: all PA public land (state forests, game lands, parks, ANF) MINUS the
phase-2 hearth areas already searched, split by county into units. Appends aoi='hearths_all'
rows to data/cache/aoi/aois.gpkg and tiles.csv (idempotent)."""

import geopandas as gpd
import pandas as pd

from palidar import search as S

ROOT = S.ROOT


def main():
    A = gpd.read_file(S.AOI_DIR / "aois.gpkg")
    A = A[A.aoi != "hearths_all"]
    pl = gpd.read_file(S.AOI_DIR / "public_land.gpkg")
    done = A[A.aoi == "hearths"].union_all()
    rest = pl.union_all().difference(done)
    co = gpd.read_file(ROOT / "data" / "raw" / "ref" / "counties.geojson").to_crs(S.CRS)
    rows = []
    for r in co.itertuples():
        g = rest.intersection(r.geometry.buffer(0))
        if g.area > 1e6:
            rows.append(dict(aoi="hearths_all", unit=f"rest {r.COUNTY_NAM.title()}", km2=g.area / 1e6, geometry=g))
    U = gpd.GeoDataFrame(rows, crs=S.CRS)
    print(len(U), "units,", round(U.km2.sum(), 1), "km2 public land not yet searched for hearths")
    pd.concat([A, U]).pipe(gpd.GeoDataFrame, crs=S.CRS).to_file(S.AOI_DIR / "aois.gpkg", driver="GPKG")
    T = pd.read_csv(S.AOI_DIR / "tiles.csv")
    T = T[T.aoi != "hearths_all"]
    NT = S.make_tiles(U)
    pd.concat([T, NT]).to_csv(S.AOI_DIR / "tiles.csv", index=False)
    print(len(NT), "new tiles")


if __name__ == "__main__":
    main()
