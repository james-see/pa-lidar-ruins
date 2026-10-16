"""Phase 2, step 1: search areas (public land only) and exclusion layers.

Public land = PGC State Game Lands + DCNR State Forests + DCNR State Parks + Allegheny National
Forest compartments + OSM National Park Service / national_park areas.

Exclusions (buffered, later rasterised per search tile):
  DEP Digitized Mined Areas + DEP AML inventory polygons (strip mines, spoil, highwalls)  +30 m
  OSM quarries / landfills / spoil heaps / mineshaft areas                              +30 m
  OSM roads (by class), tracks, paths, trails                                           6-25 m
  OSM railways incl. abandoned / disused grades                                         15 m
  OSM waterways (rivers 20 m, streams/ditches 8 m) and water polygons                   10 m
  OSM buildings                                                                          10 m
Search areas (AOIs):
  hearths: Michaux, Rothrock, Weiser, Tuscarora, Buchanan, Bald Eagle state forests; French Creek SP
           + all public land within 10 km of Hopewell Furnace (SGL 43 surroundings)
  mounds:  public land within 1.5 km of the Ohio, Allegheny, Beaver, Monongahela and Youghiogheny
           rivers (incl. the Allegheny Reservoir in ANF/Warren Co.)
  works:   public land in Adams, Franklin, Cumberland, York and Dauphin counties (1863 campaign,
           Harrisburg/Camp Hill defences, South Mountain) + within 3 km of US 30 (Forbes Road) and
           US 40 (Braddock Road) in PA
"""

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "raw" / "ref"
PBF = ROOT / "data" / "raw" / "osm" / "pennsylvania-latest.osm.pbf"
OUT = ROOT / "data" / "cache" / "aoi"
OUT.mkdir(parents=True, exist_ok=True)
CRS = 5070  # CONUS Albers (metres) for buffering / areas


T0 = time.time()


def t(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def tag(other, key):
    return other.str.extract(rf'"{key}"=>"([^"]*)"', expand=False)


def osm_layers():
    lines = pyogrio.read_dataframe(
        PBF,
        layer="lines",
        columns=["highway", "waterway", "railway", "other_tags", "name"],
        where="highway IS NOT NULL OR waterway IS NOT NULL OR railway IS NOT NULL",
    )
    t(f"lines {len(lines)}")
    lines["ref"] = tag(lines.other_tags.fillna(""), "ref")
    mp = pyogrio.read_dataframe(
        PBF,
        layer="multipolygons",
        columns=["building", "landuse", "natural", "boundary", "man_made", "leisure", "other_tags", "name"],
        where="building IS NOT NULL OR landuse IN ('quarry','landfill','industrial') OR natural='water' "
        "OR man_made IN ('spoil_heap','mineshaft','wastewater_plant') "
        "OR boundary IN ('national_park','protected_area')",
    )
    t(f"multipolygons {len(mp)}")
    return lines, mp


def main():
    lines, mp = osm_layers()
    lines = lines.to_crs(CRS)
    mp = mp.to_crs(CRS)
    # ---------------- public land
    pl = []
    for f, label in [
        ("state_game_lands", "SGL"),
        ("state_forests", "SF"),
        ("state_parks", "SP"),
        ("anf_compartments", "ANF"),
    ]:
        g = gpd.read_file(REF / f"{f}.geojson").to_crs(CRS)
        col = next((c for c in ("SF_Name", "NAME", "PARK_NAME", "StParkName", "PARKNAME") if c in g), None)
        name = g[col] if col else pd.Series([""] * len(g))
        pl.append(gpd.GeoDataFrame(dict(owner=label, name=name.astype(str)), geometry=g.geometry.make_valid(), crs=CRS))
    ot = mp.other_tags.fillna("")
    nps = mp[(mp.boundary == "national_park") | ot.str.contains('"operator"=>"National Park Service"')]
    pl.append(
        gpd.GeoDataFrame(dict(owner="NPS", name=nps.name.astype(str)), geometry=nps.geometry.make_valid(), crs=CRS)
    )
    PL = pd.concat(pl, ignore_index=True)
    PL = PL[PL.geometry.notna() & ~PL.geometry.is_empty]
    PL.to_file(OUT / "public_land.gpkg")
    t(f"public land polygons {len(PL)}, area {PL.union_all().area / 1e6:.0f} km2")
    pub = PL.union_all()

    # ---------------- exclusions
    dep = gpd.read_file(ROOT / "data" / "raw" / "dep" / "DigitizedMinedAreas2026_10.shp").to_crs(CRS)
    aml = gpd.read_file(REF / "aml_polys.geojson").to_crs(CRS)
    mined = pd.concat([dep[["geometry"]], aml[["geometry"]]]).geometry.make_valid().buffer(30)
    gpd.GeoDataFrame(geometry=mined, crs=CRS).to_file(OUT / "excl.gpkg", layer="mined")
    t(f"mined polygons {len(mined)}")
    q = mp[mp.landuse.isin(["quarry", "landfill"]) | mp.man_made.isin(["spoil_heap", "mineshaft"])]
    gpd.GeoDataFrame(geometry=q.geometry.make_valid().buffer(30), crs=CRS).to_file(OUT / "excl.gpkg", layer="quarry")
    hw = lines[lines.highway.notna()]
    width = hw.highway.map(
        {
            "motorway": 25,
            "trunk": 25,
            "primary": 18,
            "secondary": 15,
            "tertiary": 12,
            "motorway_link": 15,
            "trunk_link": 15,
            "primary_link": 12,
            "secondary_link": 10,
            "residential": 10,
            "unclassified": 10,
            "service": 8,
        }
    ).fillna(6)
    gpd.GeoDataFrame(dict(w=width), geometry=hw.geometry, crs=CRS).to_file(OUT / "excl.gpkg", layer="roads")
    rr = lines[lines.railway.notna() & ~lines.railway.isin(["platform", "proposed"])]
    gpd.GeoDataFrame(geometry=rr.geometry, crs=CRS).to_file(OUT / "excl.gpkg", layer="rail")
    ww = lines[lines.waterway.notna()]
    gpd.GeoDataFrame(
        dict(w=ww.waterway.map({"river": 20, "canal": 15}).fillna(8)), geometry=ww.geometry, crs=CRS
    ).to_file(OUT / "excl.gpkg", layer="waterways")
    wp = mp[mp.natural == "water"]
    gpd.GeoDataFrame(geometry=wp.geometry.make_valid(), crs=CRS).to_file(OUT / "excl.gpkg", layer="water")
    b = mp[mp.building.notna()]
    gpd.GeoDataFrame(geometry=b.geometry.make_valid(), crs=CRS).to_file(OUT / "excl.gpkg", layer="buildings")
    t(f"roads {len(hw)} rail {len(rr)} waterways {len(ww)} water {len(wp)} buildings {len(b)} quarries {len(q)}")

    # ---------------- AOIs
    sf = PL[PL.owner == "SF"]
    names = ["MICHAUX", "ROTHROCK", "WEISER", "TUSCARORA", "BUCHANAN", "BALD EAGLE"]
    parts = []
    for n in names:
        g = sf[sf.name.str.upper().str.contains(n)]
        parts.append(
            gpd.GeoDataFrame(dict(aoi=["hearths"], unit=[n.title() + " SF"]), geometry=[g.union_all()], crs=CRS)
        )
    hop = gpd.GeoSeries([Point(-75.7756, 40.1986)], crs=4326).to_crs(CRS).iloc[0]  # Hopewell Furnace NHS
    fc = PL[PL.intersects(hop.buffer(10000))].union_all().intersection(hop.buffer(10000))
    parts.append(
        gpd.GeoDataFrame(dict(aoi=["hearths"], unit=["French Creek / SGL 43 (10 km)"]), geometry=[fc], crs=CRS)
    )
    riv = lines[
        (lines.waterway == "river")
        & lines.name.fillna("").str.match(r"^(Ohio|Allegheny|Beaver|Monongahela|Youghiogheny) River$")
    ]
    res_ = mp[mp.name.fillna("").str.contains("Allegheny Reservoir")]
    rb = pd.concat([riv.geometry, res_.geometry]).buffer(1500).union_all()
    parts.append(
        gpd.GeoDataFrame(dict(aoi=["mounds"], unit=["W PA rivers 1.5 km"]), geometry=[rb.intersection(pub)], crs=CRS)
    )
    cty = gpd.read_file(REF / "counties.geojson").to_crs(CRS)
    cname = "COUNTY_NAM"
    camp = cty[cty[cname].str.upper().isin(["ADAMS", "FRANKLIN", "CUMBERLAND", "YORK", "DAUPHIN"])].union_all()
    roads = hw[hw.ref.fillna("").str.contains(r"\bUS 30\b|\bUS 40\b")]
    rd = roads.buffer(3000).union_all()
    parts.append(
        gpd.GeoDataFrame(
            dict(aoi=["works"], unit=["1863 campaign counties"]), geometry=[camp.intersection(pub)], crs=CRS
        )
    )
    parts.append(
        gpd.GeoDataFrame(
            dict(aoi=["works"], unit=["Forbes/Braddock road corridors (US30/US40 3 km)"]),
            geometry=[rd.intersection(pub)],
            crs=CRS,
        )
    )
    A = pd.concat(parts, ignore_index=True)
    A["km2"] = A.area / 1e6
    A.to_file(OUT / "aois.gpkg")
    print(A[["aoi", "unit", "km2"]].round(1).to_string())


if __name__ == "__main__":
    main()
