#!/usr/bin/env bash
# Phase-2 bulk downloads (verified by size/zip test). Safe to re-run.
set -u
cd "$(dirname "$0")/.."
get() { # url out
  for i in 1 2 3 4 5; do
    curl -fL --retry 3 -C - -o "$2" "$1" && return 0
    sleep 10
  done
  echo "FAILED $1"; return 1
}
get https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf data/raw/osm/pennsylvania-latest.osm.pbf
curl -fsL https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf.md5 -o data/raw/osm/pa.md5
get https://www.pasda.psu.edu/download/dep/DigitizedMinedAreas2026_10.zip data/raw/dep/DigitizedMinedAreas2026_10.zip
get https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/DomesticNames/DomesticNames_PA_Text.zip data/raw/gnis/DomesticNames_PA_Text.zip
( cd data/raw/osm && md5 -q pennsylvania-latest.osm.pbf; cat pa.md5 )
unzip -tq data/raw/dep/DigitizedMinedAreas2026_10.zip && echo DEP_ZIP_OK
unzip -tq data/raw/gnis/DomesticNames_PA_Text.zip && echo GNIS_ZIP_OK
echo ALL_DONE
