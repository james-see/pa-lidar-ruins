# pa-lidar-ruins

Looking for unrecorded archaeological earthworks in Pennsylvania with free USGS 3DEP lidar,
following the loop in Nate's [*I went looking for lost cities with Claude*](https://bakerbench.substack.com/)
(Baker Bench, Oct 2026):

1. build detectors on bare-earth lidar, 2. prove they work on sites we already know,
3. search big areas, 4. have independent "skeptics" try to kill every find,
5. check survivors against the published record, 6. hand what's left to the people who can judge it.

**This repo is phase 1: steps 1–2 only (detectors + calibration). No wide-area search has been run yet.**

![four views](figures/01_visualisations.png)

## Site-location policy

Exact locations of candidate (possibly unrecorded) sites are **never** committed here. Search
outputs, candidate coordinates and chips live in the git-ignored `results/private/` and will be
reported privately to the Pennsylvania State Historic Preservation Office (PA SHPO / PHMC) and the
land manager (DCNR, PGC, NPS). Pennsylvania's site files (CRGIS / PASS) are not public, and we only
publish county-level summaries. Calibration uses sites that are already public (OSM, Wikipedia,
NRHP / park pages, and a published open hearth dataset).

## Three target classes, three detectors

| | Target | Detector (all classical except the hearth ranker) |
|---|---|---|
| **A** | Prehistoric mounds (Adena/Hopewell/Monongahela-type conical mounds) | Multi-scale local relief model (LRM = DEM − Gaussian-smoothed DEM at 12/25/50/100 m). For every LRM peak, grow the half-height blob; keep compact, round blobs 6–110 m across; score = relief × circularity × solidity × dome-ness. `palidar.detectors.detect_mounds` |
| **B** | Relict charcoal hearths (RCH): flat round platforms ~8–15 m cut into hillsides, the fuel-making sites of 18th–19th c. iron furnaces | (1) candidates = peaks of a "flat disc inside a sloping ring" contrast map (4–7 m radii); (2) a gradient-boosted classifier on 29 rotation-normalised morphometrics (ring-wise slope profile, upslope-vs-downslope relief = the cut/fill signature, platform planarity, regional slope). `palidar.hearths` |
| **C** | Forts / military earthworks (colonial, Revolutionary, War of 1812, Civil War); also prehistoric enclosures | Hessian ridge (berm) and valley (ditch) maps on a 25 m LRM at 1.5/3/6 m scales; berm pixels with a ditch within 7 m (and vice-versa) = "pairs"; 100 m tiles scored by pair density × (0.3 + orientation diversity). `palidar.detectors.tile_scores` |

## Data

* **Elevation:** USGS 3DEP bare-earth DEM via the 3DEP ImageServer `exportImage`
  (best-available; requested at 1 m in the local NAD83/UTM zone). `scripts/10_provenance.py` checks
  the TNM API: **all 200 calibration windows lie on USGS 1 m DEM projects** (`results/dem_provenance.csv`).
* **Hearth reference:** Carter, Blackadar & Conner (2021) *When Computers Dream of Charcoal*,
  Advances in Archaeological Practice 9(4); open data on Zenodo (CC-BY):
  manual hearth points [4593622](https://zenodo.org/records/4593622) (4,361 points),
  assessed Mask R-CNN predictions [4593767](https://zenodo.org/records/4593767),
  false negatives around SGL 43 [4758647](https://zenodo.org/records/4758647).
* **Known mounds / enclosures / forts:** `data/calibration/known_sites.csv` (87 sites, each with its
  source: OpenStreetMap feature and/or Wikipedia article). Pennsylvania's own mounds were mostly
  destroyed (e.g. McKees Rocks Mound, quarried away) and the survivors are not publicly located, so
  class-A calibration uses 47 mounds and 15 enclosures from the same Upper Ohio Valley tradition in
  OH / WV / KY. Class C uses 25 earthworks in PA, MD, VA, WV, DC and OH.
* **Negative controls:** 60 random 1.2 km windows inside Pennsylvania (`bg`) and 40 windows on PA DEP
  Abandoned Mine Land polygons (dry/flooded strip mines, spoil piles; `aml`), via PASDA.
* **Search-planning layers (phase 2):** PGC State Game Lands, DCNR State Forests and State Parks (PASDA).

Every visible known site was also eyeballed in the combo view and tagged (`visual_audit` column:
`clear_dome`, `faint`, `not_visible`, `platform`, `ring_or_u`, ...). This was one person, not blind
to the detector, and it is reported separately from the headline numbers. About a third of the
"known" mounds are not visible at all in 1 m lidar (plowed down, excavated, or the published
point is off).

![known sites](figures/02_known_sites_gallery.png)

## Calibration results (all measured; method stated with each)

### A. Mounds (`scripts/04_eval_mounds.py`, `results/mounds_*.csv`)

Found = a candidate within 30 m of the published point. Background rate = candidates per km² in
random PA windows (an upper bound on false positives, since a few could be real). Fixed threshold
grid (not tuned). Dev/test = odd/even site ids.

| score ≥ | recall, 25 clearly visible domes | recall, all 47 published | dev / test (clear domes) | candidates/km², random PA | candidates/km², strip mines |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 96% | 68% | 91% / 100% | 65 | 114 |
| 1 | 92% | 60% | 91% / 93% | 16 | 40 |
| 2 | 80% | 43% | 73% / 86% | 3.0 | 11.6 |
| 3 | 68% | 36% | 64% / 71% | 0.9 | 4.8 |
| 5 | 44% | 23% | 55% / 36% | 0.17 | 1.1 |

Inside its own 1.2 km window the known mound was the #1-ranked candidate for 13 of 47 sites and in
the top 5 for 21 (median window has 238 candidates). The detector misses ring/U-shaped mounds,
platform mounds and loaf-shaped mounds, which it wasn't built for. In practice: at score ≥ 2 it finds
about 4 in 5 clearly visible mounds and returns about 3 candidates per km² of Pennsylvania, roughly
4× that on strip-mined land.

![mounds](figures/03_mound_calibration.png)

### B. Charcoal hearths (`scripts/05_eval_hearths.py`, `results/hearths_*`)

*Closed-world test:* the full lidar image around State Game Land 43 (Berks/Chester Co., next to
Hopewell Furnace NHS), where Carter et al. visually checked every model prediction and logged the
hearths their model missed. That gives 908 reference hearths. The ranker was **trained on the west
half and tested on the east half** (516 reference hearths). Found = a detection within 6 m.

| p ≥ | recall | precision (vs reference) | detections/km² |
|---:|---:|---:|---:|
| 0.5 | 78.5% | 14.7% | 46 |
| 0.7 | 72.1% | 20.3% | 31 |
| 0.8 | 68.4% | 26.3% | 22 |
| 0.9 | 61.0% | 36.7% | 14 |
| 0.95 | 49.0% | 51.0% | 8 |

The reference is itself incomplete, so measured precision is a lower bound. A **blind visual audit**
(`scripts/07_audit_chips.py`, `results/hearths_visual_audit.json`) looked at 30 random unmatched
detections at p ≥ 0.9, shuffled with 10 matched ones as controls. All 10 controls were judged hearth or
maybe. Of the 30 unmatched, 5 looked like hearths and 2 were maybes, which puts precision at p ≥ 0.9
around **47–51%**. This is indicative only (one rater, n = 30). The real false positives were road and
trail junctions, road cuts, stream banks and ditches.

*Independent recall:* in 12 four-km² windows in other game lands, the authors' manual hearth points
(581, not exhaustive, so recall only) gave recall of 88.5% / 80.9% / 75.0% / 58.2% at p ≥ 0.3 / 0.5 / 0.7 / 0.9
(per-window 44–80% at 0.9). Detection density at p ≥ 0.9 is a median of 3.5/km² in random PA windows
and 3.5/km² on strip-mine windows. Random PA forest genuinely contains hearths, so that isn't all false.

![hearths](figures/04_hearth_calibration.png)

### C. Military earthworks and enclosures (`scripts/06_eval_earthworks.py`, `results/earthworks_*`)

This is a tile-ranking test like Nate's pueblo test: each 1.2 km window is cut into 100 m tiles (14,400
background tiles: 8,640 random PA and 5,760 strip-mine). For each known site we take its best tile
within 150 m (250 m for enclosures, whose points are the centres of 200–400 m circles) and ask what
fraction of background tiles score higher. Two parameter sets were compared on the dev split
(`small` vs `multi` scales). `multi` was better on dev (military median 7.0% vs 10.3%) and is the
one reported.

| class | sites | median % of random-PA tiles scoring higher | median % of strip-mine tiles scoring higher | in top 5% vs random PA |
|---|---:|---:|---:|---:|
| military, clearly visible | 12 | 3.0% | 13.9% | 8 / 12 |
| military, faint / not visible | 13 | 7.6% | 27.3% | 5 / 13 |
| military, test split only | 12 | 5.7% | 22.5% | 6 / 12 |
| enclosures (mostly plowed Hopewell) | 15 | 8.1% | 28.5% | 6 / 15 |

**C is the weakest detector.** It separates forts from ordinary Pennsylvania terrain moderately well,
but strip-mine spoil ridges, road cut-and-fill and stream banks produce the same berm-next-to-ditch
signal. The best-preserved Hopewell circles (Newark Great Circle, Observatory Circle) score low
because their walls are much wider than the filter scales.

![earthworks](figures/05_earthwork_calibration.png)

## What's hard

* **Coal country.** Strip-mine spoil and highwalls are the dominant false positive for all three
  detectors: mounds come out about 4× denser per km² there, and earthwork tiles score far higher.
  Phase 2 must mask PA DEP *Digitized Mined Areas* + AML polygons before ranking.
* **Roads and drainage.** Cut-and-fill road benches, trail junctions and stream banks mimic hearth
  platforms and berm/ditch pairs.
* **Ground truth is thin and biased.** PA's site files aren't public. Many "known" mounds are invisible
  in lidar (plowed or excavated), and OSM points are sometimes off the feature. Known forts are mostly
  restored park sites, which are better preserved than whatever is still undiscovered.
* **Base rates.** Even 3 candidates/km² over a 1,000 km² forest is 3,000 leads. The skeptic and
  historic-record steps (4–5) have to do most of the filtering, as they did for Nate.

## Phase 2 plan (not started)

1. **Masks:** DEP Digitized Mined Areas + AML, quarries (OSM), PennDOT roads/rail buffers, water bodies.
2. **Search areas, by class**
   * **A – mounds:** river terraces and bluffs of the Ohio, Allegheny, Beaver, Monongahela and
     Youghiogheny valleys, and the Allegheny National Forest / Warren County (home of Sugar Run Mound),
     restricted to state forest, state game land and federal land outside mined areas.
   * **B – hearths:** iron-furnace districts on public forest: Michaux SF (Pine Grove, Caledonia),
     Rothrock SF (Greenwood), French Creek SP / SGL 43 (Hopewell), Weiser, Tuscarora, Buchanan and Bald Eagle SFs.
     Hearths themselves are common. What makes them interesting is that hearth fields with no known
     furnace nearby point to unrecorded furnaces and collier camps.
   * **C – earthworks:** 1863 Gettysburg-campaign corridor (Adams, Franklin, Cumberland and York
     counties; Harrisburg/Camp Hill militia defences), the Forbes and Braddock road corridors
     (1755–58 camps and redoubts), and South Mountain (Michaux SF).
3. **Detector upgrades:** train a small CNN on the 4,361 published hearth points with hard negatives;
   add a dome-profile check and hard-negative mining for mounds; add straight-segment / corner
   features and a strip-mine classifier for C.
4. **Skeptic review** (independent AI reviewers with hidden known-site and natural/modern controls), then
   **record check** (historic USGS topos back to the 1890s, OSM, GNIS, NAIP, published surveys).
5. **Private report** of surviving leads with exact locations to PA SHPO and the land manager.
   Only county-level counts get published.

## Reproduce

```bash
uv sync
uv run python scripts/01_reference_layers.py   # PASDA layers -> data/raw/ref
# hearth reference: download the three Zenodo files above into data/raw/rch/
uv run python scripts/02_build_windows.py      # windows.csv + ~200 DEM windows (~1.5 GB)
uv run python scripts/04_eval_mounds.py
uv run python scripts/05_eval_hearths.py
uv run python scripts/06_eval_earthworks.py multi
uv run python scripts/07_audit_chips.py 0.9    # private blind-audit grid
uv run python scripts/08_figures.py
uv run python scripts/09_tables.py
uv run pytest && uv run ruff check .
```

Raw data, DEMs and caches are git-ignored. `02_build_windows.py` also fetches the full SGL 43 lidar extent
(12.5 × 9.6 km) used for the hearth closed-world test.

## License

MIT, see [LICENSE](LICENSE). Hearth reference data © Carter, Blackadar & Conner, CC-BY 4.0.
Map data © OpenStreetMap contributors (ODbL). Elevation: USGS 3DEP (public domain).
