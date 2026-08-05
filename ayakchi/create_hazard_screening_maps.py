"""Create transparent first-pass hazard susceptibility maps for Ayakchi.

These products combine existing morphology and Earth Engine layers. They are
screening indices (relative susceptibility inside this basin), not calibrated
probabilities, forecasts, inundation depths, or engineering design products.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parent
MORPH = ROOT / "morphology_outputs"
HYDRO = ROOT / "hydromorphology_outputs"
GEE = ROOT / "gee_environmental_data" / "rasters"
OUT = ROOT / "hazard_screening_outputs"
BASE = MORPH / "01_dem_utm42n_30m.tif"
NODATA = -9999.0
CLASS_NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]


def read_base():
    with rasterio.open(BASE) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype("float32")
        return dem, src.profile.copy()


def aligned(path, profile, band=1, resampling=Resampling.bilinear):
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        source = src.read(band, masked=True).astype("float32").filled(np.nan)
        reproject(source, destination, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return destination


def robust01(array, mask, low=2, high=98):
    values = array[mask & np.isfinite(array)]
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) / max(hi - lo, 1e-9), 0, 1)


def aspect_exposure(aspect, preferred=180):
    """0 north-facing to 1 south-facing (preferred=180 degrees)."""
    return (1 + np.cos(np.radians(aspect - preferred))) / 2


def landcover_score(classes, scores):
    result = np.full(classes.shape, np.nan, dtype="float32")
    for code, value in scores.items():
        result[classes == code] = value
    return result


def classify(index, mask):
    """Fixed relative susceptibility thresholds for cross-map comparability."""
    classes = np.full(index.shape, np.nan, dtype="float32")
    classes[mask] = np.digitize(index[mask], [0.2, 0.4, 0.6, 0.8]) + 1
    return classes


def write_raster(path, data, profile, dtype="float32", nodata=NODATA):
    metadata = profile.copy()
    metadata.update(dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    output = np.where(np.isfinite(data), data, nodata).astype(dtype)
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(output, 1)


def extent(profile):
    t = profile["transform"]
    return [t.c, t.c + profile["width"] * t.a,
            t.f + profile["height"] * t.e, t.f]


def save_map(path, classes, hillshade, profile, title, subtitle):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(hillshade, extent=extent(profile), cmap="gray", alpha=.22)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(.5, 6, 1), cmap.N)
    image = ax.imshow(classes, extent=extent(profile), cmap=cmap, norm=norm,
                      interpolation="nearest", alpha=.86)
    cb = fig.colorbar(image, ax=ax, shrink=.75, ticks=np.arange(1, 6))
    cb.ax.set_yticklabels(CLASS_NAMES)
    cb.set_label("Relative susceptibility")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.text(.5, 1.012, subtitle, transform=ax.transAxes, ha="center", fontsize=8, color=".3")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.annotate("N", xy=(.94, .93), xytext=(.94, .83), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=2, headwidth=8),
                ha="center", fontweight="bold")
    ax.text(.01, .01, "WGS 84 / UTM zone 42N | 30 m screening grid",
            transform=ax.transAxes, fontsize=7, color=".25")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    dem, profile = read_base()
    mask = np.isfinite(dem)
    slope = aligned(MORPH / "02_slope_degrees.tif", profile)
    aspect = aligned(MORPH / "03_aspect_degrees.tif", profile)
    hillshade = aligned(MORPH / "04_hillshade.tif", profile)
    relief = aligned(MORPH / "05_local_relief_1km.tif", profile)
    tpi = aligned(MORPH / "06_tpi_1km.tif", profile)
    accumulation = aligned(HYDRO / "10_contributing_area_km2.tif", profile)
    landcover = aligned(GEE / "01_dynamic_world_mode_growing_season_2025.tif", profile,
                        resampling=Resampling.nearest)
    indices_path = GEE / "03_sentinel2_environmental_indices_growing_season_2025.tif"
    ndvi = aligned(indices_path, profile, band=1)
    ndmi = aligned(indices_path, profile, band=2)
    snow = aligned(GEE / "08_modis_snow_frequency_2001_2025.tif", profile)

    # Fill only small optical gaps with basin medians; retain the DEM basin mask.
    for layer in (ndvi, ndmi, snow):
        layer[mask & ~np.isfinite(layer)] = np.nanmedian(layer[mask])

    slope_n = robust01(slope, mask)
    relief_n = robust01(relief, mask)
    accumulation_n = robust01(np.log1p(accumulation), mask)
    low_tpi_n = robust01(-tpi, mask)
    ndvi_n = robust01(ndvi, mask)
    ndvi_low = 1 - ndvi_n
    ndmi_low = 1 - robust01(ndmi, mask)
    snow_n = robust01(snow, mask)
    elevation_n = robust01(dem, mask)
    southness = aspect_exposure(aspect)
    southness[mask & ~np.isfinite(southness)] = .5  # neutral exposure on flat cells

    runoff_lc = landcover_score(landcover, {
        0: 1.0, 1: .25, 2: .45, 3: .8, 4: .65, 5: .55, 6: 1.0, 7: .85, 8: .5})
    erosion_lc = landcover_score(landcover, {
        0: 0, 1: .2, 2: .55, 3: .25, 4: .75, 5: .7, 6: .2, 7: 1.0, 8: .15})
    instability_lc = landcover_score(landcover, {
        0: 0, 1: .25, 2: .6, 3: .35, 4: .7, 5: .6, 6: .4, 7: .9, 8: .25})
    fire_lc = landcover_score(landcover, {
        0: 0, 1: .8, 2: .9, 3: .3, 4: .6, 5: 1.0, 6: .1, 7: .15, 8: 0})
    drought_lc = landcover_score(landcover, {
        0: 0, 1: .35, 2: .85, 3: .25, 4: 1.0, 5: .9, 6: .3, 7: .55, 8: .1})

    # Replace any unclassified pixels conservatively with neutral susceptibility.
    for layer in (runoff_lc, erosion_lc, instability_lc, fire_lc, drought_lc):
        layer[mask & ~np.isfinite(layer)] = .5

    hazards = {
        "rapid_runoff": {
            "index": .38*accumulation_n + .22*slope_n + .20*runoff_lc + .20*relief_n,
            "title": "Rapid Runoff and Channel-Concentration Susceptibility",
            "subtitle": "Contributing area 38% | slope 22% | land cover 20% | local relief 20%",
        },
        "erosion": {
            "index": .35*slope_n + .25*relief_n + .25*erosion_lc + .15*ndvi_low,
            "title": "Soil-Erosion Susceptibility",
            "subtitle": "Slope 35% | local relief 25% | land cover 25% | low vegetation cover 15%",
        },
        "terrain_instability": {
            "index": .42*slope_n + .23*relief_n + .15*low_tpi_n + .20*instability_lc,
            "title": "Terrain-Instability Susceptibility",
            "subtitle": "Slope 42% | relief 23% | lower-slope position 15% | land cover 20%",
        },
        "drought_stress": {
            "index": .32*ndmi_low + .28*ndvi_low + .25*drought_lc + .15*southness,
            "title": "Drought and Ecosystem-Stress Susceptibility",
            "subtitle": "Low moisture 32% | low greenness 28% | land cover 25% | solar exposure 15%",
        },
        "wildfire": {
            "index": .30*ndmi_low + .25*fire_lc + .20*southness + .15*slope_n + .10*ndvi_n,
            "title": "Wildfire Susceptibility",
            "subtitle": "Dryness 30% | fuel cover 25% | solar exposure 20% | slope 15% | biomass 10%",
        },
        "snowmelt_runoff": {
            "index": .40*snow_n + .20*elevation_n + .20*slope_n + .20*accumulation_n,
            "title": "Seasonal Snowmelt-Runoff Potential",
            "subtitle": "Historical snow 40% | elevation 20% | slope 20% | flow concentration 20%",
        },
    }

    class_arrays = {}
    statistics = []
    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    for number, (key, spec) in enumerate(hazards.items(), start=1):
        index = np.clip(spec["index"], 0, 1).astype("float32")
        index[~mask] = np.nan
        classes = classify(index, mask)
        class_arrays[key] = classes
        write_raster(OUT / f"{number:02d}_{key}_index.tif", index, profile)
        write_raster(OUT / f"{number:02d}_{key}_class.tif", classes, profile, "uint8", 255)
        save_map(OUT / f"map_{number:02d}_{key}.png", classes, hillshade, profile,
                 f"Ayakchi Basin — {spec['title']}", spec["subtitle"])
        for class_id, class_name in enumerate(CLASS_NAMES, start=1):
            count = int(np.sum(classes == class_id))
            statistics.append([key, class_id, class_name, count*cell_area_km2,
                               count / mask.sum() * 100])

    # Equal-weight synthesis of the five non-seasonal hazards.
    multi_keys = ["rapid_runoff", "erosion", "terrain_instability", "drought_stress", "wildfire"]
    multi_index = np.mean([hazards[key]["index"] for key in multi_keys], axis=0)
    multi_index[~mask] = np.nan
    multi_classes = classify(multi_index, mask)
    write_raster(OUT / "07_multi_hazard_hotspot_index.tif", multi_index, profile)
    write_raster(OUT / "07_multi_hazard_hotspot_class.tif", multi_classes, profile, "uint8", 255)
    save_map(OUT / "map_07_multi_hazard_hotspots.png", multi_classes, hillshade, profile,
             "Ayakchi Basin — Multi-Hazard Hotspot Screening",
             "Equal-weight synthesis: runoff, erosion, terrain instability, drought stress and wildfire")
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        count = int(np.sum(multi_classes == class_id))
        statistics.append(["multi_hazard", class_id, class_name, count*cell_area_km2,
                           count / mask.sum() * 100])

    with (OUT / "hazard_class_areas.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["hazard", "class_id", "class_name", "area_km2", "basin_percent"])
        writer.writerows(statistics)

    methodology = {
        "purpose": "relative basin-scale susceptibility screening",
        "limitations": [
            "not a calibrated probability or forecast", "not a flood-inundation or landslide inventory model",
            "does not include detailed soils, geology, roads, buildings, population or observed event inventories",
            "class thresholds are fixed index intervals and express relative susceptibility inside Ayakchi",
        ],
        "hazards": {key: {"title": value["title"], "weights": value["subtitle"]}
                    for key, value in hazards.items()},
        "multi_hazard_components": multi_keys,
        "inputs": ["SRTM terrain derivatives", "D8 contributing area", "Dynamic World 2025 land cover",
                   "Sentinel-2 2025 NDVI and NDMI", "MODIS 2001–2025 snow frequency"],
    }
    (OUT / "methodology.json").write_text(json.dumps(methodology, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Ayakchi hazard screening maps\n\n"
        "Seven relative susceptibility products derived from the existing morphology, hydrology and "
        "Earth Engine layers. Each hazard has a continuous 0–1 index, a five-class raster and a 300 dpi map. "
        "See `methodology.json` for weights and limitations and `hazard_class_areas.csv` for class areas.\n\n"
        "These are prioritization layers for CIEWS development, not calibrated hazard probabilities, forecasts, "
        "flood depths or engineering-design products. Validation requires event inventories and local soils, "
        "geology, infrastructure and exposure data.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
