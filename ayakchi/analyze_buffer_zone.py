"""Create a 5 km contextual buffer and screen land cover and exposure.

The buffer is a study-area boundary for CIEWS stakeholder analysis. It is not
an inundation, dam-break, or flood-hazard boundary.
"""

import csv
import json
import shutil
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "downstream_analysis_outputs" / "connected_basin_boundary.geojson"
UPSTREAM = ROOT / "hydromorphology_outputs" / "basin_boundary.geojson"
FLOW_PATH = ROOT / "downstream_analysis_outputs" / "dam_to_city_flow_path.geojson"
OUT = ROOT / "buffer_zone_outputs"
CRS = "EPSG:32642"
BUFFER_M = 5000
RIVER_CORRIDOR_M = 2000

DW_NAMES = ["Water", "Trees", "Grass", "Flooded vegetation", "Crops",
            "Shrub/scrub", "Built", "Bare", "Snow/ice"]
DW_COLORS = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
             "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
WC_CLASSES = {10: "Trees", 20: "Shrub/scrub", 30: "Grass", 40: "Crops", 50: "Built",
              60: "Bare", 70: "Snow/ice", 80: "Water", 90: "Wetland",
              95: "Mangroves", 100: "Moss/lichen"}
WC_COLORS = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4",
             "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"]


def first_geometry(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return shape(data["features"][0]["geometry"])


def write_geojson(path, geometry, properties):
    obj = {"type": "FeatureCollection", "name": path.stem,
           "crs": {"type": "name", "properties": {"name": CRS}},
           "features": [{"type": "Feature", "properties": properties,
                         "geometry": mapping(geometry)}]}
    path.write_text(json.dumps(obj), encoding="utf-8")


def download(image, filename, region, scale):
    path = OUT / filename
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path.name}")
        return path
    params = {"name": path.stem, "region": region, "scale": scale, "crs": CRS,
              "format": "GEO_TIFF", "filePerBand": False}
    url = image.clip(region).getDownloadURL(params)
    temporary = path.with_suffix(".download")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    if zipfile.is_zipfile(temporary):
        with zipfile.ZipFile(temporary) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if len(names) != 1:
                raise RuntimeError(f"Expected one raster for {filename}, got {names}")
            with archive.open(names[0]) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary.unlink()
    else:
        temporary.replace(path)
    print(f"Downloaded {path.name}")
    return path


def raster_values(path, geometry):
    with rasterio.open(path) as src:
        mask = geometry_mask([mapping(geometry)], out_shape=(src.height, src.width),
                             transform=src.transform, invert=True)
        data = src.read(1)
        valid = mask & np.isfinite(data)
        if src.nodata is not None:
            valid &= data != src.nodata
        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
        return data[valid], pixel_area_km2


def landcover_stats(path, zones):
    rows = []
    for zone_name, geom in zones.items():
        values, pixel_area = raster_values(path, geom)
        total = len(values) * pixel_area
        for code, label in enumerate(DW_NAMES):
            area = float(np.count_nonzero(values == code) * pixel_area)
            rows.append({"zone": zone_name, "class_code": code, "land_cover": label,
                         "area_km2": round(area, 3),
                         "percent": round(100 * area / total, 2) if total else 0})
    with (OUT / "landcover_statistics.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return rows


def worldcover_stats(path, zones):
    rows = []
    for zone_name, geom in zones.items():
        values, pixel_area = raster_values(path, geom)
        total = len(values) * pixel_area
        for code, label in WC_CLASSES.items():
            area = float(np.count_nonzero(values == code) * pixel_area)
            rows.append({"zone": zone_name, "class_code": code, "land_cover": label,
                         "area_km2": round(area, 3),
                         "percent": round(100 * area / total, 2) if total else 0})
    with (OUT / "worldcover_statistics.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return rows


def exposure_stats(built_path, population_path, zones):
    rows = []
    for zone_name, geom in zones.items():
        built, built_pixel = raster_values(built_path, geom)
        population, _ = raster_values(population_path, geom)
        rows.append({
            "zone": zone_name,
            "zone_area_km2": round(geom.area / 1e6, 3),
            "built_surface_km2": round(float(np.nansum(np.maximum(built, 0))) / 1e6, 3),
            "indicative_population_2020": round(float(np.nansum(np.maximum(population, 0)))),
            "built_raster_pixel_area_km2": round(built_pixel, 6),
        })
    with (OUT / "exposure_statistics.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return rows


def plot_landcover(path, connected, buffer_geom, upstream, flow_path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    cmap = ListedColormap(DW_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    for geom, color, width, label in [(buffer_geom, "#111111", 1.2, "5 km context buffer"),
                                       (connected, "#ffffff", 1.4, "Connected basin"),
                                       (upstream, "#00ffff", 1.5, "Upstream dam catchment")]:
        x, y = geom.exterior.xy; ax.plot(x, y, color=color, linewidth=width, label=label)
    x, y = flow_path.xy; ax.plot(x, y, color="#08306b", linewidth=2, label="Dam-to-outlet flow path")
    cbar = fig.colorbar(image, ax=ax, ticks=range(9), fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels(DW_NAMES); cbar.set_label("Dynamic World modal class")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_title("Ayakchi CIEWS — land cover in the 5 km contextual buffer")
    ax.set_xlabel("Easting (m, UTM zone 42N)"); ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(OUT / "map_01_buffer_landcover.png", dpi=300); plt.close(fig)


def plot_exposure(path, connected, buffer_geom, upstream, flow_path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    positive = np.asarray(data.filled(0)); vmax = max(1, float(np.nanpercentile(positive[positive > 0], 99))) if np.any(positive > 0) else 1
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(data, extent=extent, cmap="magma", vmin=0, vmax=vmax)
    for geom, color, width, label in [(buffer_geom, "#222222", 1.2, "5 km context buffer"),
                                       (connected, "#00ffff", 1.4, "Connected basin"),
                                       (upstream, "#66ff66", 1.4, "Upstream dam catchment")]:
        x, y = geom.exterior.xy; ax.plot(x, y, color=color, linewidth=width, label=label)
    x, y = flow_path.xy; ax.plot(x, y, color="#ffffff", linewidth=1.8, label="Dam-to-outlet flow path")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Indicative population per 100 m cell (2020)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_title("Ayakchi CIEWS — preliminary settlement exposure context")
    ax.set_xlabel("Easting (m, UTM zone 42N)"); ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(OUT / "map_02_buffer_population_context.png", dpi=300); plt.close(fig)


def plot_worldcover(path, connected, buffer_geom, upstream, flow_path):
    with rasterio.open(path) as src:
        raw = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    codes = list(WC_CLASSES); display = np.full(raw.shape, np.nan, dtype="float32")
    raw_values = raw.filled(-9999)
    for index, code in enumerate(codes):
        display[raw_values == code] = index
    display = np.ma.masked_invalid(display)
    cmap = ListedColormap(WC_COLORS); norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(display, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    for geom, color, width, label in [(buffer_geom, "#111111", 1.2, "5 km context buffer"),
                                       (connected, "#ffffff", 1.4, "Connected basin"),
                                       (upstream, "#00ffff", 1.5, "Upstream dam catchment")]:
        x, y = geom.exterior.xy; ax.plot(x, y, color=color, linewidth=width, label=label)
    x, y = flow_path.xy; ax.plot(x, y, color="#08306b", linewidth=2, label="Dam-to-outlet flow path")
    cbar = fig.colorbar(image, ax=ax, ticks=range(len(codes)), fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels(list(WC_CLASSES.values())); cbar.set_label("ESA WorldCover class")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_title("Ayakchi CIEWS — benchmark land cover in the 5 km contextual buffer")
    ax.set_xlabel("Easting (m, UTM zone 42N)"); ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(OUT / "map_03_worldcover_benchmark.png", dpi=300); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    connected = first_geometry(SOURCE).buffer(0)
    upstream = first_geometry(UPSTREAM).buffer(0)
    flow_path = first_geometry(FLOW_PATH)
    buffer_geom = connected.buffer(BUFFER_M)
    buffer_ring = buffer_geom.difference(connected)
    downstream_incremental = connected.difference(upstream)
    river_corridor = flow_path.buffer(RIVER_CORRIDOR_M).intersection(buffer_geom)
    zones = {"upstream_catchment": upstream, "downstream_incremental": downstream_incremental,
             "connected_basin": connected, "five_km_buffer_ring": buffer_ring,
             "two_km_river_corridor": river_corridor, "total_study_area": buffer_geom}
    write_geojson(OUT / "study_area_5km_buffer.geojson", buffer_geom,
                  {"buffer_distance_m": BUFFER_M, "purpose": "CIEWS context; not a flood boundary"})
    write_geojson(OUT / "buffer_ring_5km.geojson", buffer_ring,
                  {"buffer_distance_m": BUFFER_M, "excludes_connected_basin": True})
    write_geojson(OUT / "river_corridor_2km.geojson", river_corridor,
                  {"corridor_distance_m": RIVER_CORRIDOR_M, "purpose": "exposure screening; not inundation"})

    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    # Simplification is far below the thematic raster resolution and keeps the
    # direct Earth Engine request below geometry/payload limits.
    region_wgs84 = transform_geometry(to_wgs84.transform, buffer_geom.simplify(30))
    ee.Initialize(project="ee-sabitovty"); ee.Number(1).getInfo()
    region = ee.Geometry(mapping(region_wgs84))
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
          .filterDate("2025-04-01", "2025-11-01").select("label").mode().rename("dw_class").toByte())
    wc = ee.ImageCollection("ESA/WorldCover/v200").mosaic().select("Map").toByte()
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").toFloat()
    population = (ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2021-01-01")
                  .filter(ee.Filter.eq("country", "UZB")).mosaic().rename("population").toFloat())
    dw_path = download(dw, "01_dynamic_world_2025.tif", region, 20)
    wc_path = download(wc, "02_esa_worldcover_2021.tif", region, 20)
    built_path = download(built, "03_ghsl_built_surface_2020.tif", region, 100)
    population_path = download(population, "04_worldpop_population_2020.tif", region, 100)

    land_rows = landcover_stats(dw_path, zones)
    worldcover_rows = worldcover_stats(wc_path, zones)
    exposure_rows = exposure_stats(built_path, population_path, zones)
    plot_landcover(dw_path, connected, buffer_geom, upstream, flow_path)
    plot_exposure(population_path, connected, buffer_geom, upstream, flow_path)
    plot_worldcover(wc_path, connected, buffer_geom, upstream, flow_path)
    metadata = {"created": date.today().isoformat(), "crs": CRS, "buffer_distance_m": BUFFER_M,
                "river_corridor_distance_m": RIVER_CORRIDOR_M,
                "study_area_km2": round(buffer_geom.area / 1e6, 3),
                "connected_basin_km2": round(connected.area / 1e6, 3),
                "sources": {"land_cover": "GOOGLE/DYNAMICWORLD/V1 (Apr-Oct 2025 mode)",
                            "benchmark_land_cover": "ESA/WorldCover/v200 (2021)",
                            "built_surface": "JRC/GHSL/P2023A/GHS_BUILT_S/2020",
                            "population": "WorldPop/GP/100m/pop (UZB, 2020)"},
                "limitations": ["The 5 km buffer is contextual and is not a flood or dam-break boundary.",
                                "Population and built-surface layers are preliminary screening datasets.",
                                "Land-cover classifications require local validation.",
                                "Dynamic World over-classifies built land relative to ESA WorldCover and GHSL in this dry landscape; use ESA WorldCover as the benchmark."]}
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Ayakchi 5 km buffer-zone screening\n\n"
        "Land-cover, built-surface and indicative population screening for the connected basin, its 5 km "
        "context buffer, and a 2 km corridor around the dam-to-outlet flow path. The buffer and corridor "
        "support CIEWS stakeholder and exposure discussions; neither is a flood-inundation or dam-break boundary.\n\n"
        "`worldcover_statistics.csv` is the primary land-cover benchmark; `landcover_statistics.csv` is a "
        "recent Dynamic World comparison. Dynamic World identifies much more built terrain than ESA WorldCover "
        "and GHSL in this dry landscape, so its built class must not be used as an urban-area estimate without "
        "local validation. `exposure_statistics.csv` summarizes GHSL built surface and indicative population.\n",
        encoding="utf-8")
    print(f"Complete: {len(land_rows)} Dynamic World rows, {len(worldcover_rows)} WorldCover rows, "
          f"{len(exposure_rows)} exposure rows")


if __name__ == "__main__":
    main()
