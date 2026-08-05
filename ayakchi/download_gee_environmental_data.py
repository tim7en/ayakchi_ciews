"""Download open environmental data from Google Earth Engine for Ayakchi CIEWS.

The script uses the local basin boundary, initializes Earth Engine with a
Google Cloud project, and writes clipped GeoTIFFs, basin-average CSV time
series, and a provenance manifest. It never stores OAuth credentials locally.

Example:
    python download_gee_environmental_data.py --project ee-sabitovty
"""

import argparse
import csv
import json
import shutil
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import ee
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry


ROOT = Path(__file__).resolve().parent
BOUNDARY = ROOT / "hydromorphology_outputs" / "basin_boundary.geojson"
OUT = ROOT / "gee_environmental_data"
TARGET_CRS = "EPSG:32642"

CATALOG = {
    "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    "worldcover": "ESA/WorldCover/v200",
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "surface_water": "JRC/GSW1_4/GlobalSurfaceWater",
    "chirps": "UCSB-CHG/CHIRPS/DAILY",
    "era5_land": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "modis_snow": "MODIS/061/MOD10A1",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ee-sabitovty", help="Google Cloud project ID")
    parser.add_argument("--recent-start", default="2021-01-01")
    parser.add_argument("--recent-end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--baseline-start", default="1991-01-01")
    parser.add_argument("--baseline-end", default="2021-01-01", help="Exclusive end date")
    parser.add_argument("--skip-rasters", action="store_true")
    parser.add_argument("--skip-timeseries", action="store_true")
    return parser.parse_args()


def load_region():
    data = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    polygon_utm = shape(data["features"][0]["geometry"])
    transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    polygon_wgs84 = transform_geometry(transformer.transform, polygon_utm)
    return ee.Geometry(mapping(polygon_wgs84)), polygon_wgs84.bounds


def download_image(image, name, region, scale, manifest, description, source, units):
    destination = OUT / "rasters" / f"{name}.tif"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file": str(destination.relative_to(OUT)), "source_asset": source,
        "description": description, "units": units, "nominal_scale_m": scale,
    }
    if destination.exists() and destination.stat().st_size > 0:
        manifest.append(record)
        print(f"Using existing {destination.name}")
        return
    params = {
        "name": name,
        "region": region,
        "scale": scale,
        "crs": TARGET_CRS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = image.clip(region).getDownloadURL(params)
    temporary = destination.with_suffix(".download")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    # Earth Engine may return a TIFF directly or a ZIP despite the format flag.
    if zipfile.is_zipfile(temporary):
        with zipfile.ZipFile(temporary) as archive:
            tif_names = [item for item in archive.namelist() if item.lower().endswith((".tif", ".tiff"))]
            if len(tif_names) != 1:
                raise RuntimeError(f"Expected one TIFF for {name}; received {tif_names}")
            with archive.open(tif_names[0]) as source_file, destination.open("wb") as target:
                shutil.copyfileobj(source_file, target)
        temporary.unlink()
    else:
        temporary.replace(destination)
    manifest.append(record)
    print(f"Downloaded {destination.name}")


def mask_sentinel2(image):
    scl = image.select("SCL")
    clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
             .And(scl.neq(11)).And(scl.neq(1)))
    return image.updateMask(clear).divide(10000).copyProperties(image, ["system:time_start"])


def raster_products(region, args, manifest):
    recent_start, recent_end = args.recent_start, args.recent_end

    # A single complete growing season is more interpretable and substantially
    # lighter than a multi-year optical composite at 10 m.
    optical_start, optical_end = "2025-04-01", "2025-11-01"

    dw = (ee.ImageCollection(CATALOG["dynamic_world"]).filterBounds(region)
          .filterDate(optical_start, optical_end))
    download_image(dw.select("label").mode().rename("dw_class"),
                   "01_dynamic_world_mode_growing_season_2025", region, 10, manifest,
                   "Modal 9-class land cover for April–October 2025", CATALOG["dynamic_world"],
                   "class codes 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, "
                   "5 shrub/scrub, 6 built, 7 bare, 8 snow/ice")

    worldcover = ee.ImageCollection(CATALOG["worldcover"]).mosaic().select("Map")
    download_image(worldcover, "02_esa_worldcover_2021", region, 10, manifest,
                   "Independent 2021 land-cover benchmark", CATALOG["worldcover"], "ESA class code")

    s2 = (ee.ImageCollection(CATALOG["sentinel2"]).filterBounds(region)
          .filterDate(optical_start, optical_end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
          .map(mask_sentinel2).median())
    indices = ee.Image.cat([
        s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        s2.normalizedDifference(["B8", "B11"]).rename("NDMI"),
        s2.normalizedDifference(["B8", "B12"]).rename("NBR"),
        s2.normalizedDifference(["B3", "B11"]).rename("MNDWI"),
    ]).toFloat()
    download_image(indices, "03_sentinel2_environmental_indices_growing_season_2025", region, 20, manifest,
                   "April–October 2025 cloud-masked median NDVI, NDMI, NBR and MNDWI",
                   CATALOG["sentinel2"], "unitless")

    water = (ee.Image(CATALOG["surface_water"])
             .select(["occurrence", "seasonality", "transition"])
             .unmask(0, sameFootprint=False).toInt16())
    download_image(water, "04_jrc_surface_water_history", region, 30, manifest,
                   "Long-term surface-water occurrence, seasonality and transition", CATALOG["surface_water"],
                   "percent; months; transition class")

    chirps = ee.ImageCollection(CATALOG["chirps"]).filterDate(args.baseline_start, args.baseline_end)
    years = ee.List.sequence(int(args.baseline_start[:4]), int(args.baseline_end[:4]) - 1)
    annual = ee.ImageCollection.fromImages(years.map(
        lambda year: chirps.filter(ee.Filter.calendarRange(year, year, "year")).sum()))
    download_image(annual.mean().rename("annual_precipitation_mean_mm"),
                   "05_chirps_annual_precipitation_1991_2020", region, 5566, manifest,
                   "Mean annual precipitation for the WMO 1991–2020 normal period", CATALOG["chirps"], "mm/year")

    monthly_images = []
    for month in range(1, 13):
        def sum_month(year):
            return (chirps.filter(ee.Filter.calendarRange(year, year, "year"))
                    .filter(ee.Filter.calendarRange(month, month, "month")).sum())
        month_years = ee.ImageCollection.fromImages(years.map(sum_month))
        monthly_images.append(month_years.mean().rename(f"precip_month_{month:02d}_mm"))
    download_image(ee.Image.cat(monthly_images).toFloat(), "06_chirps_monthly_climatology_1991_2020",
                   region, 5566, manifest, "Mean monthly precipitation climatology", CATALOG["chirps"], "mm/month")

    era = ee.ImageCollection(CATALOG["era5_land"]).filterDate(recent_start, recent_end)
    era_summary = ee.Image.cat([
        era.select("temperature_2m").mean().subtract(273.15).rename("temperature_mean_c"),
        era.select("temperature_2m_max").mean().subtract(273.15).rename("daily_tmax_mean_c"),
        era.select("temperature_2m_min").mean().subtract(273.15).rename("daily_tmin_mean_c"),
        era.select("volumetric_soil_water_layer_1").mean().rename("soil_water_0_7cm"),
        era.select("snow_cover").mean().rename("snow_cover_mean_pct"),
        era.select("surface_runoff_sum").sum().multiply(1000).divide(5).rename("annual_surface_runoff_mean_mm"),
    ]).toFloat()
    download_image(era_summary, "07_era5_land_recent_climate_2021_2025", region, 11132, manifest,
                   "Recent temperature, surface soil water, snow cover and runoff summary",
                   CATALOG["era5_land"], "mixed; see band names")

    snow = (ee.ImageCollection(CATALOG["modis_snow"]).filterBounds(region)
            .filterDate("2001-01-01", recent_end).select("NDSI_Snow_Cover")
            .map(lambda image: image.gte(10).And(image.lte(100)).rename("snow")))
    download_image(snow.mean().multiply(100).rename("snow_observation_frequency_pct"),
                   "08_modis_snow_frequency_2001_2025", region, 500, manifest,
                   "Fraction of valid daily MODIS observations classified as snow (NDSI snow cover >=10%)",
                   CATALOG["modis_snow"], "percent")


def feature_rows(collection, region, scale, selectors):
    def summarize(image):
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale,
            bestEffort=True, maxPixels=1_000_000)
        return ee.Feature(None, values).set("date", image.date().format("YYYY-MM-dd"))
    features = collection.map(summarize).getInfo()["features"]
    for feature in features:
        properties = feature["properties"]
        yield [properties.get("date")] + [properties.get(name) for name in selectors]


def download_daily_csv(collection, region, scale, fields, path, start_year, end_year):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path.name}")
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target); writer.writerow(["date"] + list(fields))
        for year in range(start_year, end_year):
            annual = collection.filterDate(f"{year}-01-01", f"{year+1}-01-01").sort("system:time_start")
            writer.writerows(feature_rows(annual, region, scale, fields))
            print(f"Downloaded {path.name}: {year}")


def time_series(region, args, manifest):
    start_year, end_year = int(args.recent_start[:4]), int(args.recent_end[:4])
    chirps = ee.ImageCollection(CATALOG["chirps"]).filterBounds(region).select("precipitation")
    chirps_path = OUT / "timeseries" / "chirps_daily_basin_mean_2021_2025.csv"
    download_daily_csv(chirps, region, 5566, ["precipitation"], chirps_path, start_year, end_year)
    manifest.append({"file": str(chirps_path.relative_to(OUT)), "source_asset": CATALOG["chirps"],
                     "description": "Daily basin-average precipitation", "units": "mm/day", "nominal_scale_m": 5566})

    fields = ["temperature_2m", "temperature_2m_min", "temperature_2m_max",
              "volumetric_soil_water_layer_1", "snow_cover", "total_precipitation_sum",
              "surface_runoff_sum"]
    era = ee.ImageCollection(CATALOG["era5_land"]).filterBounds(region).select(fields)
    # Convert units before regional reduction: K to C and metres to millimetres.
    def convert(image):
        return ee.Image.cat([
            image.select("temperature_2m").subtract(273.15),
            image.select("temperature_2m_min").subtract(273.15),
            image.select("temperature_2m_max").subtract(273.15),
            image.select("volumetric_soil_water_layer_1"), image.select("snow_cover"),
            image.select("total_precipitation_sum").multiply(1000),
            image.select("surface_runoff_sum").multiply(1000),
        ]).rename(fields).copyProperties(image, ["system:time_start"])
    era_path = OUT / "timeseries" / "era5_land_daily_basin_mean_2021_2025.csv"
    download_daily_csv(era.map(convert), region, 11132, fields, era_path, start_year, end_year)
    manifest.append({"file": str(era_path.relative_to(OUT)), "source_asset": CATALOG["era5_land"],
                     "description": "Daily basin-average climate and land-surface variables",
                     "units": "C; m3/m3; fraction; mm/day", "nominal_scale_m": 11132})


def main():
    args = arguments()
    ee.Initialize(project=args.project)
    # A server round trip catches expired credentials or a disabled project early.
    ee.Number(1).getInfo()
    region, bounds = load_region()
    OUT.mkdir(exist_ok=True)
    manifest = []
    if not args.skip_rasters:
        raster_products(region, args, manifest)
    if not args.skip_timeseries:
        time_series(region, args, manifest)
    metadata = {
        "created_utc": date.today().isoformat(), "earth_engine_project": args.project,
        "aoi_source": str(BOUNDARY.relative_to(ROOT)), "aoi_wgs84_bounds": bounds,
        "recent_period": [args.recent_start, args.recent_end],
        "climate_normal_period": [args.baseline_start, args.baseline_end],
        "catalog": CATALOG, "outputs": manifest,
    }
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Ayakchi Google Earth Engine environmental data\n\n"
        "Open environmental inputs downloaded for the Climate Information and Early Warning System. "
        "See `manifest.json` for dataset IDs, periods, resolution, units and provenance. Climate rasters "
        "retain their native information content even when exported in UTM; CHIRPS (~5.6 km) and ERA5-Land "
        "(~11.1 km) must not be interpreted as field-scale spatial data. Daily CSVs are basin-average series.\n\n"
        "Reproduce with `python download_gee_environmental_data.py --project ee-sabitovty`.\n",
        encoding="utf-8")
    print(f"Complete: {len(manifest)} products; manifest at {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
