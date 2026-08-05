# Ayakchi basin morphology maps

This package is derived from `Ayakchi_DEM.tif` (SRTM). All analysis rasters are
in **WGS 84 / UTM zone 42N (EPSG:32642)** at 30 m resolution so horizontal and
vertical units are compatible for terrain calculations.

## Outputs

- `01_dem_utm42n_30m.tif`: elevation in metres
- `02_slope_degrees.tif`: slope angle in degrees
- `03_aspect_degrees.tif`: downslope direction, clockwise from north
- `04_hillshade.tif`: shaded-relief visualization (315° azimuth, 45° altitude)
- `05_local_relief_1km.tif`: maximum minus minimum elevation in a ~1 km window
- `06_tpi_1km.tif`: elevation relative to the ~1 km neighbourhood mean
- `07_elevation_zones.tif`: 200 m elevation bands beginning at 800 m
- `map_*.png`: 300 dpi map layouts for reports and presentations
- `morphology_summary.txt`: compact statistics and provenance

NoData is `-9999` for floating-point products and `255` for hillshade. The
source DEM's valid-data footprint is used as the basin mask.

## Reproduce

From the `ayakchi` folder, run:

```powershell
python .\create_morphology_maps.py
```

The script requires rasterio, NumPy, SciPy, and Matplotlib. It recreates the
contents of this directory without altering the source DEM.

## Interpretation notes

Slope, local relief, and TPI are useful screening layers for erosion,
landslides, runoff response, and station siting, but they are not hazard maps
on their own. Hydrologic flow direction, stream extraction, and watershed
metrics should follow DEM sink treatment and validation of the basin outlet.
