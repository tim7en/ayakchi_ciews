"""Create a first-pass morphology map package for the Ayakchi basin DEM.

Outputs GIS-ready GeoTIFFs and presentation PNGs in ``morphology_outputs``.
The workflow uses only rasterio/numpy/scipy/matplotlib and is repeatable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import generic_filter, uniform_filter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Ayakchi_DEM.tif"
OUT = ROOT / "morphology_outputs"
TARGET_CRS = "EPSG:32642"  # WGS 84 / UTM zone 42N; metres
TARGET_RESOLUTION = 30.0
NODATA = -9999.0


def write_raster(path, data, profile, *, dtype="float32", nodata=NODATA):
    output = np.where(np.isfinite(data), data, nodata).astype(dtype)
    meta = profile.copy()
    meta.update(dtype=dtype, nodata=nodata, count=1, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(output, 1)


def reproject_dem():
    with rasterio.open(SOURCE) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RESOLUTION,
        )
        dem = np.full((height, width), NODATA, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
    dem[dem == NODATA] = np.nan
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "crs": TARGET_CRS,
        "transform": transform,
    }
    return dem, profile


def terrain_derivatives(dem, resolution):
    # Fill NoData locally only for stable gradients, then restore the basin mask.
    valid = np.isfinite(dem)
    filled = dem.copy()
    nearest_seed = np.nanmedian(dem)
    filled[~valid] = nearest_seed
    dz_dy, dz_dx = np.gradient(filled, resolution, resolution)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = (90.0 - np.degrees(np.arctan2(-dz_dy, dz_dx))) % 360.0
    aspect[slope < 0.1] = np.nan

    # Topographic position relative to a 1,010 m square neighbourhood.
    weights = valid.astype("float32")
    local_sum = uniform_filter(np.where(valid, dem, 0), size=33, mode="nearest")
    local_weight = uniform_filter(weights, size=33, mode="nearest")
    local_mean = np.divide(
        local_sum, local_weight, out=np.full_like(dem, np.nan), where=local_weight > 0
    )
    tpi = dem - local_mean

    # Local relief (max minus min) over the same neighbourhood.
    relief = generic_filter(dem, np.nanmax, size=33, mode="nearest") - generic_filter(
        dem, np.nanmin, size=33, mode="nearest"
    )
    for array in (slope, aspect, tpi, relief):
        array[~valid] = np.nan
    return slope, aspect, tpi, relief


def map_extent(profile):
    t = profile["transform"]
    return [
        t.c,
        t.c + profile["width"] * t.a,
        t.f + profile["height"] * t.e,
        t.f,
    ]


def add_map_furniture(ax, profile):
    extent = map_extent(profile)
    width_m = extent[1] - extent[0]
    # Use a compact, readable scale bar close to one fifth of map width.
    target_km = width_m / 5000
    bar_km = min((1, 2, 5, 10, 20), key=lambda value: abs(value - target_km))
    x0 = extent[0] + width_m * 0.07
    y0 = extent[2] + (extent[3] - extent[2]) * 0.07
    ax.plot([x0, x0 + bar_km * 1000], [y0, y0], color="black", lw=3)
    ax.text(x0 + bar_km * 500, y0 + (extent[3] - extent[2]) * 0.015,
            f"{bar_km} km", ha="center", fontsize=8)
    ax.annotate("N", xy=(0.94, 0.93), xytext=(0.94, 0.82), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=2, headwidth=8), ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)


def save_map(filename, data, profile, title, cmap, label, *, norm=None, overlay=None):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = ax.imshow(data, extent=map_extent(profile), cmap=cmap, norm=norm)
    if overlay is not None:
        ax.imshow(overlay, extent=map_extent(profile), cmap="gray", alpha=0.25)
    cbar = fig.colorbar(image, ax=ax, shrink=0.76, pad=0.025)
    cbar.set_label(label)
    ax.set_title(title, fontsize=14, fontweight="bold")
    add_map_furniture(ax, profile)
    ax.text(0.01, 0.01, "CRS: WGS 84 / UTM zone 42N | Source: SRTM DEM",
            transform=ax.transAxes, fontsize=7, color="0.25")
    fig.savefig(OUT / filename, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    dem, profile = reproject_dem()
    slope, aspect, tpi, relief = terrain_derivatives(dem, TARGET_RESOLUTION)

    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = np.full_like(dem, np.nan)
    valid = np.isfinite(dem)
    shaded = ls.hillshade(np.where(valid, dem, np.nanmedian(dem)),
                          vert_exag=1, dx=TARGET_RESOLUTION, dy=TARGET_RESOLUTION)
    hillshade[valid] = shaded[valid] * 255

    elev_breaks = np.arange(800, 2201, 200)
    elevation_zone = np.digitize(dem, elev_breaks).astype("float32")
    elevation_zone[~valid] = np.nan

    products = {
        "01_dem_utm42n_30m.tif": dem,
        "02_slope_degrees.tif": slope,
        "03_aspect_degrees.tif": aspect,
        "04_hillshade.tif": hillshade,
        "05_local_relief_1km.tif": relief,
        "06_tpi_1km.tif": tpi,
        "07_elevation_zones.tif": elevation_zone,
    }
    for name, data in products.items():
        dtype = "uint8" if "hillshade" in name else "float32"
        write_raster(OUT / name, data, profile, dtype=dtype,
                     nodata=255 if dtype == "uint8" else NODATA)

    save_map("map_01_elevation.png", dem, profile, "Ayakchi Basin — Elevation",
             "terrain", "Elevation (m)", overlay=hillshade)
    save_map("map_02_slope.png", slope, profile, "Ayakchi Basin — Slope",
             "YlOrRd", "Slope (degrees)")
    aspect_colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60",
                     "#1a9850", "#74add1", "#4575b4", "#f46d43"]
    aspect_norm = BoundaryNorm(np.arange(0, 361, 45), len(aspect_colors))
    save_map("map_03_aspect.png", aspect, profile, "Ayakchi Basin — Aspect",
             ListedColormap(aspect_colors), "Aspect (degrees from north)", norm=aspect_norm)
    save_map("map_04_hillshade.png", hillshade, profile, "Ayakchi Basin — Hillshade",
             "gray", "Illumination (0–255)")
    save_map("map_05_local_relief.png", relief, profile,
             "Ayakchi Basin — Local Relief (1 km)", "magma", "Relief (m)")
    save_map("map_06_tpi.png", tpi, profile,
             "Ayakchi Basin — Topographic Position Index (1 km)", "RdBu_r",
             "Elevation relative to neighbourhood (m)")

    # Compact summary for QA and later reporting.
    with (OUT / "morphology_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Ayakchi basin morphology summary\n")
        f.write("Source: Ayakchi_DEM.tif (SRTM)\n")
        f.write(f"Analysis CRS: {TARGET_CRS}; resolution: {TARGET_RESOLUTION:.0f} m\n")
        f.write(f"Valid DEM cells: {valid.sum():,}\n")
        f.write(f"Approximate raster-covered area: {valid.sum()*TARGET_RESOLUTION**2/1e6:.2f} km2\n")
        for label, array, unit in [
            ("Elevation", dem, "m"), ("Slope", slope, "degrees"),
            ("Local relief", relief, "m"), ("TPI", tpi, "m")]:
            v = array[np.isfinite(array)]
            f.write(f"{label}: min {np.min(v):.2f}, mean {np.mean(v):.2f}, "
                    f"max {np.max(v):.2f} {unit}\n")


if __name__ == "__main__":
    main()
