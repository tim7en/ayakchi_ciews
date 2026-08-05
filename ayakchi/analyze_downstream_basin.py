"""Analyze the connected Ayakchi basin from headwaters, through dam, to city."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import unary_union

from create_hydromorphology_maps import priority_flood_tree, flow_accumulation, strahler_order


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Ayakchi_downstream_basin.tif"
OUT = ROOT / "downstream_analysis_outputs"
CRS = "EPSG:32642"
RESOLUTION = 30.0
NODATA = -9999.0
DAM_LONLAT = (66.78400725556058, 39.3004695565556)


def prepare_dem():
    with rasterio.open(SOURCE) as src:
        transform, width, height = calculate_default_transform(
            src.crs, CRS, src.width, src.height, *src.bounds, resolution=RESOLUTION)
        dem = np.full((height, width), NODATA, dtype="float32")
        reproject(rasterio.band(src, 1), dem, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=transform, dst_crs=CRS,
                  dst_nodata=NODATA, resampling=Resampling.bilinear)
    dem[dem == NODATA] = np.nan
    profile = {"driver": "GTiff", "width": width, "height": height,
               "count": 1, "crs": CRS, "transform": transform}
    return dem, profile


def boundary_cells(mask):
    inside = np.pad(mask, 1, constant_values=False)
    interior = (inside[1:-1, 1:-1] & inside[:-2, 1:-1] & inside[2:, 1:-1]
                & inside[1:-1, :-2] & inside[1:-1, 2:])
    return mask & ~interior


def nearest_cell(mask, transform, xy):
    rr, cc = np.nonzero(mask)
    xx, yy = rasterio.transform.xy(transform, rr, cc, offset="center")
    distance = np.hypot(np.asarray(xx)-xy[0], np.asarray(yy)-xy[1])
    i = int(np.argmin(distance))
    return (int(rr[i]), int(cc[i])), float(distance[i])


def snap_to_channel(accumulation, transform, xy, radius_m=100):
    rows, cols = accumulation.shape
    row, col = rasterio.transform.rowcol(transform, *xy)
    radius_cells = int(math.ceil(radius_m / RESOLUTION))
    r0, r1 = max(0, row-radius_cells), min(rows, row+radius_cells+1)
    c0, c1 = max(0, col-radius_cells), min(cols, col+radius_cells+1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    xx, yy = rasterio.transform.xy(transform, rr, cc, offset="center")
    distance = np.hypot(np.asarray(xx).reshape(rr.shape)-xy[0],
                        np.asarray(yy).reshape(rr.shape)-xy[1])
    candidates = (distance <= radius_m) & np.isfinite(accumulation[r0:r1, c0:c1])
    score = np.where(candidates, accumulation[r0:r1, c0:c1], -1)
    local = np.unravel_index(np.argmax(score), score.shape)
    snapped = (int(r0+local[0]), int(c0+local[1]))
    return snapped, float(distance[local])


def trace_receivers(start, receiver):
    cols = receiver.shape[1]
    current = start[0]*cols + start[1]
    ids = [current]
    seen = {current}
    while receiver.ravel()[current] >= 0:
        current = int(receiver.ravel()[current])
        if current in seen:
            raise RuntimeError("Receiver loop encountered")
        seen.add(current); ids.append(current)
    return [divmod(idx, cols) for idx in ids]


def path_geometry(cells, transform):
    xy = [rasterio.transform.xy(transform, r, c, offset="center") for r, c in cells]
    distance = [0.0]
    for a, b in zip(xy, xy[1:]):
        distance.append(distance[-1] + math.dist(a, b))
    return xy, np.asarray(distance)


def write_raster(path, data, profile, dtype="float32", nodata=NODATA):
    meta = profile.copy(); meta.update(dtype=dtype, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(np.where(np.isfinite(data), data, nodata).astype(dtype), 1)


def extent(profile):
    t = profile["transform"]
    return [t.c, t.c+profile["width"]*t.a, t.f+profile["height"]*t.e, t.f]


def polygonize(mask, transform):
    return unary_union([shape(geom) for geom, value in shapes(
        mask.astype("uint8"), mask=mask, transform=transform) if value == 1])


def main():
    OUT.mkdir(exist_ok=True)
    dem, profile = prepare_dem(); transform = profile["transform"]
    mask = np.isfinite(dem)
    edge = boundary_cells(mask)
    outlet_flat = np.nanargmin(np.where(edge, dem, np.nan))
    outlet = tuple(map(int, np.unravel_index(outlet_flat, dem.shape)))
    dam_xy_source = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(*DAM_LONLAT)
    dam_initial, _ = nearest_cell(mask, transform, dam_xy_source)

    conditioned, receiver = priority_flood_tree(dem, outlet)
    accumulation, descending = flow_accumulation(conditioned, receiver)
    area_km2 = accumulation * RESOLUTION**2 / 1e6
    dam, dam_snap = snap_to_channel(accumulation, transform, dam_xy_source)
    stream = area_km2 >= 2.0
    order = strahler_order(stream, receiver, descending)
    downstream_cells = trace_receivers(dam, receiver)
    downstream_xy, downstream_distance = path_geometry(downstream_cells, transform)
    outlet_xy = downstream_xy[-1]; dam_xy = downstream_xy[0]

    path_ids = {r*dem.shape[1]+c for r, c in downstream_cells}
    downstream_stream = np.zeros(dem.shape, dtype=bool)
    # Retain stream cells whose receiver chain reaches the dam-to-city path
    # without first passing through the dam (incremental downstream tributaries).
    memo = {}
    for idx in np.flatnonzero(stream.ravel()):
        trail, current = [], int(idx)
        while current not in memo and current not in path_ids and receiver.ravel()[current] >= 0:
            trail.append(current); current = int(receiver.ravel()[current])
        reaches = current in path_ids or memo.get(current, False)
        for item in trail: memo[item] = reaches
        if reaches: downstream_stream.ravel()[idx] = True
    for idx in path_ids: downstream_stream.ravel()[idx] = True

    dz_dy, dz_dx = np.gradient(np.where(mask, dem, np.nanmedian(dem)), RESOLUTION, RESOLUTION)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))); slope[~mask] = np.nan
    from matplotlib.colors import LightSource
    hillshade = LightSource(315, 45).hillshade(np.where(mask, dem, np.nanmedian(dem)),
                                               dx=RESOLUTION, dy=RESOLUTION)*255
    hillshade[~mask] = np.nan

    write_raster(OUT / "01_downstream_dem_utm42n_30m.tif", dem, profile)
    write_raster(OUT / "02_downstream_slope_degrees.tif", slope, profile)
    write_raster(OUT / "03_total_contributing_area_km2.tif", np.where(mask, area_km2, np.nan), profile)
    write_raster(OUT / "04_stream_mask_2km2.tif", np.where(mask, downstream_stream, np.nan), profile, "uint8", 255)
    write_raster(OUT / "05_strahler_order.tif", np.where(downstream_stream, order, np.nan), profile, "uint8", 255)

    basin = polygonize(mask, transform)
    upstream_area = float(area_km2[dam])
    total_area = float(area_km2[outlet])
    incremental_area = total_area-upstream_area
    path = LineString(downstream_xy)
    crs_info = {"type": "name", "properties": {"name": CRS}}
    features = [
        {"type": "Feature", "properties": {"name": "Supplied Ayakchi Dam coordinate"},
         "geometry": mapping(Point(dam_xy_source))},
        {"type": "Feature", "properties": {"name": "Hydrologically snapped Ayakchi Dam", "snap_distance_m": dam_snap},
         "geometry": mapping(Point(dam_xy))},
        {"type": "Feature", "properties": {"name": "City-facing basin outlet"},
         "geometry": mapping(Point(outlet_xy))},
    ]
    (OUT / "control_points.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "crs": crs_info, "features": features}), encoding="utf-8")
    (OUT / "dam_to_city_flow_path.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "crs": crs_info, "features": [{"type": "Feature",
         "properties": {"distance_km": path.length/1000}, "geometry": mapping(path)}]}), encoding="utf-8")
    (OUT / "connected_basin_boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "crs": crs_info, "features": [{"type": "Feature",
         "properties": {"area_km2": basin.area/1e6}, "geometry": mapping(basin)}]}), encoding="utf-8")

    elevations = np.array([dem[r, c] for r, c in downstream_cells])
    with (OUT / "dam_to_city_profile.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target); writer.writerow(["distance_from_dam_km", "elevation_m", "easting", "northing"])
        writer.writerows((d/1000, z, x, y) for d, z, (x, y) in zip(downstream_distance, elevations, downstream_xy))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(downstream_distance/1000, elevations, color="#2166ac", lw=2)
    ax.fill_between(downstream_distance/1000, elevations, elevations.min(), color="#92c5de", alpha=.3)
    ax.set(xlabel="Distance downstream from dam (km)", ylabel="Elevation (m)",
           title="Ayakchi — Dam-to-City-Facing Outlet Profile")
    ax.grid(alpha=.25); fig.savefig(OUT / "chart_01_dam_to_city_profile.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.imshow(dem, extent=extent(profile), cmap="terrain", alpha=.72)
    ax.imshow(hillshade, extent=extent(profile), cmap="gray", alpha=.23)
    stream_layer = np.where(downstream_stream, order, np.nan)
    ax.imshow(stream_layer, extent=extent(profile), cmap="Blues", alpha=.9, interpolation="nearest")
    ax.plot(*zip(*downstream_xy), color="#00ffff", lw=2.2, label="Dam-to-city flow path")
    ax.scatter(*dam_xy, marker="*", s=140, color="#d7191c", edgecolor="white", label="Ayakchi Dam", zorder=5)
    ax.scatter(*outlet_xy, marker="v", s=90, color="#762a83", edgecolor="white", label="City-facing outlet", zorder=5)
    ax.legend(loc="upper left"); ax.set_title("Ayakchi Connected Basin — Dam-to-City Drainage System", fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    fig.savefig(OUT / "map_01_connected_drainage_system.png", dpi=300, facecolor="white"); plt.close(fig)

    # Distance reaches along the downstream channel for warning communication.
    reach_colors = ["#1a9850", "#fee08b", "#fc8d59", "#d73027"]
    reach_labels = ["0–2 km", "2–5 km", "5–10 km", ">10 km"]
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.imshow(hillshade, extent=extent(profile), cmap="gray", alpha=.35)
    for lo, hi, color, label_name in zip([0, 2, 5, 10], [2, 5, 10, np.inf], reach_colors, reach_labels):
        chosen = [(x, y) for (x, y), d in zip(downstream_xy, downstream_distance/1000) if lo <= d < hi]
        if len(chosen) > 1: ax.plot(*zip(*chosen), color=color, lw=4, label=label_name)
    ax.scatter(*dam_xy, marker="*", s=140, color="black", zorder=5)
    ax.legend(title="River distance below dam"); ax.set_title("Ayakchi — Downstream Warning Reaches", fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    fig.savefig(OUT / "map_02_downstream_warning_reaches.png", dpi=300, facecolor="white"); plt.close(fig)

    metrics = {
        "connected_basin_area_km2": total_area,
        "contributing_area_at_dam_km2": upstream_area,
        "incremental_area_below_dam_km2": incremental_area,
        "incremental_area_percent": incremental_area/total_area*100,
        "dam_to_city_outlet_path_km": float(downstream_distance[-1]/1000),
        "dam_elevation_m": float(dem[dam]), "outlet_elevation_m": float(dem[outlet]),
        "path_elevation_drop_m": float(dem[dam]-dem[outlet]),
        "average_path_gradient_percent": float((dem[dam]-dem[outlet])/downstream_distance[-1]*100),
        "dam_coordinate_snap_m": dam_snap,
        "stream_extraction_threshold_km2": 2.0,
    }
    (OUT / "downstream_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Ayakchi downstream analysis\n\n"
        "Connected 30 m DEM analysis from the headwaters through Ayakchi Dam to the city-facing basin outlet. "
        "Outputs include terrain, contributing area, drainage, the dam-to-outlet profile, and river-distance warning reaches.\n\n"
        "Warning reaches are distance bands, not hydraulic travel times. Converting them to lead times and flood levels "
        "requires channel geometry, roughness, discharge observations and release scenarios.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
