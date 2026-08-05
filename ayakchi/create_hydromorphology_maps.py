"""Phase 2 hydromorphology for the Ayakchi basin.

Uses the verified Ayakchi Dam (Option -5) KMZ point as the outlet. The DEM is
conditioned with an outlet-seeded priority flood, producing a connected D8
drainage tree without requiring external hydrology packages.
"""

import csv
import heapq
import json
import math
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import LineString, Point, shape, mapping
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
MORPH = ROOT / "morphology_outputs"
DEM_PATH = MORPH / "01_dem_utm42n_30m.tif"
KMZ_PATH = ROOT / "Ayakchi Dam Project.kmz"
OUT = ROOT / "hydromorphology_outputs"
NODATA = -9999.0
STREAM_THRESHOLD_KM2 = 1.0
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]


def find_outlet():
    with ZipFile(KMZ_PATH) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    for placemark in root.findall(".//k:Placemark", ns):
        if placemark.findtext("k:name", default="", namespaces=ns).strip() == "Ayakchi Dam (Option -5)":
            text = placemark.findtext(".//k:Point/k:coordinates", namespaces=ns)
            lon, lat, *_ = map(float, text.strip().split(","))
            return lon, lat
    raise RuntimeError("Ayakchi Dam (Option -5) was not found in the KMZ")


def snap_outlet(dem, transform, crs, lon, lat):
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    row, col = rasterio.transform.rowcol(transform, x, y)
    valid = np.isfinite(dem)
    rr, cc = np.nonzero(valid)
    cx, cy = rasterio.transform.xy(transform, rr, cc, offset="center")
    distances = np.hypot(np.asarray(cx) - x, np.asarray(cy) - y)
    chosen = int(np.argmin(distances))
    return (int(rr[chosen]), int(cc[chosen])), (x, y), float(distances[chosen])


def priority_flood_tree(dem, outlet):
    """Condition DEM and assign every valid cell a receiver leading to outlet."""
    rows, cols = dem.shape
    valid = np.isfinite(dem)
    conditioned = np.full_like(dem, np.nan, dtype="float64")
    receiver = np.full((rows, cols), -1, dtype="int64")
    visited = np.zeros((rows, cols), dtype=bool)
    orow, ocol = outlet
    oid = orow * cols + ocol
    conditioned[orow, ocol] = dem[orow, ocol]
    visited[orow, ocol] = True
    heap = [(conditioned[orow, ocol], oid)]
    epsilon = 1e-5
    while heap:
        z, idx = heapq.heappop(heap)
        row, col = divmod(idx, cols)
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not valid[nr, nc] or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            nid = nr * cols + nc
            conditioned[nr, nc] = max(float(dem[nr, nc]), z + epsilon)
            receiver[nr, nc] = idx
            heapq.heappush(heap, (conditioned[nr, nc], nid))
    receiver[orow, ocol] = -1
    if not np.all(visited[valid]):
        raise RuntimeError("Some valid DEM cells could not be routed to the outlet")
    return conditioned, receiver


def flow_accumulation(conditioned, receiver):
    valid_ids = np.flatnonzero(np.isfinite(conditioned.ravel()))
    order = valid_ids[np.argsort(conditioned.ravel()[valid_ids])[::-1]]
    accumulation = np.zeros(conditioned.size, dtype="float64")
    accumulation[valid_ids] = 1.0
    for idx in order:
        rec = receiver.ravel()[idx]
        if rec >= 0:
            accumulation[rec] += accumulation[idx]
    return accumulation.reshape(conditioned.shape), order


def strahler_order(stream, receiver, descending_ids):
    flat_stream = stream.ravel()
    flat_receiver = receiver.ravel()
    order = np.zeros(stream.size, dtype="uint8")
    max_incoming = np.zeros(stream.size, dtype="uint8")
    max_count = np.zeros(stream.size, dtype="uint8")
    for idx in descending_ids:
        if not flat_stream[idx]:
            continue
        current = max(1, int(max_incoming[idx]) + (1 if max_count[idx] >= 2 else 0))
        order[idx] = current
        rec = flat_receiver[idx]
        if rec >= 0 and flat_stream[rec]:
            if current > max_incoming[rec]:
                max_incoming[rec], max_count[rec] = current, 1
            elif current == max_incoming[rec]:
                max_count[rec] += 1
    return order.reshape(stream.shape)


def mainstem(receiver, accumulation, outlet, transform):
    rows, cols = accumulation.shape
    donors = [[] for _ in range(accumulation.size)]
    for idx, rec in enumerate(receiver.ravel()):
        if rec >= 0:
            donors[int(rec)].append(idx)
    path = [outlet[0] * cols + outlet[1]]
    while donors[path[-1]]:
        path.append(max(donors[path[-1]], key=lambda idx: accumulation.ravel()[idx]))
    cells = [divmod(idx, cols) for idx in path]  # outlet to headwater
    xy = [rasterio.transform.xy(transform, r, c, offset="center") for r, c in cells]
    distance = [0.0]
    for a, b in zip(xy, xy[1:]):
        distance.append(distance[-1] + math.dist(a, b))
    return cells, xy, np.asarray(distance)


def tributary_subbasins(receiver, accumulation, main_cells, shape, cell_area):
    """Label the largest disjoint tributaries entering the main flow path."""
    cols = shape[1]
    main_ids = {r * cols + c for r, c in main_cells}
    candidates = []
    for idx, rec in enumerate(receiver.ravel()):
        if rec in main_ids and idx not in main_ids:
            area = accumulation.ravel()[idx] * cell_area / 1e6
            if area >= 2.0:
                candidates.append((area, idx))
    selected = [idx for _, idx in sorted(candidates, reverse=True)[:8]]
    branch_label = {idx: label for label, idx in enumerate(selected, start=2)}
    labels = np.zeros(receiver.size, dtype="uint8")
    labels[list(main_ids)] = 1
    for idx in np.flatnonzero(receiver.ravel() >= -1):
        if not np.isfinite(accumulation.ravel()[idx]) or accumulation.ravel()[idx] == 0:
            continue
        trail, current = [], int(idx)
        while labels[current] == 0 and current not in branch_label:
            trail.append(current)
            rec = int(receiver.ravel()[current])
            if rec < 0:
                break
            current = rec
        label = branch_label.get(current, int(labels[current]) or 1)
        for item in trail:
            labels[item] = label
        if current in branch_label:
            labels[current] = label
    labels[accumulation.ravel() == 0] = 0
    return labels.reshape(shape), selected


def write_raster(path, data, profile, dtype="float32", nodata=NODATA):
    meta = profile.copy()
    meta.update(dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    out = np.where(np.isfinite(data), data, nodata).astype(dtype)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(out, 1)


def extent(profile):
    t = profile["transform"]
    return [t.c, t.c + profile["width"] * t.a,
            t.f + profile["height"] * t.e, t.f]


def map_base(ax, profile, title):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.annotate("N", xy=(0.94, .93), xytext=(.94, .83), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=2, headwidth=8),
                ha="center", fontweight="bold")
    ax.text(.01, .01, "WGS 84 / UTM zone 42N | 30 m SRTM",
            transform=ax.transAxes, fontsize=7, color=".25")


def save_network_map(dem, accumulation, order, outlet_xy, profile):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(dem, extent=extent(profile), cmap="terrain", alpha=.78)
    palette = ["#74add1", "#4575b4", "#313695", "#542788", "#2d004b"]
    max_order = max(1, int(order.max()))
    for stream_order in range(1, max_order + 1):
        layer = np.where(order == stream_order, 1.0, np.nan)
        ax.imshow(layer, extent=extent(profile), cmap=ListedColormap([palette[min(stream_order-1, 4)]]),
                  interpolation="nearest", alpha=.95)
    ax.scatter(*outlet_xy, marker="*", s=130, color="#d7191c", edgecolor="white",
               linewidth=.8, label="Ayakchi Dam outlet", zorder=5)
    ax.legend(loc="upper left", frameon=True)
    map_base(ax, profile, f"Ayakchi Basin — Drainage Network (≥ {STREAM_THRESHOLD_KM2:g} km²)")
    fig.savefig(OUT / "map_07_drainage_network.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_accumulation_map(dem, accumulation_km2, outlet_xy, profile):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    shown = np.where(np.isfinite(dem), np.log10(np.maximum(accumulation_km2, .001)), np.nan)
    image = ax.imshow(shown, extent=extent(profile), cmap="viridis")
    cb = fig.colorbar(image, ax=ax, shrink=.76)
    cb.set_label("log10 contributing area (km²)")
    ax.scatter(*outlet_xy, marker="*", s=120, color="red", edgecolor="white", zorder=4)
    map_base(ax, profile, "Ayakchi Basin — Flow Accumulation")
    fig.savefig(OUT / "map_08_flow_accumulation.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_subbasin_map(labels, dem, outlet_xy, profile):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(dem, extent=extent(profile), cmap="gray", alpha=.25)
    shown = np.where(np.isfinite(dem), labels, np.nan)
    cmap = ListedColormap(["#d9d9d9", "#1b9e77", "#d95f02", "#7570b3",
                           "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4"])
    boundaries = np.arange(.5, max(2, labels.max()) + 1.5)
    image = ax.imshow(shown, extent=extent(profile), cmap=cmap,
                      norm=BoundaryNorm(boundaries, cmap.N), alpha=.78)
    cb = fig.colorbar(image, ax=ax, shrink=.76, ticks=np.arange(1, labels.max()+1))
    cb.set_label("Sub-basin ID (1 = main corridor)")
    ax.scatter(*outlet_xy, marker="*", s=120, color="red", edgecolor="white", zorder=4)
    map_base(ax, profile, "Ayakchi Basin — Major Tributary Sub-basins")
    fig.savefig(OUT / "map_09_subbasins.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_hypsometry(dem):
    values = np.sort(dem[np.isfinite(dem)])[::-1]
    relative_height = (values - values.min()) / (values.max() - values.min())
    relative_area = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(relative_area, relative_height, color="#7f3b08", lw=2)
    ax.fill_between(relative_area, relative_height, color="#fdb863", alpha=.35)
    ax.set(xlabel="Relative area above elevation", ylabel="Relative elevation",
           title="Ayakchi Basin — Hypsometric Curve", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25)
    fig.savefig(OUT / "chart_01_hypsometric_curve.png", dpi=300, facecolor="white")
    plt.close(fig)
    with (OUT / "hypsometric_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["relative_area", "relative_elevation", "elevation_m"])
        step = max(1, len(values) // 1000)
        writer.writerows(zip(relative_area[::step], relative_height[::step], values[::step]))


def basin_polygon(mask, transform):
    polygons = [shape(geom) for geom, value in shapes(mask.astype("uint8"), mask=mask, transform=transform) if value == 1]
    return unary_union(polygons)


def main():
    OUT.mkdir(exist_ok=True)
    lon, lat = find_outlet()
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype("float64")
        profile, transform, crs = src.profile, src.transform, src.crs
    outlet, supplied_xy, snap_distance = snap_outlet(dem, transform, crs, lon, lat)
    outlet_xy = rasterio.transform.xy(transform, *outlet, offset="center")
    conditioned, receiver = priority_flood_tree(dem, outlet)
    accumulation, descending = flow_accumulation(conditioned, receiver)
    cell_area = abs(transform.a * transform.e)
    accumulation_km2 = accumulation * cell_area / 1e6
    stream = accumulation_km2 >= STREAM_THRESHOLD_KM2
    order = strahler_order(stream, receiver, descending)
    cells, main_xy, distance = mainstem(receiver, accumulation, outlet, transform)
    cell_area = abs(transform.a * transform.e)
    subbasins, branch_roots = tributary_subbasins(
        receiver, accumulation, cells, dem.shape, cell_area)

    write_raster(OUT / "08_conditioned_dem.tif", conditioned, profile)
    write_raster(OUT / "09_flow_accumulation_cells.tif", np.where(np.isfinite(dem), accumulation, np.nan), profile)
    write_raster(OUT / "10_contributing_area_km2.tif", np.where(np.isfinite(dem), accumulation_km2, np.nan), profile)
    write_raster(OUT / "11_stream_mask_1km2.tif", np.where(np.isfinite(dem), stream, np.nan), profile, "uint8", 255)
    write_raster(OUT / "12_strahler_order.tif", np.where(stream, order, np.nan), profile, "uint8", 255)
    write_raster(OUT / "13_major_subbasins.tif", np.where(np.isfinite(dem), subbasins, np.nan), profile, "uint8", 255)

    polygon = basin_polygon(np.isfinite(dem), transform)
    stream_segments = []
    stream_length = 0.0
    rows, cols = dem.shape
    for idx in np.flatnonzero(stream.ravel()):
        rec = receiver.ravel()[idx]
        if rec >= 0 and stream.ravel()[rec]:
            r1, c1 = divmod(int(idx), cols); r2, c2 = divmod(int(rec), cols)
            line = LineString([rasterio.transform.xy(transform, r1, c1, offset="center"),
                               rasterio.transform.xy(transform, r2, c2, offset="center")])
            stream_length += line.length
            stream_segments.append({"type": "Feature", "properties": {"strahler": int(order.ravel()[idx])},
                                    "geometry": mapping(line)})
    geojson = {"type": "FeatureCollection", "name": "Ayakchi_streams",
               "crs": {"type": "name", "properties": {"name": str(crs)}}, "features": stream_segments}
    (OUT / "drainage_network.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    outlet_feature = {"type": "FeatureCollection", "crs": geojson["crs"], "features": [{
        "type": "Feature", "properties": {"name": "Ayakchi Dam (Option -5)", "source_lon": lon,
        "source_lat": lat, "snap_distance_m": snap_distance}, "geometry": mapping(Point(outlet_xy))}]}
    (OUT / "outlet.geojson").write_text(json.dumps(outlet_feature), encoding="utf-8")
    basin_feature = {"type": "FeatureCollection", "crs": geojson["crs"], "features": [{
        "type": "Feature", "properties": {"name": "Ayakchi basin DEM footprint"}, "geometry": mapping(polygon)}]}
    (OUT / "basin_boundary.geojson").write_text(json.dumps(basin_feature), encoding="utf-8")
    main_feature = {"type": "FeatureCollection", "crs": geojson["crs"], "features": [{
        "type": "Feature", "properties": {"name": "Main flow path"}, "geometry": mapping(LineString(main_xy))}]}
    (OUT / "main_flow_path.geojson").write_text(json.dumps(main_feature), encoding="utf-8")
    sub_features = []
    with (OUT / "subbasin_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["subbasin_id", "type", "area_km2", "mean_elevation_m"])
        for label in range(1, int(subbasins.max()) + 1):
            mask = subbasins == label
            subpoly = basin_polygon(mask, transform)
            kind = "main_corridor" if label == 1 else "major_tributary"
            writer.writerow([label, kind, mask.sum()*cell_area/1e6, float(np.nanmean(dem[mask]))])
            sub_features.append({"type": "Feature", "properties": {"subbasin_id": label, "type": kind},
                                 "geometry": mapping(subpoly)})
    (OUT / "major_subbasins.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "crs": geojson["crs"], "features": sub_features}), encoding="utf-8")

    elevations = np.array([dem[r, c] for r, c in cells])
    with (OUT / "main_channel_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["distance_from_outlet_km", "elevation_m", "easting", "northing"])
        writer.writerows((d/1000, z, x, y) for d, z, (x, y) in zip(distance, elevations, main_xy))
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(distance / 1000, elevations, color="#2166ac", lw=1.8)
    ax.set(xlabel="Distance upstream from outlet (km)", ylabel="Elevation (m)",
           title="Ayakchi Basin — Main Flow Path Profile")
    ax.grid(alpha=.25)
    fig.savefig(OUT / "chart_02_main_channel_profile.png", dpi=300, facecolor="white"); plt.close(fig)

    save_network_map(dem, accumulation, order, outlet_xy, profile)
    save_accumulation_map(dem, accumulation_km2, outlet_xy, profile)
    save_subbasin_map(subbasins, dem, outlet_xy, profile)
    save_hypsometry(dem)

    area_km2 = polygon.area / 1e6
    perimeter_km = polygon.length / 1000
    length_km = max(Point(outlet_xy).distance(Point(xy)) for xy in
                    (rasterio.transform.xy(transform, r, c, offset="center") for r, c in zip(*np.nonzero(np.isfinite(dem))))) / 1000
    relief = float(np.nanmax(dem) - np.nanmin(dem))
    metrics = {
        "area_km2": area_km2, "perimeter_km": perimeter_km, "basin_length_km": length_km,
        "elevation_min_m": float(np.nanmin(dem)), "elevation_mean_m": float(np.nanmean(dem)),
        "elevation_max_m": float(np.nanmax(dem)), "total_relief_m": relief,
        "hypsometric_integral": float((np.nanmean(dem)-np.nanmin(dem))/relief),
        "form_factor": area_km2 / length_km**2,
        "elongation_ratio": 2 * math.sqrt(area_km2/math.pi) / length_km,
        "circularity_ratio": 4 * math.pi * area_km2 / perimeter_km**2,
        "relief_ratio": (relief/1000) / length_km,
        "main_flow_path_km": float(distance[-1]/1000),
        "stream_threshold_km2": STREAM_THRESHOLD_KM2,
        "stream_length_km": stream_length/1000,
        "drainage_density_km_per_km2": (stream_length/1000)/area_km2,
        "maximum_strahler_order": int(order.max()),
        "outlet_lon": lon, "outlet_lat": lat, "outlet_snap_distance_m": snap_distance,
        "outlet_easting": outlet_xy[0], "outlet_northing": outlet_xy[1],
        "maximum_dem_fill_m": float(np.nanmax(conditioned-dem)),
    }
    with (OUT / "basin_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["metric", "value"]); writer.writerows(metrics.items())
    (OUT / "README.md").write_text(
        "# Ayakchi hydromorphology — phase 2\n\n"
        "The verified `Ayakchi Dam (Option -5)` point in the supplied KMZ is used as the outlet. "
        "All valid DEM cells are routed to it using an outlet-seeded priority-flood D8 tree. "
        "The conditioned DEM should be treated as an analytical routing surface, not observed terrain.\n\n"
        "The drainage network uses a 1 km² contributing-area threshold. GeoTIFFs are GIS-ready; "
        "GeoJSON files contain the outlet, basin boundary, main flow path, and stream segments. "
        "CSVs contain the basin metrics, hypsometric curve, and longitudinal profile.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
