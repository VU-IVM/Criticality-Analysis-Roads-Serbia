"""
Shared functions for 4a / 4b / 4c hazard-exposure analysis.

Notebooks call these with hardcoded paths; scripts call them via NetworkConfig
attributes.  All saved vector outputs are reprojected to EPSG:6316 and written
to both Parquet and an ArcGIS File Geodatabase.
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

import contextily as cx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr
from affine import Affine
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio import features
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterstats import zonal_stats
from scipy.ndimage import zoom
from shapely.geometry import mapping


def _show_or_close(show_figures: bool) -> None:
    """Display the current figure when *show_figures* is True, else close it.

    The caller always saves the figure to disk beforehand; this only controls
    the interactive pop-up. The 4a/4b/4c scripts pass ``config.show_figures``
    (the single toggle in ``src/config/network_config.py``) so a batch run can
    suppress every figure; the notebooks rely on each plotting function's
    default ``show_figures=True`` so plots always render inline.
    """
    if show_figures:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Generic save helper (mirrors _save_vector in accessibility_functions.py)
# ---------------------------------------------------------------------------

# Column names the OpenFileGDB driver reserves and tries to use as the feature
# ID. If the source data carries one of these (the Deonice roads ship an
# ``OBJECTID`` that is duplicated once the network is split into directed
# edges) GDAL writes it as the explicit FID and fails with
# "Cannot create feature of ID N because one already exists".
_GDB_RESERVED_COLS = {"objectid", "fid", "oid"}


def _sanitize_for_gdb(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reset index and rename FID-reserved columns so GDAL auto-assigns FIDs."""
    gdf = gdf.reset_index(drop=True)
    renames = {c: f"{c}_orig" for c in gdf.columns if c.lower() in _GDB_RESERVED_COLS}
    if renames:
        gdf = gdf.rename(columns=renames)
    return gdf


def intermediate_excel_path(parquet_path: Path) -> Path:
    """Mirror a parquet path into the ``excel`` tree as an ``.xlsx`` path.

    ``intermediate_results/parquet/<sub>/<name>.parquet`` becomes
    ``intermediate_results/excel/<sub>/<name>.xlsx``. Paths written directly to
    the ``intermediate_results`` root are mirrored to ``excel/<name>.xlsx``.
    """
    parquet_path = Path(parquet_path)
    parts = list(parquet_path.parts)
    if "parquet" in parts:
        parts[parts.index("parquet")] = "excel"
    elif "intermediate_results" in parts:
        parts.insert(parts.index("intermediate_results") + 1, "excel")
    else:
        parts.insert(len(parts) - 1, "excel")
    return Path(*parts).with_suffix(".xlsx")


def save_excel_mirror(gdf, parquet_path: Path) -> None:
    """Write the attribute table of *gdf* as an Excel file mirroring *parquet_path*.

    The geometry column is dropped. Used so every intermediate vector output has
    a companion ``.xlsx`` under ``intermediate_results/excel/`` (see
    :func:`intermediate_excel_path`).
    """
    excel_path = intermediate_excel_path(parquet_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    data = gdf
    geom_name = getattr(gdf, "geometry", None)
    if geom_name is not None and gdf.geometry.name in gdf.columns:
        data = gdf.drop(columns=gdf.geometry.name)
    data.to_excel(excel_path, index=False)
    print(f"Saved Excel -> {excel_path}")


def save_hazard_vector(
    gdf: gpd.GeoDataFrame,
    parquet_dir: Path,
    gdb_path: Path,
    layer_name: str,
    output_crs: str = "EPSG:6316",
) -> None:
    """Save *gdf* to ``parquet_dir/<layer_name>.parquet``, the ``gdb_path`` layer,
    and a mirrored Excel attribute table under ``intermediate_results/excel/``.

    If *layer_name* already exists in the GDB (e.g. from a prior run) the GDB
    is rebuilt — preserving all other layers — so there are no FID conflicts.
    """
    gdf_out = gdf.to_crs(output_crs)

    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{layer_name}.parquet"
    gdf_out.to_parquet(parquet_path)
    print(f"Saved {layer_name} -> {parquet_path}")

    save_gdb_layer(gdf, gdb_path, layer_name, output_crs)
    save_excel_mirror(gdf_out, parquet_path)


def save_gdb_layer(
    gdf: gpd.GeoDataFrame,
    gdb_path: Path,
    layer_name: str,
    output_crs: str = "EPSG:6316",
) -> None:
    """Write *gdf* (reprojected to *output_crs*) as ``layer_name`` in *gdb_path*.

    If *layer_name* already exists in the GDB (e.g. from a prior run) the GDB
    is rebuilt — preserving all other layers — so there are no FID conflicts.
    """
    import shutil
    import pyogrio

    gdf_out = gdf.to_crs(output_crs)

    gdb_path = Path(gdb_path)
    gdb_path.parent.mkdir(parents=True, exist_ok=True)

    # If the layer already exists, preserve all other layers, delete the GDB,
    # then re-write everything. Avoids FID conflicts without needing osgeo.
    preserved: dict[str, gpd.GeoDataFrame] = {}
    if gdb_path.exists():
        try:
            for lname in pyogrio.list_layers(str(gdb_path))[:, 0]:
                if lname != layer_name:
                    preserved[lname] = gpd.read_file(str(gdb_path), layer=lname)
        except Exception:
            pass
        shutil.rmtree(gdb_path)

    for lname, ldata in preserved.items():
        _sanitize_for_gdb(ldata).to_file(
            str(gdb_path), driver="OpenFileGDB", layer=lname, promote_to_multi=True
        )

    _sanitize_for_gdb(gdf_out).to_file(
        str(gdb_path), driver="OpenFileGDB", layer=layer_name, promote_to_multi=True
    )

    print(f"Saved {layer_name} -> {gdb_path} (layer '{layer_name}')")


# ---------------------------------------------------------------------------
# 4a — Current-hazard shared functions
# ---------------------------------------------------------------------------

def load_country_boundaries(world_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth world boundaries; return (world, serbia, kosovo)."""
    world = gpd.read_file(world_path)
    serbia = world.loc[world.SOV_A3 == "SRB"]
    kosovo = world.loc[world.SOV_A3 == "KOS"]
    return world, serbia, kosovo


def clip_roads_by_country(
    roads_gdf: gpd.GeoDataFrame,
    serbia_gdf: gpd.GeoDataFrame,
    kosovo_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Clip *roads_gdf* to Serbia and Kosovo separately; return both clipped GDFs."""
    serbia = serbia_gdf.to_crs(roads_gdf.crs)
    kosovo = kosovo_gdf.to_crs(roads_gdf.crs)
    serbia_roads = gpd.clip(roads_gdf, serbia)
    kosovo_roads = gpd.clip(roads_gdf, kosovo)
    return serbia_roads, kosovo_roads


def load_and_clip_flood_raster(
    flood_path: Path,
    country_gdf: gpd.GeoDataFrame,
) -> xr.Dataset:
    """Load European flood raster clipped to *country_gdf* bounding box."""
    minx, miny, maxx, maxy = country_gdf.total_bounds
    hazard = xr.open_dataset(flood_path, engine="rasterio")
    return hazard.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy).load()


def plot_flood_depth_map(
    hazard_country: xr.Dataset,
    figure_path: Path,
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot flood depth raster with CartoDB.Positron basemap and save to *figure_path*."""
    flood_colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    flood_cmap = LinearSegmentedColormap.from_list("flood_blue", flood_colors, N=256)

    if hazard_country.rio.crs is None:
        hazard_country = hazard_country.rio.write_crs("EPSG:4326")
    hazard_mercator = hazard_country.rio.reproject("EPSG:3857")

    fig, ax = plt.subplots(1, 1, figsize=(9, 12), facecolor="white")
    hazard_mercator.band_data.plot(
        ax=ax, cmap=flood_cmap, alpha=0.7, vmin=0, vmax=6, add_colorbar=False, add_labels=False,
    )
    cx.add_basemap(ax=ax, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.4, attribution=False)
    ax.axis("off")

    flood_labels = ["0-1m", "1-2m", "2-3m", "3-4m", "4-5m", "5m+"]
    legend_colors = [flood_cmap(i / 5) for i in range(len(flood_labels))]
    legend_elements = [
        Patch(facecolor=legend_colors[i], label=flood_labels[i], edgecolor="navy", linewidth=0.5, alpha=0.8)
        for i in range(len(flood_labels))
    ]
    ax.legend(
        handles=legend_elements, title="Flood Depth (meters)", loc="lower left",
        fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.88)
    plt.savefig(Path(figure_path) / "flood_depth_map.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_flood_depth_roads(
    roads: gpd.GeoDataFrame,
    figure_path: Path,
    parquet_dir: Path,
    gdb_path: Path,
    arcgis_gpgk: Path | None = None,
    arcgis_results: Path | None = None,
    output_crs: str = "EPSG:6316",
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot roads coloured by flood class; save figure + vector to parquet+GDB."""
    from utils.arcgis import save_lyrx_layer

    roads_plot = roads.to_crs(epsg=3857)
    fig, ax = plt.subplots(1, 1, figsize=(9, 12), facecolor="white")

    roads_plot[roads_plot["flood_class"] == "No flooding"].plot(
        ax=ax, color="#d3d3d3", linewidth=0.3, alpha=0.4, zorder=1
    )
    roads_plot[roads_plot["flood_class"] != "No flooding"].plot(
        ax=ax, color="#2171b5", linewidth=1.5, alpha=0.85, zorder=2
    )

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
    ax.axis("off")

    legend_handles = [
        Patch(facecolor="#d3d3d3", label="No flooding", edgecolor="grey", linewidth=0.5),
        Patch(facecolor="#2171b5", label="Flooded", edgecolor="grey", linewidth=0.5),
    ]
    ax.legend(handles=legend_handles, title="Flood Depth", loc="lower left",
              fontsize=10, title_fontsize=12, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(Path(figure_path) / "flood_depth_roads.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)

    save_hazard_vector(roads, parquet_dir, gdb_path, "flood_depth_roads", output_crs)

    if arcgis_gpgk and arcgis_results:
        roads_out = roads.to_crs(output_crs)
        save_lyrx_layer(
            gdf=roads_out,
            gpkg_path=Path(arcgis_gpgk) / "flood_depth_roads.gpkg",
            lyrx_path=Path(arcgis_results) / "flood_depth_roads.lyrx",
            layer_name="flood_depth_roads",
            labels=["No flooding", "Flooded"],
            colors=["#d3d3d3", "#2171b5"],
            width_mapping={"No flooding": 0.3, "Flooded": 1.5},
            field="flood_class",
            title="Flood Depth on Roads",
        )


def plot_snowdrift_map(
    snow_drift_path: Path,
    serbia_roads_mercator: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot snow-drift segments classified by length; Serbia roads black, Kosovo grey."""
    snow_drift = gpd.read_file(snow_drift_path)
    snow_drift_mercator = snow_drift.to_crs(3857)

    bins = [0, 0.5, 1, 2, 5, float("inf")]
    labels = ["< 0.5 km", "0.5-1 km", "1-2 km", "2-5 km", "> 5 km"]
    snow_drift_mercator["length_class"] = pd.cut(
        snow_drift_mercator["dužina_sn"], bins=bins, labels=labels, include_lowest=True
    )

    colors = {"< 0.5 km": "#deebf7", "0.5-1 km": "#9ecae1", "1-2 km": "#4292c6", "2-5 km": "#2171b5", "> 5 km": "#084594"}
    linewidths = {"< 0.5 km": 1.0, "0.5-1 km": 1.5, "1-2 km": 2.0, "2-5 km": 3.0, "> 5 km": 4.0}

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor="white")
    serbia_roads_mercator.plot(ax=ax, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    kosovo_roads_mercator.plot(ax=ax, color="grey", linewidth=0.8, alpha=0.5, zorder=2)

    for label in labels:
        subset = snow_drift_mercator[snow_drift_mercator["length_class"] == label]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colors[label], linewidth=linewidths[label], alpha=0.9, zorder=3)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6)
    ] + [
        Line2D(
            [0], [0], color=colors[label], linewidth=linewidths[label],
            label=f"{label} (n={len(snow_drift_mercator[snow_drift_mercator['length_class'] == label])})",
        )
        for label in labels
    ]
    ax.legend(
        handles=legend_elements, title="Snow Drift Length", loc="upper right",
        fontsize=12, title_fontsize=14, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "snow_drift_map.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_landslides_map(
    landslide_path: Path,
    serbia_roads_mercator: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot landslides (tip=='Klizište' only) classified by year."""
    landslides = gpd.read_file(landslide_path)
    landslides = landslides[landslides["tip"] == "Klizište"]

    landslides_mercator = landslides.to_crs(3857)
    landslides_mercator["year"] = pd.to_datetime(landslides_mercator["datum_evid"], errors="coerce").dt.year

    bins = [0, 2000, 2010, 2015, 2020, 2030]
    labels = ["< 2000", "2000-2010", "2010-2015", "2015-2020", "2020-2025"]
    landslides_mercator["year_class"] = pd.cut(
        landslides_mercator["year"], bins=bins, labels=labels, include_lowest=True
    )

    colors = {
        "< 2000": "#fde0dd", "2000-2010": "#fa9fb5", "2010-2015": "#f768a1",
        "2015-2020": "#c51b8a", "2020-2025": "#7a0177",
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor="white")
    serbia_roads_mercator.plot(ax=ax, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    kosovo_roads_mercator.plot(ax=ax, color="grey", linewidth=0.8, alpha=0.5, zorder=2)

    for label in labels:
        subset = landslides_mercator[landslides_mercator["year_class"] == label]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colors[label], markersize=40, alpha=0.8, edgecolor="#333333", linewidth=0.3, zorder=3)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=colors[label], markersize=10,
            markeredgecolor="#333333", markeredgewidth=0.3,
            label=f"{label} (n={len(landslides_mercator[landslides_mercator['year_class'] == label])})",
        )
        for label in labels
    ]
    ax.legend(
        handles=legend_elements, title="Year Recorded", loc="upper right",
        fontsize=12, title_fontsize=14, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "landslides_map_by_year.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def assign_wildfire_risk_to_roads(
    raster_path: Path,
    roads_gdf: gpd.GeoDataFrame,
    buffer_meters: int = 10,
) -> gpd.GeoDataFrame:
    """Assign binary wildfire risk (0/1) to each road segment.

    Any raster value > 0 within ``buffer_meters`` of the road → risk=1.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        transform = src.transform
        raster_window = rasterio.windows.Window(0, 0, src.width, src.height)

        roads_reproj = roads_gdf.to_crs(raster_crs)
        wildfire_risk = []

        for geom in roads_reproj.geometry:
            try:
                buffered = geom.buffer(buffer_meters)
                bounds = buffered.bounds
                window = rasterio.windows.from_bounds(
                    bounds[0], bounds[1], bounds[2], bounds[3], transform
                ).round_lengths().round_offsets()

                if not rasterio.windows.intersect(window, raster_window):
                    wildfire_risk.append(0)
                    continue

                window = rasterio.windows.intersection(window, raster_window)
                if window.width < 1 or window.height < 1:
                    wildfire_risk.append(0)
                    continue

                data = src.read(1, window=window)
                win_transform = src.window_transform(window)
                mask = geometry_mask(
                    [mapping(buffered)], out_shape=data.shape,
                    transform=win_transform, invert=True,
                )
                pixels = data[mask & (data != nodata)]
                wildfire_risk.append(1 if np.any(pixels > 0) else 0)
            except Exception:
                wildfire_risk.append(0)

    result = roads_reproj.copy()
    result["wildfire_risk"] = wildfire_risk
    result["wildfire_class"] = result["wildfire_risk"].map({0: "No risk", 1: "Wildfire risk"})
    return result.to_crs(roads_gdf.crs)


def plot_wildfire_raster_map(
    wildfire_path: Path,
    serbia_roads_mercator: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot binary wildfire risk raster (any value > 0 = risk)."""
    with rasterio.open(wildfire_path) as src:
        scale_factor = 10
        out_shape = (1, int(src.height / scale_factor), int(src.width / scale_factor))
        data = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest).astype(float)
        data[data == src.nodata] = np.nan
        fire_bounds = src.bounds
        fire_crs = src.crs
        bounds_mercator = transform_bounds(
            fire_crs, "EPSG:3857",
            fire_bounds.left, fire_bounds.bottom, fire_bounds.right, fire_bounds.top,
        )

    binary_data = np.where(np.isfinite(data), 1, 0)
    color_array = np.zeros((*binary_data.shape, 4))
    color_array[binary_data == 1] = mcolors.to_rgba("#d7191c", alpha=0.75)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor="white")
    ax.imshow(
        color_array,
        extent=[bounds_mercator[0], bounds_mercator[2], bounds_mercator[1], bounds_mercator[3]],
        zorder=2, origin="upper",
    )
    serbia_roads_mercator.plot(ax=ax, color="black", linewidth=0.8, alpha=0.5, zorder=3)
    kosovo_roads_mercator.plot(ax=ax, color="grey", linewidth=0.8, alpha=0.5, zorder=3)
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Wildfire Risk", fontsize=16, fontweight="bold", pad=12)

    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6),
        Patch(facecolor="#d7191c", edgecolor="none", label="Wildfire risk"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12, title_fontsize=14,
              frameon=True, fancybox=True, shadow=True, framealpha=0.9, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "binary_wildfire_risk_map.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_wildfire_roads_AB(
    wildfire_path: Path,
    roads_with_risk: gpd.GeoDataFrame,
    serbia_roads_mercator: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    figure_path: Path,
    parquet_dir: Path,
    gdb_path: Path,
    arcgis_gpgk: Path | None = None,
    arcgis_results: Path | None = None,
    output_crs: str = "EPSG:6316",
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Two-panel A/B figure: wildfire raster (A) and roads exposed to wildfire (B)."""
    from utils.arcgis import save_lyrx_layer

    # --- load raster for panel A ---
    with rasterio.open(wildfire_path) as src:
        scale_factor = 10
        out_shape = (1, int(src.height / scale_factor), int(src.width / scale_factor))
        data = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest).astype(float)
        data[data == src.nodata] = np.nan
        fire_bounds = src.bounds
        fire_crs = src.crs
        bounds_mercator = transform_bounds(
            fire_crs, "EPSG:3857",
            fire_bounds.left, fire_bounds.bottom, fire_bounds.right, fire_bounds.top,
        )

    binary_data = np.where(np.isfinite(data), 1, 0)
    color_array = np.zeros((*binary_data.shape, 4))
    color_array[binary_data == 1] = mcolors.to_rgba("#d7191c", alpha=0.75)

    roads_mercator = roads_with_risk.to_crs(3857)

    # Common extent (Serbia + Kosovo road network) so both panels align tightly
    bounds_3857 = gpd.GeoSeries(
        pd.concat([serbia_roads_mercator.geometry, kosovo_roads_mercator.geometry]), crs=3857
    ).total_bounds
    figsize = _grid_figsize(bounds_3857, n_rows=1, n_cols=2, panel_height=9.0)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02)

    # Panel A — raster
    ax_a = axes[0]
    ax_a.imshow(
        color_array,
        extent=[bounds_mercator[0], bounds_mercator[2], bounds_mercator[1], bounds_mercator[3]],
        zorder=2, origin="upper",
    )
    serbia_roads_mercator.plot(ax=ax_a, color="black", linewidth=0.8, alpha=0.5, zorder=3)
    kosovo_roads_mercator.plot(ax=ax_a, color="grey", linewidth=0.8, alpha=0.5, zorder=3)

    # Panel B — roads
    ax_b = axes[1]
    serbia_roads_mercator.plot(ax=ax_b, color="black", linewidth=0.8, alpha=0.5, zorder=2)
    kosovo_roads_mercator.plot(ax=ax_b, color="grey", linewidth=0.8, alpha=0.5, zorder=2)
    exposed = roads_mercator[roads_mercator["wildfire_risk"] == 1]
    if not exposed.empty:
        exposed.plot(ax=ax_b, color="#d7191c", linewidth=1.0, alpha=0.85, zorder=3)

    _set_common_extent(axes, bounds_3857)
    for ax in axes:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")

    ax_a.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6),
            Patch(facecolor="#d7191c", edgecolor="none", label="Wildfire susceptibility"),
        ],
        loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )
    ax_b.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6),
            Patch(facecolor="#d7191c", edgecolor="none", label="Roads running through\nwildfire susceptible areas"),
        ],
        loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    for ax, label in zip(axes, ["A", "B"]):
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=16, fontweight="bold",
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.savefig(Path(figure_path) / "wildfire_risk_AB.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)

    save_hazard_vector(roads_with_risk, parquet_dir, gdb_path, "wildfire_risk_roads", output_crs)

    if arcgis_gpgk and arcgis_results:
        roads_out = roads_with_risk.to_crs(output_crs)
        save_lyrx_layer(
            gdf=roads_out,
            gpkg_path=Path(arcgis_gpgk) / "wildfire_risk_roads.gpkg",
            lyrx_path=Path(arcgis_results) / "wildfire_risk_roads.lyrx",
            layer_name="wildfire_risk_roads",
            labels=["No risk", "Wildfire risk"],
            colors=["#000000", "#d7191c"],
            width_mapping={"No risk": 0.4, "Wildfire risk": 1.0},
            field="wildfire_class",
            title="Wildfire Risk on Roads",
        )


def plot_landslide_susceptibility_map(
    landslide_susceptibility_path: Path,
    serbia_roads_mercator: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 600,
    show_figures: bool = True,
) -> None:
    """Plot landslide susceptibility raster (discrete classes 2/4/6/8/10)."""
    with rasterio.open(landslide_susceptibility_path) as src:
        landslide_data = src.read(1).astype(float)
        landslide_data[landslide_data == src.nodata] = np.nan
        landslide_bounds = src.bounds
        landslide_crs = src.crs
        bounds_mercator = transform_bounds(
            landslide_crs, "EPSG:3857",
            landslide_bounds.left, landslide_bounds.bottom,
            landslide_bounds.right, landslide_bounds.top,
        )

    susc_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    susc_colors = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
    value_to_color = {
        2: mcolors.to_rgba("#1a9641", alpha=0.75),
        4: mcolors.to_rgba("#a6d96a", alpha=0.75),
        6: mcolors.to_rgba("#ffffbf", alpha=0.75),
        8: mcolors.to_rgba("#fdae61", alpha=0.75),
        10: mcolors.to_rgba("#d7191c", alpha=0.75),
    }

    color_array = np.full((*landslide_data.shape, 4), np.nan)
    for val, rgba in value_to_color.items():
        color_array[landslide_data == val] = rgba
    color_array[np.isnan(landslide_data)] = (0, 0, 0, 0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor="white")
    ax.imshow(
        color_array,
        extent=[bounds_mercator[0], bounds_mercator[2], bounds_mercator[1], bounds_mercator[3]],
        zorder=2, origin="upper",
    )
    serbia_roads_mercator.plot(ax=ax, color="black", linewidth=0.8, alpha=0.5, zorder=3)
    kosovo_roads_mercator.plot(ax=ax, color="grey", linewidth=0.8, alpha=0.5, zorder=3)
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6)
    ] + [
        Patch(facecolor=susc_colors[i], edgecolor="none", label=susc_labels[i])
        for i in range(len(susc_labels))
    ]
    ax.legend(
        handles=legend_elements, title="Landslide Susceptibility", loc="upper right",
        fontsize=12, title_fontsize=14, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "landslide_susceptibility_map.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


# ---------------------------------------------------------------------------
# 4b — Future flood & precipitation shared functions
# ---------------------------------------------------------------------------

def max_raster_along_line(line, raster: np.ndarray, transform: Affine) -> float:
    """Return the maximum raster value intersecting *line*."""
    mask = geometry_mask([mapping(line)], transform=transform, invert=True, out_shape=raster.shape)
    values = raster[mask]
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


def calculate_future_flood_return_periods(
    ds: xr.Dataset,
    basins_3035: gpd.GeoDataFrame,
    roads_3035: gpd.GeoDataFrame,
    scenarios: list[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Compute mean new return periods per basin and join to roads.

    Returns (basins_3035_with_rp, roads_rp) — both in original CRS (EPSG:3035).
    """
    for scenario in scenarios:
        rp = ds[f"baseline_rp_shift_{scenario}"]
        rp = rp.transpose("y", "x").sortby("y")
        rp = rp.rio.write_crs("EPSG:3035", inplace=False)
        rp = rp.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

        mean_vals = []
        for _, basin in basins_3035.iterrows():
            masked = rp.rio.clip([basin.geometry], basins_3035.crs, drop=False)
            vals = masked.values
            vals = vals[~np.isnan(vals)]
            mean_vals.append(vals.mean() if vals.size else np.nan)
        basins_3035[f"rp{scenario}_mean"] = mean_vals

    bins = [10, 25, 50, 100, 150, np.inf]
    bin_labels = ["10-25", "25-50", "50-100", "100-150", "150+"]
    for scenario in scenarios:
        basins_3035[f"rp{scenario}_bin"] = pd.cut(
            basins_3035[f"rp{scenario}_mean"], bins=bins, labels=bin_labels, include_lowest=True
        )

    roads_rp = roads_3035.copy()
    roads_rp = gpd.sjoin(
        roads_rp,
        basins_3035[["geometry"] + [f"rp{s}_mean" for s in scenarios]],
        how="left", predicate="intersects",
    )
    roads_rp = roads_rp.drop(columns=[c for c in ["index_right", "index_left"] if c in roads_rp.columns])

    for scenario in scenarios:
        roads_rp[f"rp{scenario}_bin"] = pd.cut(
            roads_rp[f"rp{scenario}_mean"], bins=bins, labels=bin_labels, include_lowest=True
        )

    return basins_3035, roads_rp


def plot_future_flood_basins(
    basins_3035: gpd.GeoDataFrame,
    scenarios: list[str],
    scenario_labels: dict[str, str],
    figure_path: Path,
    parquet_dir: Path,
    gdb_path: Path,
    output_crs: str = "EPSG:6316",
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Plot 2×2 basin return-period maps and save results."""
    bins = [10, 25, 50, 100, 150, np.inf]
    bin_labels = ["10-25", "25-50", "50-100", "100-150", "150+"]
    colors = ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf"]
    color_dict = {**dict(zip(bin_labels, colors)), "No data": "white"}

    basins_3857 = basins_3035.to_crs(epsg=3857)
    bounds_3857 = basins_3857.total_bounds

    figw, figh = _grid_figsize(bounds_3857, n_rows=2, n_cols=2, panel_height=6.5)
    fig, axes = plt.subplots(2, 2, figsize=(figw, figh + 1.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.07, wspace=0.02, hspace=0.06)
    axes_flat = axes.flatten()

    for i, scenario in enumerate(scenarios):
        col_mean = f"rp{scenario}_mean"
        basins_3857[f"rp{scenario}_bin"] = pd.cut(
            basins_3857[col_mean], bins=bins, labels=bin_labels, include_lowest=True
        )
        plot_col = basins_3857[f"rp{scenario}_bin"].astype(str)
        plot_col[plot_col == "nan"] = "No data"
        basins_3857.plot(color=plot_col.map(color_dict), edgecolor="black", linewidth=0.5, ax=axes_flat[i])
        axes_flat[i].set_title(f"Future warming scenario: {scenario_labels[scenario]}°C", fontsize=14)

    _set_common_extent(axes, bounds_3857)
    for ax in axes_flat:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=1.0, attribution=False)
        ax.set_axis_off()

    handles = [
        __import__("matplotlib").patches.Patch(facecolor=color_dict[lbl], edgecolor="black", label=lbl)
        for lbl in bin_labels + ["No data"]
    ]
    fig.legend(
        handles=handles,
        title="Future return periods of floods with a current 100 year return period",
        loc="lower center", ncol=len(handles), frameon=True, edgecolor="black", facecolor="white",
    )

    axes_sorted = sorted(axes_flat[:4], key=lambda ax: (-ax.get_position().y0, ax.get_position().x0))
    for i, ax in enumerate(axes_sorted):
        ax.text(0.05, 0.95, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.savefig(Path(figure_path) / "Change in return period.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)

    save_hazard_vector(basins_3857, parquet_dir, gdb_path, "Future Floods change in RP", output_crs)


def plot_future_flood_roads(
    roads_rp_3035: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    scenarios: list[str],
    scenario_labels: dict[str, str],
    figure_path: Path,
    parquet_dir: Path,
    gdb_path: Path,
    output_crs: str = "EPSG:6316",
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Plot 2×2 road return-period maps and save results."""
    bin_labels = ["10-25", "25-50", "50-100", "100-150", "150+"]
    colors = ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf"]
    color_dict = {**dict(zip(bin_labels, colors)), "No data": "white"}

    roads_rp = roads_rp_3035.to_crs(epsg=3857)
    bounds_3857 = gpd.GeoSeries(
        pd.concat([roads_rp.geometry, kosovo_roads_mercator.geometry]), crs=3857
    ).total_bounds

    figw, figh = _grid_figsize(bounds_3857, n_rows=2, n_cols=2, panel_height=6.5)
    fig, axes = plt.subplots(2, 2, figsize=(figw, figh + 1.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.07, wspace=0.02, hspace=0.06)
    axes_flat = axes.flatten()

    for i, scenario in enumerate(scenarios):
        plot_col = roads_rp[f"rp{scenario}_bin"].astype(str)
        plot_col[plot_col == "nan"] = "No data"
        roads_rp.plot(color=plot_col.map(color_dict), edgecolor="black", linewidth=1.2, ax=axes_flat[i])
        kosovo_roads_mercator.plot(ax=axes_flat[i], color="grey", linewidth=0.8, alpha=0.5, zorder=2)
        axes_flat[i].set_title(f"Future warming scenario: {scenario_labels[scenario]}°C", fontsize=14)

    _set_common_extent(axes, bounds_3857)
    for ax in axes_flat:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=1.0, attribution=False)
        ax.set_axis_off()

    handles = [
        __import__("matplotlib").patches.Patch(facecolor=color_dict[lbl], edgecolor="black", label=lbl)
        for lbl in bin_labels
    ]
    fig.legend(
        handles=handles,
        title="Future return periods of floods with a current 100 year return period",
        loc="lower center", ncol=len(handles), frameon=True, edgecolor="black", facecolor="white",
    )

    axes_sorted = sorted(axes_flat[:4], key=lambda ax: (-ax.get_position().y0, ax.get_position().x0))
    for i, ax in enumerate(axes_sorted):
        ax.text(0.05, 0.95, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.savefig(Path(figure_path) / "Change in return period experienced by roads.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)

    save_hazard_vector(roads_rp, parquet_dir, gdb_path, "Future Floods change in RP experienced by roads", output_crs)


def calculate_future_max_precipitation(
    data_path: Path,
    roads_path: Path,
    world_path: Path,
    parquet_dir: Path,
    gdb_path: Path,
    output_crs: str = "EPSG:6316",
) -> dict:
    """Compute max daily precipitation change per road for all rcp/period combinations.

    Saves parquet for each combination to *parquet_dir* and all to *gdb_path*.
    Returns ``results[rcp][period]`` dict with ``roads_with_agreement`` / ``no_agreement``.
    """
    results: dict = {}

    for rcp in ("45", "85"):
        results[rcp] = {}
        for period in ("1", "2"):
            file_name = rf"Climate Change Precipitation\results\rcp{rcp}_rx1d_change{period}.nc"
            file = Path(data_path) / file_name
            file_name_ens = rf"Climate Change Precipitation\results\rcp{rcp}_rx1d_change{period}_ensmed.nc"
            file_ens = Path(data_path) / file_name_ens

            ds = xr.open_dataset(file)
            ensemble_median = xr.open_dataset(file_ens)

            roads = gpd.read_file(roads_path)
            if roads.crs != "EPSG:4326":
                roads = roads.to_crs("EPSG:4326")

            # Serbia boundary only (no Kosovo) for masking
            world = gpd.read_file(world_path)
            serbia = world[world.SOV_A3 == "SRB"]

            ds = ds.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            ensemble_median = ensemble_median.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

            ensemble_median = ensemble_median.rename({
                "Change_ensmed_T20": "Change_T20",
                "Change_ensmed_T50": "Change_T50",
                "Change_ensmed_T100": "Change_T100",
                "Change_ensmed_T500": "Change_T500",
                "Change_ensmed_T1000": "Change_T1000",
            })

            var = "Change_T20"
            data_var = ds[var]
            ensemble = ensemble_median[var]

            n_pos = (data_var > 0).sum(dim="model")
            n_neg = (data_var < 0).sum(dim="model")
            agreement = xr.zeros_like(n_pos)
            for count, sign in [(8, -4), (7, -3), (6, -2), (5, -1)]:
                agreement = xr.where(n_neg == count, sign, agreement)
            for count, sign in [(5, 1), (6, 2), (7, 3), (8, 4)]:
                agreement = xr.where(n_pos == count, sign, agreement)

            lon = agreement.longitude.values
            lat = agreement.latitude.values
            transform = Affine.translation(lon[0], lat[0]) * Affine.scale(lon[1] - lon[0], lat[1] - lat[0])

            # Rasterize Serbia mask
            mask_arr = features.rasterize(
                [(geom, 1) for geom in serbia.geometry],
                out_shape=(len(lat), len(lon)),
                transform=transform, fill=0, dtype="uint8",
            )
            mask_da = xr.DataArray(mask_arr, coords={"latitude": lat, "longitude": lon}, dims=("latitude", "longitude"))
            ensemble_masked = ensemble.where(agreement != 0).where(mask_da == 1)
            ensemble_masked_pct = ensemble_masked * 100

            raster_values = ensemble_masked_pct.values
            lon2 = ensemble_masked_pct.longitude.values
            lat2 = ensemble_masked_pct.latitude.values
            transform2 = Affine.translation(lon2[0], lat2[0]) * Affine.scale(lon2[1] - lon2[0], lat2[1] - lat2[0])

            roads["max_rx1day_pct"] = roads.geometry.apply(
                lambda geom: max_raster_along_line(geom, raster_values, transform2)
            )

            roads_with_agreement = roads[roads["max_rx1day_pct"].notna()]
            roads_no_agreement = roads[roads["max_rx1day_pct"].isna()]

            results[rcp][period] = {
                "roads_with_agreement": roads_with_agreement,
                "no_agreement": roads_no_agreement,
            }

            layer_name = f"precipitation_change_rcp{rcp}_period{period}"
            save_hazard_vector(roads_with_agreement, parquet_dir, gdb_path, layer_name, output_crs)

    return results


def plot_precipitation_change(
    results: dict,
    figure_path: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Plot 2×2 precipitation-change road maps."""
    bins = [-10, -5, -2, 0, 2, 5, 10, 15, 20, 25]
    colors_list = [
        "#08306b", "#2171b5", "#6baed6", "#fcae91",
        "#fb6a4a", "#de2d26", "#a50f15", "#770111", "#360108",
    ]
    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(bins, cmap.N)

    rcps = ["45", "85"]
    periods = ["1", "2"]
    rcps_title = ["4.5", "8.5"]
    periods_title = ["2031 - 2060", "2071 - 2100"]

    # Road geometry is identical across panels → one common extent for all
    sample = results["45"]["1"]
    bounds_3857 = gpd.GeoSeries(
        pd.concat([
            sample["roads_with_agreement"].to_crs(3857).geometry,
            sample["no_agreement"].to_crs(3857).geometry,
        ]), crs=3857,
    ).total_bounds

    figw, figh = _grid_figsize(bounds_3857, n_rows=2, n_cols=2, panel_height=6.0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figw, figh + 0.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.10, wspace=0.02, hspace=0.02)

    for row, period in enumerate(periods):
        for col, rcp in enumerate(rcps):
            ax = axes[row, col]
            results[rcp][period]["no_agreement"].to_crs(epsg=3857).plot(ax=ax, color="grey", linewidth=0.6)
            results[rcp][period]["roads_with_agreement"].to_crs(epsg=3857).plot(
                ax=ax, column="max_rx1day_pct", norm=norm, cmap=cmap, linewidth=1.2, legend=False,
            )

    _set_common_extent(axes, bounds_3857)
    for ax in axes.flat:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
        ax.set_axis_off()

    # Panel labels (A–D) and axis-relative RCP (column) / period (row) headings
    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        ax.text(0.05, 0.95, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=14, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    for col, rcp in enumerate(rcps_title):
        axes[0, col].set_title(f"RCP {rcp}", fontsize=14, fontweight="bold")
    for row, period in enumerate(periods_title):
        axes[row, 0].text(-0.04, 0.5, period, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=12, rotation=90, fontweight="bold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Change (%)")
    cbar.set_ticks(bins)

    plt.savefig(Path(figure_path) / "change in Rx1d.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


# ---------------------------------------------------------------------------
# 4c — Temperature shared functions
# ---------------------------------------------------------------------------

# Pavement temperature bins/colours (shared between plot functions)
ABS_BINS_PAV = [50, 53, 56, 59, 62, 64]
ABS_COLORS_PAV = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

ABS_BINS = [24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
ABS_COLORS = [
    "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#fddbc7",
    "#f4a582", "#d6604d", "#b2182b", "#800026", "#4d0010",
]
DIFF_BINS = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
DIFF_COLORS = [
    "#2166ac", "#92c5de", "#d1e5f0", "#fddbc7",
    "#f4a582", "#d6604d", "#b2182b", "#67000d",
]


def read_tif(path) -> tuple[np.ndarray, Any, Any]:
    """Open a GeoTIFF; return (data_2d, bounds, crs) with nodata → NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, bounds, crs


def resample_to_match(data_src: np.ndarray, data_ref: np.ndarray) -> np.ndarray:
    """Bilinear-resample *data_src* to the same shape as *data_ref*."""
    zoom_r = data_ref.shape[0] / data_src.shape[0]
    zoom_c = data_ref.shape[1] / data_src.shape[1]
    nan_mask = np.isnan(data_src).astype(float)
    resampled = zoom(np.nan_to_num(data_src, nan=0.0), (zoom_r, zoom_c), order=1)
    resampled_mask = zoom(nan_mask, (zoom_r, zoom_c), order=1) > 0.5
    resampled[resampled_mask] = np.nan
    return resampled


def _reproject_bounds_to_3857(bounds, crs) -> tuple[float, float, float, float]:
    """Convert raster bounds from native CRS to Web Mercator (EPSG:3857)."""
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    left, bottom = transformer.transform(bounds.left, bounds.bottom)
    right, top = transformer.transform(bounds.right, bounds.top)
    return left, bottom, right, top


def _apply_panel_style(ax, title: str, serbia_3857: gpd.GeoDataFrame) -> None:
    """White-background panel with Serbia outline and CartoDB.Positron basemap."""
    serbia_3857.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.5, zorder=4)
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.35, attribution=False, zoom="auto")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#222222")


def mask_to_serbia(
    data: np.ndarray,
    bounds,
    crs,
    serbia_gdf: gpd.GeoDataFrame,
    all_touched: bool = False,
) -> np.ndarray:
    """Return *data* with pixels outside Serbia set to NaN.

    With ``all_touched=True`` any pixel overlapping Serbia even partly is kept
    (otherwise only pixels whose centre falls inside Serbia are kept).
    """
    from rasterio.transform import from_bounds as _from_bounds

    serbia_proj = serbia_gdf.to_crs(crs)
    rows, cols = data.shape
    transform = _from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, cols, rows)
    outside = geometry_mask(
        [mapping(geom) for geom in serbia_proj.geometry],
        out_shape=(rows, cols), transform=transform, invert=False,
        all_touched=all_touched,
    )
    masked = data.copy()
    masked[outside] = np.nan
    return masked


# ---------------------------------------------------------------------------
# Multi-panel layout helpers — eliminate empty space between equal-aspect maps
# ---------------------------------------------------------------------------

def _grid_figsize(bounds_3857, n_rows: int, n_cols: int, panel_height: float = 6.0) -> tuple[float, float]:
    """Figure size whose per-panel aspect matches the map extent (Web Mercator).

    Sizing the figure to the data aspect lets each equal-aspect map fill its
    axes box, so adjacent panels touch instead of floating in whitespace.
    """
    minx, miny, maxx, maxy = bounds_3857
    dx, dy = (maxx - minx), (maxy - miny)
    aspect = (dx / dy) if dy else 1.0
    return (n_cols * panel_height * aspect, n_rows * panel_height)


def _set_common_extent(axes, bounds_3857, pad: float = 0.02) -> None:
    """Clip every axis to the same padded extent so maps align edge-to-edge.

    Call *before* ``cx.add_basemap`` so the basemap is fetched for this extent.
    """
    minx, miny, maxx, maxy = bounds_3857
    dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)


def plot_temperature_difference(
    historic_path: Path,
    current_path: Path,
    serbia_3857: gpd.GeoDataFrame,
    output_folder: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Three-panel historic/current/difference temperature figure."""
    hist_stem = os.path.splitext(os.path.basename(historic_path))[0]
    curr_stem = os.path.splitext(os.path.basename(current_path))[0]

    hist_data, hist_bounds, hist_crs = read_tif(historic_path)
    curr_data, curr_bounds, curr_crs = read_tif(current_path)

    if hist_data.shape != curr_data.shape:
        hist_data = resample_to_match(hist_data, curr_data)

    diff_data = curr_data - hist_data
    diff_serbia = mask_to_serbia(diff_data, curr_bounds, curr_crs, serbia_3857)
    valid = diff_serbia[~np.isnan(diff_serbia)]

    if len(valid) > 0:
        print(f"  Min Δ: {valid.min():.4f} °C | Max Δ: {valid.max():.4f} °C | Mean Δ: {valid.mean():.4f} °C")

    cmap_abs = ListedColormap(ABS_COLORS)
    norm_abs = BoundaryNorm(ABS_BINS, cmap_abs.N)
    cmap_diff = ListedColormap(DIFF_COLORS)
    norm_diff = BoundaryNorm(DIFF_BINS, cmap_diff.N)

    l, b, r, t = _reproject_bounds_to_3857(curr_bounds, curr_crs)
    ext = [l, r, b, t]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.13, wspace=0.04)
    fig.suptitle(f"Temperature comparison: {hist_stem} vs {curr_stem}", fontsize=13, fontweight="bold", y=0.97, color="#222222")

    for ax, (data, norm, cmap, title) in zip(axes, [
        (hist_data, norm_abs, cmap_abs, f"Historic\n{hist_stem}"),
        (curr_data, norm_abs, cmap_abs, f"Current\n{curr_stem}"),
        (diff_data, norm_diff, cmap_diff, "Difference\n(current − historic)"),
    ]):
        ax.imshow(data, extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest", alpha=0.80, zorder=3)
        _apply_panel_style(ax, title, serbia_3857)

    if len(valid) > 0:
        stats_text = (
            f"Within Serbia\nMin :  {valid.min():.2f} °C\nMax :  {valid.max():.2f} °C\n"
            f"Mean: {valid.mean():.2f} °C\nMed : {np.median(valid):.2f} °C"
        )
        axes[2].text(0.03, 0.97, stats_text, transform=axes[2].transAxes, fontsize=8.5, verticalalignment="top",
                     fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.92), zorder=10)

    sm_abs = plt.cm.ScalarMappable(cmap=cmap_abs, norm=norm_abs); sm_abs.set_array([])
    cax_abs = fig.add_axes([0.02, 0.06, 0.62, 0.025])
    cbar_abs = fig.colorbar(sm_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_label("Temperature (°C)", fontsize=9); cbar_abs.set_ticks(ABS_BINS); cbar_abs.ax.tick_params(labelsize=8)

    sm_diff = plt.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff); sm_diff.set_array([])
    cax_diff = fig.add_axes([0.68, 0.06, 0.30, 0.025])
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Δ Temperature (°C)", fontsize=9); cbar_diff.set_ticks(DIFF_BINS); cbar_diff.ax.tick_params(labelsize=8)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_folder / f"diff_{hist_stem}_vs_{curr_stem}.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_pavement_temperature(
    data: np.ndarray,
    title: str,
    output_folder: Path,
    bounds,
    crs,
    serbia_3857: gpd.GeoDataFrame,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Single-panel pavement temperature map."""
    l, b, r, t = _reproject_bounds_to_3857(bounds, crs)

    cmap = ListedColormap(ABS_COLORS_PAV)
    norm = BoundaryNorm(ABS_BINS_PAV, cmap.N)

    data_serbia = mask_to_serbia(data, bounds, crs, serbia_3857)
    valid = data_serbia[~np.isnan(data_serbia)]

    print(f"Bin distribution for: {title}")
    print(f"  {'Bin':<15} {'Cells':>8}  {'% of total':>10}")
    print("  " + "─" * 37)
    n = (valid < ABS_BINS_PAV[0]).sum()
    print(f"  {'< ' + str(ABS_BINS_PAV[0]):<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")
    for lo, hi in zip(ABS_BINS_PAV[:-1], ABS_BINS_PAV[1:]):
        n = ((valid >= lo) & (valid < hi)).sum()
        print(f"  {f'{lo} – {hi}':<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")
    n = (valid >= ABS_BINS_PAV[-1]).sum()
    print(f"  {'> ' + str(ABS_BINS_PAV[-1]):<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")
    print(f"  {'─' * 37}\n  {'Total':<15} {len(valid):>8,}  {'100.0%':>10}\n")

    fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)
    ax.imshow(data, extent=[l, r, b, t], origin="upper", cmap=cmap, norm=norm,
              interpolation="nearest", alpha=0.80, zorder=3)
    _apply_panel_style(ax, title, serbia_3857)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Pavement Temperature (°C)", fontsize=9)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.ax.set_xticklabels(["<50", "53", "56", "59", "62", ">64"], fontsize=8)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = title.lower().replace(" ", "_")
    fig.savefig(output_folder / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def apply_urban_heat_island(
    bounds,
    degree_of_urbanization_path: Path,
    current_max_pavement_temperature_path: Path,
    output_tif_path: Path,
) -> np.ndarray:
    """Apply +2°C climate change offset plus UHI offsets for urban classes; save result GeoTIFF."""
    climate_bounds_mollweide = transform_bounds(
        "EPSG:4326", "ESRI:54009",
        bounds.left, bounds.bottom, bounds.right, bounds.top,
    )
    print("Climate bounds in Mollweide:", climate_bounds_mollweide)

    with rasterio.open(degree_of_urbanization_path) as smod_src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            smod_src.crs, "EPSG:4326", smod_src.width, smod_src.height, *smod_src.bounds,
            dst_width=1000, dst_height=1000,
        )
        smod_reprojected = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(smod_src, 1), destination=smod_reprojected,
            src_transform=smod_src.transform, src_crs=smod_src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.nearest, src_nodata=smod_src.nodata, dst_nodata=np.nan,
        )

    smod_reprojected = smod_reprojected.astype(float)
    row_min, col_min = rowcol(dst_transform, bounds.left, bounds.top, op=int)
    row_max, col_max = rowcol(dst_transform, bounds.right, bounds.bottom, op=int)
    row_min, col_min = max(0, row_min), max(0, col_min)
    row_max, col_max = min(dst_height, row_max), min(dst_width, col_max)
    smod_clipped = smod_reprojected[row_min:row_max, col_min:col_max]

    smod_clip_transform = transform_from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        smod_clipped.shape[1], smod_clipped.shape[0],
    )

    data_reprojected = np.empty(smod_clipped.shape, dtype=np.float32)
    with rasterio.open(current_max_pavement_temperature_path) as climate_src:
        reproject(
            source=rasterio.band(climate_src, 1), destination=data_reprojected,
            dst_crs="EPSG:4326", dst_transform=smod_clip_transform, resampling=Resampling.nearest,
        )

    data_reprojected = data_reprojected.astype(float)
    data_reprojected = np.where(data_reprojected <= 0, np.nan, data_reprojected)

    data_future_fine = data_reprojected + 2
    UHI_OFFSETS = {30: 5.0, 23: 5.0, 22: 5.0}
    data_uhi = data_future_fine.copy()
    for smod_class, offset in UHI_OFFSETS.items():
        mask = smod_clipped == smod_class
        data_uhi[mask] = data_uhi[mask] + offset
        print(f"  Class {smod_class}: {mask.sum():>8,} cells  (+{offset}°C)")

    print(f"\nOutput shape: {data_uhi.shape} | Min: {np.nanmin(data_uhi):.2f} °C | Max: {np.nanmax(data_uhi):.2f} °C")

    with rasterio.open(
        output_tif_path, "w", driver="GTiff",
        height=data_uhi.shape[0], width=data_uhi.shape[1],
        count=1, dtype="float32", crs="EPSG:4326",
        transform=smod_clip_transform, nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(data_uhi.astype("float32"), 1)
    print(f"Saved UHI raster: {output_tif_path}")

    return data_uhi


def assign_max_temp_to_roads(
    roads: gpd.GeoDataFrame,
    raster_data: np.ndarray,
    raster_bounds,
    raster_crs,
) -> gpd.GeoDataFrame:
    """Add a ``max_temp`` column with the max raster value per road segment."""
    rows, cols = raster_data.shape
    raster_transform = transform_from_bounds(
        raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top, cols, rows
    )
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=rows, width=cols, count=1,
            dtype=raster_data.dtype, crs=raster_crs, transform=raster_transform,
        ) as dataset:
            dataset.write(raster_data, 1)
        stats = zonal_stats(roads, memfile.name, stats=["max"], nodata=np.nan, all_touched=True)

    roads = roads.copy()
    roads["max_temp"] = [s["max"] for s in stats]
    return roads


def assign_and_plot_road_temperatures(
    raster_data: np.ndarray,
    raster_bounds,
    raster_crs,
    input_parquet: Path,
    parquet_dir: Path,
    gdb_path: Path,
    output_folder: Path,
    title: str | None = None,
    output_crs: str = "EPSG:6316",
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Assign max temperature from raster to each road segment; save parquet+GDB+figure."""
    from utils.arcgis import save_lyrx_layer

    roads = gpd.read_parquet(input_parquet).to_crs(raster_crs)
    roads = assign_max_temp_to_roads(roads, raster_data, raster_bounds, raster_crs)

    n_missing = roads["max_temp"].isna().sum()
    print(f"Roads with temperature assigned: {(~roads['max_temp'].isna()).sum():,}")
    print(f"Roads outside raster extent    : {n_missing:,}")
    if roads["max_temp"].notna().any():
        print(f"Temperature range              : {roads['max_temp'].min():.2f} – {roads['max_temp'].max():.2f} °C")

    valid_temps = roads["max_temp"].dropna()
    print(f"\nRoad segment distribution:")
    print(f"  {'Bin':<15} {'Roads':>8}  {'% of total':>10}")
    print("  " + "─" * 37)
    for lo, hi in zip(ABS_BINS_PAV[:-1], ABS_BINS_PAV[1:]):
        n = ((valid_temps >= lo) & (valid_temps < hi)).sum()
        print(f"  {f'{lo} – {hi}':<15} {n:>8,}  {n/len(valid_temps)*100:>9.1f}%")
    print(f"  {'─' * 37}\n  {'Total':<15} {len(valid_temps):>8,}  {'100.0%':>10}")

    # Determine layer name from parquet_dir + title
    stem = (title or "road_temperatures").lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("°", "deg").replace(",", "")
    stem = stem.replace("__", "_").strip("_")

    save_hazard_vector(roads, parquet_dir, gdb_path, stem, output_crs)

    # Plot
    roads_plot = roads.to_crs(epsg=3857)
    cmap = ListedColormap(ABS_COLORS_PAV)
    norm = BoundaryNorm(ABS_BINS_PAV, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)

    roads_no_temp = roads_plot[roads_plot["max_temp"].isna()]
    if len(roads_no_temp) > 0:
        roads_no_temp.plot(ax=ax, color="grey", linewidth=0.4, alpha=0.5, zorder=3)
    roads_with_temp = roads_plot[~roads_plot["max_temp"].isna()]
    roads_with_temp.plot(ax=ax, column="max_temp", cmap=cmap, norm=norm, linewidth=1.2, zorder=4)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
    ax.set_aspect("equal"); ax.margins(0); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#222222")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Pavement Temperature (°C)", fontsize=9)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.ax.set_xticklabels(["50", "53", "56", "59", "62", "64"], fontsize=8)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_folder / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_pavement_temperature_roads_AB(
    raster_data: np.ndarray,
    raster_bounds,
    raster_crs,
    input_parquet: Path,
    serbia_3857: gpd.GeoDataFrame,
    kosovo_roads_mercator: gpd.GeoDataFrame,
    output_folder: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Two-panel A/B figure: UHI pavement-temperature raster (A) and the road
    network coloured by the same temperatures (B). No title, no Serbia outline.

    Panel A shows only heat pixels overlapping Serbia (partial overlap kept);
    Panel B adds the Kosovo road network in grey.
    """
    cmap = ListedColormap(ABS_COLORS_PAV)
    norm = BoundaryNorm(ABS_BINS_PAV, cmap.N)

    # Panel A — UHI raster, masked to Serbia (keep pixels partly overlapping)
    data_serbia = mask_to_serbia(raster_data, raster_bounds, raster_crs, serbia_3857, all_touched=True)
    l, b, r, t = _reproject_bounds_to_3857(raster_bounds, raster_crs)

    # Panel B — roads coloured by the same (UHI) temperatures
    roads = gpd.read_parquet(input_parquet).to_crs(raster_crs)
    roads = assign_max_temp_to_roads(roads, raster_data, raster_bounds, raster_crs)
    roads_plot = roads.to_crs(epsg=3857)

    # Common extent (Serbia + Kosovo roads) so both panels align edge-to-edge
    # without whitespace while keeping the Kosovo network visible in panel B.
    serbia_geom = serbia_3857.to_crs(3857).geometry
    bounds_3857 = gpd.GeoSeries(
        pd.concat([serbia_geom, kosovo_roads_mercator.geometry]), crs=3857
    ).total_bounds

    # Size the figure so each map fills its axes box exactly (no horizontal gap):
    # reserve the colorbar strip as extra inches rather than shrinking the panels.
    minx, miny, maxx, maxy = bounds_3857
    map_aspect = (maxx - minx) / (maxy - miny)
    panel_h = 8.0
    cb_inches = 0.9
    figw, figh = 2 * panel_h * map_aspect, panel_h + cb_inches

    fig, axes = plt.subplots(1, 2, figsize=(figw, figh), facecolor="white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=cb_inches / figh, wspace=0.02)

    ax_a, ax_b = axes
    ax_a.imshow(data_serbia, extent=[l, r, b, t], origin="upper", cmap=cmap, norm=norm,
                interpolation="nearest", alpha=0.85, zorder=3)

    kosovo_roads_mercator.plot(ax=ax_b, color="grey", linewidth=0.8, alpha=0.5, zorder=2)
    roads_no_temp = roads_plot[roads_plot["max_temp"].isna()]
    if len(roads_no_temp) > 0:
        roads_no_temp.plot(ax=ax_b, color="grey", linewidth=0.4, alpha=0.5, zorder=3)
    roads_plot[~roads_plot["max_temp"].isna()].plot(
        ax=ax_b, column="max_temp", cmap=cmap, norm=norm, linewidth=1.2, zorder=4
    )

    _set_common_extent(axes, bounds_3857)
    for ax in axes:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")

    # A/B labels on top of everything (roads must not cover them)
    for ax, label in zip(axes, ["A", "B"]):
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=16, fontweight="bold",
                verticalalignment="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Pavement Temperature (°C)", fontsize=9)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.ax.set_xticklabels(["50", "53", "56", "59", "62", "64"], fontsize=8)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_folder / "pavement_temperature_roads_AB.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)
