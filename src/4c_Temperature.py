import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import contextily as cx
from scipy.ndimage import zoom
from rasterstats import zonal_stats
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.mask import mask as rio_mask
import pyproj
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from rasterio.warp import calculate_default_transform
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.io import MemoryFile
from rasterio.transform import rowcol

from config.network_config import NetworkConfig

sys.path.append(str(NetworkConfig.BASE_DIR))
from utils.arcgis import save_lyrx_layer    

# ── settings ───────────────────────────────────────────────────────────
OUTPUT_FOLDER  = NetworkConfig.temperature_figures_folder    # Folder to save PNGs; set to None to only display
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
COLORMAP       = "RdYlBu_r"  # Colormap for individual / source panels
DIFF_COLORMAP  = "RdBu_r"    # Diverging colormap for the difference panel
DPI            = 300


# ── Bins & colours: absolute temperature panels (°C) ─────────────────────────
# Covers 24–35 °C range, finer steps in the middle where most data falls
ABS_BINS   = [24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
ABS_COLORS = [
    "#2166ac",  # 24–26  cool blue
    "#4393c3",  # 26–27
    "#92c5de",  # 27–28
    "#d1e5f0",  # 28–29  light blue
    "#fddbc7",  # 29–30  light orange
    "#f4a582",  # 30–31
    "#d6604d",  # 31–32
    "#b2182b",  # 32–33
    "#800026",  # 33–34
    "#4d0010",  # 34–35  deep red
]

# ── Bins & colours: difference panel (°C, current − historic) ─────────────────
# Asymmetric: finer resolution on the positive (warming) side
DIFF_BINS   = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
DIFF_COLORS = [
    "#2166ac",  # −2.5 – −1.5  strong cooling
    "#92c5de",  # −1.5 – −0.5  slight cooling
    "#d1e5f0",  # −0.5 –  0.0  near zero / slight cooling
    "#fddbc7",  #  0.0 –  0.5  near zero / slight warming
    "#f4a582",  #  0.5 –  1.0
    "#d6604d",  #  1.0 –  1.5
    "#b2182b",  #  1.5 –  2.0
    "#67000d",  #  2.0 –  2.5  extreme warming
]

#config for pavement plotting
ABS_BINS_PAV   = [50, 53, 56, 59, 62, 64]
ABS_COLORS_PAV = [
    "#ffffb2",  # 50 – 53
    "#fecc5c",  # 53 – 56
    "#fd8d3c",  # 56 – 59
    "#f03b20",  # 59 – 62
    "#bd0026",  # 62 – 64
]

def load_serbia_outline():
    """Return a GeoDataFrame with Serbia polygon in EPSG:4326."""
    countries = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )
    serbia = countries[countries["NAME"].str.lower() == "serbia"]
    if serbia.empty:
        raise ValueError("Serbia not found in Natural Earth dataset.")
    return serbia.to_crs("EPSG:4326")


def read_tif(path):
    """Open a GeoTIFF; return (data_2d, bounds, crs) with nodata → NaN."""
    with rasterio.open(path) as src:
        data   = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        crs    = src.crs
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, bounds, crs


def resample_to_match(data_src, data_ref):
    """Bilinear-resample data_src to the same shape as data_ref."""
    zoom_r = data_ref.shape[0] / data_src.shape[0]
    zoom_c = data_ref.shape[1] / data_src.shape[1]
    nan_mask       = np.isnan(data_src).astype(float)
    resampled      = zoom(np.nan_to_num(data_src, nan=0.0), (zoom_r, zoom_c), order=1)
    resampled_mask = zoom(nan_mask, (zoom_r, zoom_c), order=1) > 0.5
    resampled[resampled_mask] = np.nan
    return resampled


def _reproject_bounds_to_3857(bounds, crs):
    """Convert raster bounds from native CRS to Web Mercator (EPSG:3857)."""
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    left,  bottom = transformer.transform(bounds.left,  bounds.bottom)
    right, top    = transformer.transform(bounds.right, bounds.top)
    return left, bottom, right, top


def _apply_panel_style(ax, title, serbia_3857):
    """White-background panel with Serbia outline and OSM basemap."""
    serbia_3857.plot(ax=ax, facecolor="none", edgecolor="#333333",
                     linewidth=1.5, zorder=4)
    cx.add_basemap(ax=ax, source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.35, attribution=False, zoom="auto")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#222222")


def _add_colorbar(fig, ax, img, label=None):
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    if label:
        cbar.set_label(label, fontsize=8)


def _save_or_show(fig, output_folder, stem):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"{stem}.png")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved → {out_path}")
    
    if NetworkConfig.show_figures:
        plt.show()

def mask_to_serbia(data, bounds, crs, serbia_gdf):
    """
    Return a copy of `data` with all pixels outside Serbia set to NaN.
    Uses rasterio.features.geometry_mask to burn the Serbia polygon onto
    the raster grid, then masks out-of-bounds cells.
    """

    serbia_proj = serbia_gdf.to_crs(crs)
    rows, cols  = data.shape
    transform   = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, cols, rows
    )
    outside = geometry_mask(
        [mapping(geom) for geom in serbia_proj.geometry],
        out_shape=(rows, cols),
        transform=transform,
        invert=False,
    )
    masked = data.copy()
    masked[outside] = np.nan
    return masked

def plot_difference(historic_path, current_path, serbia_3857, output_folder):
    hist_stem = os.path.splitext(os.path.basename(historic_path))[0]
    curr_stem = os.path.splitext(os.path.basename(current_path))[0]

    hist_data, hist_bounds, hist_crs = read_tif(historic_path)
    curr_data, curr_bounds, curr_crs = read_tif(current_path)

    # Extent mismatch warning
    if any(abs(getattr(hist_bounds, k) - getattr(curr_bounds, k)) > 0.5
           for k in ("left", "right", "bottom", "top")):
        print("⚠ Warning: spatial extents differ — difference values may be unreliable.")

    # Resample if grids differ
    if hist_data.shape != curr_data.shape:
        print(f"  Resampling historic {hist_data.shape} → current {curr_data.shape} …")
        hist_data = resample_to_match(hist_data, curr_data)

    diff_data = curr_data - hist_data

    # ── Statistics: Serbia pixels only ───────────────────────────────────────
    diff_serbia = mask_to_serbia(diff_data, curr_bounds, curr_crs, serbia_3857)
    valid       = diff_serbia[~np.isnan(diff_serbia)]

    if len(valid) == 0:
        print("⚠ No valid pixels found inside Serbia — check raster extent.")
    else:
        print("─" * 45)
        print(f"  Difference statistics (within Serbia)")
        print(f"  {'Pixels used':<20}: {len(valid):,}")
        print(f"  {'Min  Δ':<20}: {valid.min():.4f} °C")
        print(f"  {'Max  Δ':<20}: {valid.max():.4f} °C")
        print(f"  {'Mean Δ':<20}: {valid.mean():.4f} °C")
        print(f"  {'Median Δ':<20}: {np.median(valid):.4f} °C")
        print(f"  {'Std dev Δ':<20}: {valid.std():.4f} °C")
        print("─" * 45)

    # ── Colormaps & norms ─────────────────────────────────────────────────────
    cmap_abs  = ListedColormap(ABS_COLORS)
    norm_abs  = BoundaryNorm(ABS_BINS, cmap_abs.N)
    cmap_diff = ListedColormap(DIFF_COLORS)
    norm_diff = BoundaryNorm(DIFF_BINS, cmap_diff.N)

    # Web-Mercator extent (same for all three panels)
    l, b, r, t = _reproject_bounds_to_3857(curr_bounds, curr_crs)
    ext = [l, r, b, t]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="white")
    fig.subplots_adjust(
        left=0.02, right=0.98,
        top=0.91,  bottom=0.13,
        wspace=0.04,
    )
    fig.suptitle(
        f"Temperature comparison:  {hist_stem}  vs  {curr_stem}",
        fontsize=13, fontweight="bold", y=0.97, color="#222222",
    )

    datasets = [
        (hist_data, norm_abs,  cmap_abs,  f"Historic\n{hist_stem}"),
        (curr_data, norm_abs,  cmap_abs,  f"Current\n{curr_stem}"),
        (diff_data, norm_diff, cmap_diff, "Difference\n(current − historic)"),
    ]

    for ax, (data, norm, cmap, title) in zip(axes, datasets):
        ax.imshow(
            data,
            extent=ext,
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            alpha=0.80,
            zorder=3,
        )
        _apply_panel_style(ax, title, serbia_3857)

    # ── Stats annotation on the difference panel ──────────────────────────────
    if len(valid) > 0:
        stats_text = (
            f"Within Serbia\n"
            f"Min :  {valid.min():.2f} °C\n"
            f"Max :  {valid.max():.2f} °C\n"
            f"Mean: {valid.mean():.2f} °C\n"
            f"Med : {np.median(valid):.2f} °C"
        )
        axes[2].text(
            0.03, 0.97, stats_text,
            transform=axes[2].transAxes,
            fontsize=8.5, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
            zorder=10,
        )

    # ── Shared horizontal colorbar — absolute panels (Historic + Current) ─────
    sm_abs = plt.cm.ScalarMappable(cmap=cmap_abs, norm=norm_abs)
    sm_abs.set_array([])
    # Span under the first two panels
    cax_abs = fig.add_axes([0.02, 0.06, 0.62, 0.025])
    cbar_abs = fig.colorbar(sm_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_label("Temperature (°C)", fontsize=9)
    cbar_abs.set_ticks(ABS_BINS)
    cbar_abs.ax.tick_params(labelsize=8)

    # ── Horizontal colorbar — difference panel ────────────────────────────────
    sm_diff = plt.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array([])
    # Span under the third panel
    cax_diff = fig.add_axes([0.68, 0.06, 0.30, 0.025])
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Δ Temperature (°C)", fontsize=9)
    cbar_diff.set_ticks(DIFF_BINS)
    cbar_diff.ax.tick_params(labelsize=8)

    _save_or_show(fig, output_folder, f"diff_{hist_stem}_vs_{curr_stem}")


def plot_single(tif_path_or_data, title, output_folder, serbia_3857, bounds=None, crs=None):
    if isinstance(tif_path_or_data, str):
        data, bounds, crs = read_tif(tif_path_or_data)
        stem = os.path.splitext(os.path.basename(tif_path_or_data))[0]
    else:
        data = tif_path_or_data
        stem = title.lower().replace(" ", "_")

    l, b, r, t = _reproject_bounds_to_3857(bounds, crs)

    cmap = ListedColormap(ABS_COLORS_PAV)
    norm = BoundaryNorm(ABS_BINS_PAV, cmap.N)

    # ── Bin statistics (Serbia only) ──────────────────────────────────────────
    data_serbia = mask_to_serbia(data, bounds, crs, serbia_3857)
    valid = data_serbia[~np.isnan(data_serbia)]

    print(f"Bin distribution for: {title}")
    print(f"  {'Bin':<15} {'Cells':>8}  {'% of total':>10}")
    print("  " + "─" * 37)

    # Below lowest bin
    n = (valid < ABS_BINS_PAV[0]).sum()
    print(f"  {'< ' + str(ABS_BINS_PAV[0]):<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")

    # Each bin
    for lo, hi in zip(ABS_BINS_PAV[:-1], ABS_BINS_PAV[1:]):
        n = ((valid >= lo) & (valid < hi)).sum()
        print(f"  {f'{lo} – {hi}':<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")

    # Above highest bin
    n = (valid >= ABS_BINS_PAV[-1]).sum()
    print(f"  {'> ' + str(ABS_BINS_PAV[-1]):<15} {n:>8,}  {n/len(valid)*100:>9.1f}%")

    print(f"  {'─' * 37}")
    print(f"  {'Total':<15} {len(valid):>8,}  {'100.0%':>10}")
    print()

    fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)

    ax.imshow(
        data,
        extent=[l, r, b, t],
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.80,
        zorder=3,
    )
    _apply_panel_style(ax, title, serbia_3857)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Pavement Temperature (°C)", fontsize=9)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.ax.set_xticklabels(["<50", "53", "56", "59", "62", ">64"], fontsize=8)

    _save_or_show(fig, output_folder, stem)


def apply_urban_heat_island(bounds):
    # ── Reproject climate bounds into SMOD CRS (Mollweide) ───────────────────────
    climate_bounds_mollweide = transform_bounds(
        "EPSG:4326",    # from
        "ESRI:54009",   # to
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    print("Climate bounds in Mollweide:", climate_bounds_mollweide)

    # ── Reproject SMOD to EPSG:4326 clipped to climate extent ────────────────────
    # First figure out the output shape and transform

    with rasterio.open(NetworkConfig.degree_of_urbanization) as smod_src:
        # Calculate transform for reprojected SMOD in EPSG:4326
        # matching the climate raster extent
        dst_transform, dst_width, dst_height = calculate_default_transform(
            smod_src.crs,
            "EPSG:4326",
            smod_src.width,
            smod_src.height,
            *smod_src.bounds,
            dst_width  = 1000,   # keep manageable resolution
            dst_height = 1000,
        )

        smod_reprojected = np.empty((dst_height, dst_width), dtype=np.float32)

        reproject(
            source        = rasterio.band(smod_src, 1),
            destination   = smod_reprojected,
            src_transform = smod_src.transform,
            src_crs       = smod_src.crs,
            dst_transform = dst_transform,
            dst_crs       = "EPSG:4326",
            resampling    = Resampling.nearest,   # nearest for class data — never interpolate classes
            src_nodata    = smod_src.nodata,
            dst_nodata    = np.nan,
        )

    smod_reprojected = smod_reprojected.astype(float)

    # ── Clip SMOD to the climate raster extent ───────────────────────────────────
    # Get pixel coordinates of the climate bounds within the reprojected SMOD

    row_min, col_min = rowcol(dst_transform, bounds.left,  bounds.top,    op=int)
    row_max, col_max = rowcol(dst_transform, bounds.right, bounds.bottom, op=int)

    # Clamp to valid array bounds
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(dst_height, row_max)
    col_max = min(dst_width,  col_max)

    smod_clipped = smod_reprojected[row_min:row_max, col_min:col_max]
    print("SMOD clipped shape:", smod_clipped.shape)
    print("Unique SMOD classes:", np.unique(smod_clipped[~np.isnan(smod_clipped)]))

    # ── Reproject climate onto the clipped SMOD grid ─────────────────────────────
    # Compute the transform for the clipped SMOD window

    smod_clip_transform = transform_from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        smod_clipped.shape[1], smod_clipped.shape[0]
    )

    data_reprojected = np.empty(smod_clipped.shape, dtype=np.float32)


    with rasterio.open(NetworkConfig.current_max_pavement_temperature) as climate_src:
        reproject(
            source        = rasterio.band(climate_src, 1),
            destination   = data_reprojected,
            dst_crs       = "EPSG:4326",
            dst_transform = smod_clip_transform,
            resampling    = Resampling.nearest,   # ← was bilinear
        )

    data_reprojected = data_reprojected.astype(float)
    data_reprojected = np.where(data_reprojected <= 0, np.nan, data_reprojected)

    # ── Apply offsets ─────────────────────────────────────────────────────────────
    data_future_fine = data_reprojected + 2

    UHI_OFFSETS = {
        30: 5.0,   # Urban Centre
        23: 5.0,   # Dense Urban Cluster
        22: 5.0,   # Semi-dense Cluster
    }

    data_uhi = data_future_fine.copy()
    for smod_class, offset in UHI_OFFSETS.items():
        mask = smod_clipped == smod_class
        data_uhi[mask] = data_uhi[mask] + offset
        print(f"  Class {smod_class}: {mask.sum():>8,} cells  (+{offset}°C)")

    # bounds and crs to pass into plot_single
    uhi_bounds = bounds   # now in EPSG:4326 matching climate raster
    uhi_crs    = "EPSG:4326"

    print(f"\nOutput shape : {data_uhi.shape}")
    print(f"Min          : {np.nanmin(data_uhi):.2f} °C")
    print(f"Max          : {np.nanmax(data_uhi):.2f} °C")

    #save results
    out_tif = NetworkConfig.intermediate_results_path / "future_pavement_temperatures.tif"

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=data_uhi.shape[0],
        width=data_uhi.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=smod_clip_transform,
        nodata=np.nan,
        compress="deflate"  # optional but recommended
    ) as dst:
        dst.write(data_uhi.astype("float32"), 1)

    print(f"Saved: {out_tif}")

    return data_uhi

def assign_and_plot_road_temperatures(raster_data, raster_bounds, raster_crs, 
                                       input_parquet, output_parquet, output_folder, title=None):
    """
    Assigns max temperature from raster to each road segment,
    saves the result to parquet, prints bin distribution and plots the road network.
    """
    # ── Load roads ────────────────────────────────────────────────────────────
    roads = gpd.read_parquet(input_parquet)
    roads = roads.to_crs(raster_crs)

    # ── Write raster to memory for rasterstats ────────────────────────────────
    rows, cols        = raster_data.shape
    raster_transform  = transform_from_bounds(
        raster_bounds.left, raster_bounds.bottom,
        raster_bounds.right, raster_bounds.top,
        cols, rows
    )

    with MemoryFile() as memfile:
        with memfile.open(
            driver    = "GTiff",
            height    = rows,
            width     = cols,
            count     = 1,
            dtype     = raster_data.dtype,
            crs       = raster_crs,
            transform = raster_transform,
        ) as dataset:
            dataset.write(raster_data, 1)

        stats = zonal_stats(
            roads,
            memfile.name,
            stats       = ["max"],
            nodata      = np.nan,
            all_touched = True,
        )

    roads["max_temp"] = [s["max"] for s in stats]

    # ── Print summary ─────────────────────────────────────────────────────────
    n_missing = roads["max_temp"].isna().sum()
    print(f"Roads with temperature assigned : {(~roads['max_temp'].isna()).sum():,}")
    print(f"Roads outside raster extent     : {n_missing:,}")
    print(f"Temperature range on roads      : {roads['max_temp'].min():.2f} – {roads['max_temp'].max():.2f} °C")

    # ── Bin distribution ──────────────────────────────────────────────────────
    valid_temps = roads["max_temp"].dropna()
    print(f"\nRoad segment distribution:")
    print(f"  {'Bin':<15} {'Roads':>8}  {'% of total':>10}")
    print("  " + "─" * 37)
    for lo, hi in zip(ABS_BINS_PAV[:-1], ABS_BINS_PAV[1:]):
        n = ((valid_temps >= lo) & (valid_temps < hi)).sum()
        print(f"  {f'{lo} – {hi}':<15} {n:>8,}  {n/len(valid_temps)*100:>9.1f}%")
    print(f"  {'─' * 37}")
    print(f"  {'Total':<15} {len(valid_temps):>8,}  {'100.0%':>10}")

    # ── Save to parquet ───────────────────────────────────────────────────────
    roads.to_parquet(output_parquet)
    print(f"\nSaved → {output_parquet}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    roads_plot = roads.to_crs(epsg=3857)

    cmap = ListedColormap(ABS_COLORS_PAV)
    norm = BoundaryNorm(ABS_BINS_PAV, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)

    roads_no_temp = roads_plot[roads_plot["max_temp"].isna()]
    if len(roads_no_temp) > 0:
        roads_no_temp.plot(ax=ax, color="grey", linewidth=0.4, alpha=0.5, zorder=3)

    roads_with_temp = roads_plot[~roads_plot["max_temp"].isna()]
    roads_with_temp.plot(
        ax        = ax,
        column    = "max_temp",
        cmap      = cmap,
        norm      = norm,
        linewidth = 1.2,
        zorder    = 4,
    )

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
    ax.set_aspect("equal")
    ax.margins(0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#222222")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Pavement Temperature (°C)", fontsize=9)
    cbar.set_ticks(ABS_BINS_PAV)
    cbar.ax.set_xticklabels(["50", "53", "56", "59", "62", "64"], fontsize=8)

    stem = output_parquet.stem

    # ── Classify into bins for ArcGIS styling ────────────────────────────────
    bin_labels = [f"{lo}–{hi} °C" for lo, hi in zip(ABS_BINS_PAV[:-1], ABS_BINS_PAV[1:])]
    roads["temp_class"] = pd.cut(
        roads["max_temp"],
        bins=ABS_BINS_PAV,
        labels=bin_labels,
        include_lowest=True,
    ).astype(str)
    roads["temp_class"] = roads["temp_class"].where(roads["max_temp"].notna(), other="No data")

    # ── Save GeoPackage + ArcGIS layer ───────────────────────────────────────
    original_crs = roads.crs
    roads = roads.to_crs(original_crs)  # already in original CRS here, no reproject needed

    temp_labels = ["No data"] + bin_labels
    temp_colors = ["#d3d3d3"] + ABS_COLORS_PAV
    temp_widths = {label: 1.2 for label in temp_labels}  # uniform width

    save_lyrx_layer(
        gdf=roads,
        gpkg_path=NetworkConfig.arcgis_gpgk / f"{stem}.gpkg",
        lyrx_path=NetworkConfig.arcgis_results / f"{stem}.lyrx",
        layer_name=stem,
        labels=temp_labels,
        colors=temp_colors,
        width_mapping=temp_widths,
        field="temp_class",
        title=title or stem,
    )

    _save_or_show(fig, output_folder, stem)



def main():
    """
    Load and process heat data from the climate atlas of Serbia about the seven hottest days of the year for air and pavement temperature. 
    To account for climate change in the medium future, 2 degree Celsius and urban heat islands are added.
    Finally, the heat data is applied to the national road network of Serbia. 
    """
    print("Loading Serbia outline …")
    # Load country outline
    world = gpd.read_file(NetworkConfig.world_boundaries)
    country_plot = world.loc[world.SOV_A3 == 'SRB']
    serbia_3857 = country_plot.to_crs(3857)
    print(f"Loaded Serbia: {len(serbia_3857)} polygon(s)")

    # ── Run ───────────────────────────────────────────────────────────────────────
    plot_difference(NetworkConfig.historic_temperature, NetworkConfig.current_temperature, serbia_3857, OUTPUT_FOLDER)
    
    data, bounds, crs = read_tif(NetworkConfig.current_max_pavement_temperature)

    # Mask to Serbia only
    data_serbia = mask_to_serbia(data, bounds, crs, serbia_3857)
    valid = data_serbia[~np.isnan(data_serbia)]

    print(f"Min  : {valid.min():.2f} °C")
    print(f"Max  : {valid.max():.2f} °C")
    print(f"\nQuintile breaks (5 equal portions):")
    for p, v in zip([0, 20, 40, 60, 80, 100], np.percentile(valid, [0, 20, 40, 60, 80, 100])):
        print(f"  {p:>3}th percentile: {v:.2f} °C")

    #plot max 7 day pavement temperature (average 2021 to 2025)
    plot_single(data, "Pavement Temperature", OUTPUT_FOLDER, serbia_3857, bounds=bounds, crs=crs)

    # Adjust for future climate change
    data_future = data + 2
    plot_single(data_future, "Pavement Temperature (+ 2°C)", OUTPUT_FOLDER, serbia_3857, bounds=bounds, crs=crs)

    #adjust for urban heat island effect
    data_uhi = apply_urban_heat_island(bounds)
    plot_single(data_uhi, "Pavement Temperature with urban heat islands under climate change", OUTPUT_FOLDER, serbia_3857, bounds=bounds, crs=crs)

    # ── Load roads ────────────────────────────────────────────────────────────────
    roads = gpd.read_parquet(NetworkConfig.intermediate_results_path / "PERS_directed_final.parquet")

    # Ensure roads are in the same CRS as the temperature data
    roads = roads.to_crs(crs)

    # Current pavement temperatures
    print("Current max pavement temperatures (without urban heat island effect)")
    assign_and_plot_road_temperatures(
        raster_data   = data,
        raster_bounds = bounds,
        raster_crs    = crs,
        input_parquet  = NetworkConfig.Path_processed_road_network,
        output_parquet = NetworkConfig.roads_current_max_pavement_temperature,
        output_folder  = OUTPUT_FOLDER,
        title          = "Road Network — Pavement Temperature",
    )

    # Future temperatures with UHI
    print("Future max pavement temperatures considering climate change and urban heat islands")
    assign_and_plot_road_temperatures(
        raster_data   = data_uhi,
        raster_bounds = bounds,
        raster_crs    = crs,
        input_parquet  = NetworkConfig.Path_processed_road_network,
        output_parquet = NetworkConfig.roads_future_max_pavement_temperature,
        output_folder  = OUTPUT_FOLDER,
    )

if __name__ == "__main__":
    main()