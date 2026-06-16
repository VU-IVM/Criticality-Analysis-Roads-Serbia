"""
Shared functions for the 5a (main-network hazard criticality) and
5c (flood-scenario accessibility plotting) analysis steps.

Notebooks call these with hardcoded paths; scripts call them via NetworkConfig
attributes.  All saved vector outputs are reprojected to EPSG:6316 and written
to Parquet, an ArcGIS File Geodatabase layer, and an Excel attribute table.
"""

from __future__ import annotations

import pickle
import string
from pathlib import Path
from typing import Any

import contextily as cx
import geopandas as gpd
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from damagescanner.core import DamageScanner
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterstats import zonal_stats
from scipy import stats as scipy_stats
from tqdm import tqdm

from utils.hazard_functions import (
    _grid_figsize,
    _set_common_extent,
    _show_or_close,
    save_hazard_vector,
)

tqdm.pandas()


# ---------------------------------------------------------------------------
# Generic save helper: parquet + GDB layer + Excel attribute table (EPSG:6316)
# ---------------------------------------------------------------------------

def save_criticality_vector(
    gdf: gpd.GeoDataFrame,
    parquet_dir: Path,
    gdb_path: Path,
    layer_name: str,
    output_crs: str = "EPSG:6316",
) -> None:
    """Save *gdf* as parquet + GDB layer (via ``save_hazard_vector``) and as an
    Excel attribute table (geometry column dropped) next to the parquet file."""
    save_hazard_vector(gdf, parquet_dir, gdb_path, layer_name, output_crs)

    excel_path = Path(parquet_dir) / f"{layer_name}.xlsx"
    gdf.drop(columns=gdf.geometry.name).to_excel(excel_path, index=False)
    print(f"Saved {layer_name} -> {excel_path}")


# ===========================================================================
# 5a — Main network hazard criticality
# ===========================================================================

def merge_baseline_roads(
    gdf_results: gpd.GeoDataFrame,
    baseline_roads_path: Path,
    world_path: Path,
) -> gpd.GeoDataFrame:
    """Merge criticality results with the baseline Deonice road sections.

    Sections present in the baseline network but missing from the criticality
    results are appended with zero-filled metric columns so that hazard
    exposure is evaluated for the full network. Only sections within core
    Serbia (excluding Kosovo) are considered.
    """
    gdf_deonice = gpd.read_file(baseline_roads_path)

    world = gpd.read_file(world_path)
    serbia = world.loc[world.SOV_A3 == "SRB"]
    kosovo = world.loc[world.SOV_A3 == "KOS"]
    serbia_only = gpd.overlay(serbia, kosovo, how="difference").to_crs(gdf_deonice.crs)

    gdf_deonice = gpd.sjoin(
        gdf_deonice, serbia_only[["geometry"]], how="inner", predicate="within"
    ).drop(columns=["index_right"])

    existing_ids = set(gdf_results["oznaka_deo"])
    mask_new = ~gdf_deonice["oznaka_deo"].isin(existing_ids)
    gdf_new_rows = gdf_deonice[mask_new].copy()

    shared_cols = [col for col in gdf_results.columns if col in gdf_new_rows.columns]
    cols_only_in_results = [col for col in gdf_results.columns if col not in gdf_new_rows.columns]

    gdf_new_rows = gdf_new_rows[shared_cols].copy()
    for col in cols_only_in_results:
        if gdf_results[col].dtype == object:
            gdf_new_rows[col] = ""
        else:
            gdf_new_rows[col] = 0
    gdf_new_rows = gdf_new_rows[gdf_results.columns]

    gdf_merged = pd.concat([gdf_results, gdf_new_rows], ignore_index=True)
    gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry="geometry", crs=gdf_results.crs)

    print("=== Merge Summary ===")
    print(f"Original gdf_results rows:          {len(gdf_results)}")
    print(f"Deonice rows after Serbia filter:   {len(gdf_deonice)}")
    print(f"Deonice rows already in results:    {(~mask_new).sum()}  (skipped)")
    print(f"New rows added from Deonice:        {len(gdf_new_rows)}")
    print(f"Total rows in merged GeoDataFrame:  {len(gdf_merged)}")

    return gdf_merged


def calculate_flood_exposure(
    gdf_results: gpd.GeoDataFrame,
    flood_path: Path,
    world_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Run DamageScanner flood exposure for the road network.

    Returns (exposed_roads, country_plot). ``exposed_roads`` carries
    ``exposed`` (any depth > 0.25 m) and ``max_depth`` columns.
    """
    world = gpd.read_file(world_path)
    country_plot = world.loc[world.SOV_A3 == "SRB"]
    country_bounds = country_plot.bounds

    hazard_map = xr.open_dataset(flood_path, engine="rasterio")
    hazard_country = hazard_map.rio.clip_box(
        minx=country_bounds.minx.values[0],
        miny=country_bounds.miny.values[0],
        maxx=country_bounds.maxx.values[0],
        maxy=country_bounds.maxy.values[0],
    ).load()

    exposed_roads = DamageScanner(
        hazard_country, gdf_results, curves=pd.DataFrame(), maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False, return_full=False)

    exposed_roads["exposed"] = exposed_roads.progress_apply(
        lambda row: any(val > 0.25 for val in row["values"]), axis=1
    )
    exposed_roads["max_depth"] = exposed_roads.progress_apply(
        lambda row: np.max(row["values"]), axis=1
    )

    return exposed_roads, country_plot


def fill_missing_categories(exposed_roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill missing ``kategorija`` from the road designation (``oznaka_put``)."""
    cat_map = {"A1": "IA", "A2": "IA", "A3": "IA", "A4": "IA", "39": "IB"}
    mask = exposed_roads["kategorija"].isna()
    exposed_roads.loc[mask, "kategorija"] = exposed_roads.loc[mask, "oznaka_put"].map(cat_map)
    return exposed_roads


# ---------------------------------------------------------------------------
# 5a — Elevation bias analysis (road Z-profiles vs DEM)
# ---------------------------------------------------------------------------

def load_vertical_coordinates(vertical_coordinates_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load road sections with vertical (Z) coordinates.

    Keeps only roads where every vertex has a non-zero Z value; adds
    z_min/z_max/z_range columns. Returns (metric_gdf, gdf_4326).
    """
    vertical_coordinates = gpd.read_file(vertical_coordinates_path)

    def has_complete_z(geom):
        if geom.geom_type == "MultiLineString":
            z = [c[2] for line in geom.geoms for c in line.coords]
        else:
            z = [c[2] for c in geom.coords]
        return all(v != 0 for v in z) and len(z) > 1

    vertical_coordinates = vertical_coordinates[
        vertical_coordinates.geometry.apply(has_complete_z)
    ].copy()

    vertical_coordinates["z_min"] = vertical_coordinates.geometry.apply(
        lambda g: min(c[2] for line in g.geoms for c in line.coords)
        if g.geom_type == "MultiLineString" else min(c[2] for c in g.coords)
    )
    vertical_coordinates["z_max"] = vertical_coordinates.geometry.apply(
        lambda g: max(c[2] for line in g.geoms for c in line.coords)
        if g.geom_type == "MultiLineString" else max(c[2] for c in g.coords)
    )
    vertical_coordinates["z_range"] = vertical_coordinates["z_max"] - vertical_coordinates["z_min"]

    vertical_coordinates_h = vertical_coordinates.set_crs(epsg=3909, allow_override=True)
    vertical_coordinates_4326 = vertical_coordinates_h.to_crs(epsg=4326)

    print(f"Roads with complete Z profiles: {len(vertical_coordinates)}")
    return vertical_coordinates, vertical_coordinates_4326


def extract_profiles(
    gdf_metric: gpd.GeoDataFrame,
    gdf_4326: gpd.GeoDataFrame,
    dem_path: Path,
) -> gpd.GeoDataFrame:
    """Extract per-vertex distance, measured Z, and DEM elevation profiles."""
    all_distances, all_z_values, all_dem_values = [], [], []

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)

        for idx in tqdm(range(len(gdf_metric)), desc="Extracting profiles"):
            geom_m = gdf_metric.geometry.iloc[idx]
            if geom_m.geom_type == "MultiLineString":
                coords_m = [c for line in geom_m.geoms for c in line.coords]
            else:
                coords_m = list(geom_m.coords)

            distances = [0]
            for i in range(1, len(coords_m)):
                dx = coords_m[i][0] - coords_m[i - 1][0]
                dy = coords_m[i][1] - coords_m[i - 1][1]
                distances.append(distances[-1] + (dx**2 + dy**2) ** 0.5)
            distances = [d / 1000 for d in distances]

            z_values = [c[2] if c[2] != 0 else np.nan for c in coords_m]

            geom_4326 = gdf_4326.geometry.iloc[idx]
            if geom_4326.geom_type == "MultiLineString":
                coords_4326 = [c for line in geom_4326.geoms for c in line.coords]
            else:
                coords_4326 = list(geom_4326.coords)

            dem_values = []
            for c in coords_4326:
                try:
                    row, col = src.index(c[0], c[1])
                    val = dem_data[row, col]
                    dem_values.append(val / 1000 if val != src.nodata else np.nan)
                except (IndexError, ValueError):
                    dem_values.append(np.nan)

            all_distances.append(distances)
            all_z_values.append(z_values)
            all_dem_values.append(dem_values)

    gdf_metric["distances_km"] = all_distances
    gdf_metric["z_profile"] = all_z_values
    gdf_metric["dem_profile"] = all_dem_values
    return gdf_metric


def plot_elevation_profiles(
    vertical_coordinates: gpd.GeoDataFrame,
    figure_path: Path,
    examples: dict[str, int] | None = None,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Plot example measured-Z vs DEM elevation profiles, one panel per category."""
    if examples is None:
        examples = {"IA": 188, "IB": 1832, "IIA": 1881, "IIB": 1578, "IM": 453}

    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 22))
    panels = [f"{letter}." for letter in string.ascii_uppercase]

    for i, (cat, idx) in enumerate(examples.items()):
        row = vertical_coordinates.loc[idx]

        axes[i].plot(row["distances_km"], row["z_profile"], label="Measured Z", linewidth=2)
        axes[i].plot(row["distances_km"], row["dem_profile"], label="DEM Elevation",
                     linewidth=2, linestyle="--", color="brown")
        axes[i].fill_between(
            row["distances_km"], row["z_profile"], row["dem_profile"],
            alpha=0.15, color="blue",
            where=[not (np.isnan(z) or np.isnan(d))
                   for z, d in zip(row["z_profile"], row["dem_profile"])],
        )

        title = f"{panels[i]} {row['PutOzn']} ({row['PutKateg']}) — {row['CvorPocNaz']} → {row['CvorZavNaz']}"
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_ylabel("Elevation (m)", fontsize=12, fontweight="bold")
        axes[i].legend(fontsize=11, loc="upper right")
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=11)

    axes[-1].set_xlabel("Distance along segment (km)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "elevation_profiles_by_category.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def compute_bias_statistics(vertical_coordinates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute per-road bias statistics (measured Z minus DEM) and terrain stats.

    Returns roads with > 5 valid Z vertices and mean/median/std difference plus
    DEM terrain characteristics, filtered to |mean_diff| < 20 m ("clean" set).
    """
    has_z = vertical_coordinates["z_profile"].apply(
        lambda z: sum(1 for v in z if not np.isnan(v)) > 5
    )
    subset = vertical_coordinates[has_z].copy()
    print(f"Roads with Z data: {has_z.sum()}")

    subset["mean_diff"] = subset.apply(
        lambda row: np.nanmean([z - d for z, d in zip(row["z_profile"], row["dem_profile"])
                                if not (np.isnan(z) or np.isnan(d))]), axis=1)
    subset["median_diff"] = subset.apply(
        lambda row: np.nanmedian([z - d for z, d in zip(row["z_profile"], row["dem_profile"])
                                  if not (np.isnan(z) or np.isnan(d))]), axis=1)
    subset["std_diff"] = subset.apply(
        lambda row: np.nanstd([z - d for z, d in zip(row["z_profile"], row["dem_profile"])
                               if not (np.isnan(z) or np.isnan(d))]), axis=1)

    subset["dem_min"] = subset["dem_profile"].apply(lambda d: np.nanmin(d))
    subset["dem_max"] = subset["dem_profile"].apply(lambda d: np.nanmax(d))
    subset["dem_range"] = subset["dem_max"] - subset["dem_min"]
    subset["dem_mean"] = subset["dem_profile"].apply(lambda d: np.nanmean(d))
    subset["road_length_km"] = subset["distances_km"].apply(lambda d: max(d))
    subset["dem_slope"] = subset["dem_range"] / subset["road_length_km"]

    clean = subset[subset["mean_diff"].abs() < 20]
    bias_lookup = clean.groupby("PutKateg")["mean_diff"].median().to_dict()
    print(f"Median bias per category: {bias_lookup}")

    return clean


def compute_bias_confidence_intervals(clean: gpd.GeoDataFrame) -> pd.DataFrame:
    """Bootstrap 95% CIs for median and parametric CIs for mean bias per category.

    Also prints the pooled IIA+IIB bootstrap estimate used to justify a shared
    bias value for lower-category roads.
    """
    ci_results = []
    for cat, group in clean.groupby("PutKateg"):
        data = group["mean_diff"].dropna()
        n = len(data)
        mean, std = data.mean(), data.std()

        ci_mean = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))

        np.random.seed(42)
        boot_medians = [np.median(np.random.choice(data.values, size=n, replace=True))
                        for _ in range(10000)]
        ci_median = np.percentile(boot_medians, [2.5, 97.5])

        ci_results.append({
            "PutKateg": cat, "n": n, "median": data.median(),
            "median_ci_low": ci_median[0], "median_ci_high": ci_median[1],
            "mean": mean, "mean_ci_low": ci_mean[0], "mean_ci_high": ci_mean[1],
        })

    ci_df = pd.DataFrame(ci_results)
    print(ci_df.to_string(index=False))

    # Pooled IIA + IIB estimate
    pooled = clean[clean["PutKateg"].isin(["IIA", "IIB"])]
    n = len(pooled)
    np.random.seed(42)
    boot = [np.median(np.random.choice(pooled["mean_diff"].values, size=n, replace=True))
            for _ in range(10000)]
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"\nPooled IIA+IIB (n={n})")
    print(f"Median: {pooled['mean_diff'].median():.2f} m")
    print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    return ci_df


def plot_elevation_bias(ci_df: pd.DataFrame, figure_path: Path, dpi: int = 300, show_figures: bool = True) -> None:
    """Plot median/mean elevation bias with 95% CIs per road category."""
    category_order = ["IM", "IA", "IB", "IIA", "IIB"]
    ci_df = ci_df.set_index("PutKateg").loc[category_order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ci_df))
    labels = [f"{row['PutKateg']}\n(n={row['n']})" for _, row in ci_df.iterrows()]
    ax.errorbar(x, ci_df["median"],
                yerr=[ci_df["median"] - ci_df["median_ci_low"],
                      ci_df["median_ci_high"] - ci_df["median"]],
                fmt="o", capsize=8, capthick=2, markersize=10, label="Median + 95% CI", zorder=3)
    ax.errorbar([i + 0.15 for i in x], ci_df["mean"],
                yerr=[ci_df["mean"] - ci_df["mean_ci_low"],
                      ci_df["mean_ci_high"] - ci_df["mean"]],
                fmt="s", capsize=8, capthick=2, markersize=10, color="red", label="Mean + 95% CI", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlabel("Road Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Bias: Road - DEM (m)", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="grey", linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(figure_path) / "elevation_bias_by_category.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


# Final elevation bias (m) per road category derived from the bootstrap analysis
FINAL_ELEVATION_BIAS = {"IA": 4.601, "IM": 3.659, "IB": 0.82, "IIA": 0.82, "IIB": 0.82}


def apply_bias_correction(
    exposed_roads: gpd.GeoDataFrame,
    final_bias: dict[str, float] | None = None,
) -> gpd.GeoDataFrame:
    """Subtract the per-category elevation bias from max flood depth (clipped at 0)."""
    if final_bias is None:
        final_bias = FINAL_ELEVATION_BIAS

    exposed_roads["bias"] = exposed_roads["kategorija"].map(final_bias)
    exposed_roads["corrected_max_depth"] = (
        exposed_roads["max_depth"] - exposed_roads["bias"]
    ).clip(lower=0)

    summary = exposed_roads.groupby("kategorija").agg(
        total=("max_depth", "count"),
        originally_flooded=("max_depth", lambda x: (x > 0).sum()),
        still_flooded=("corrected_max_depth", lambda x: (x > 0).sum()),
        bias=("bias", "first"),
        avg_original_depth=("max_depth", "mean"),
        avg_corrected_depth=("corrected_max_depth", "mean"),
    ).reset_index()
    summary["removed"] = summary["originally_flooded"] - summary["still_flooded"]
    summary["pct_removed"] = (summary["removed"] / summary["originally_flooded"] * 100).round(1)
    print(summary.to_string(index=False))

    return exposed_roads


def plot_flood_exposure_correction(
    exposed_roads: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Two-panel map: flood exposure before (A) and after (B) bias correction."""
    gdf_mercator = exposed_roads.to_crs(3857)
    bounds_3857 = gdf_mercator.total_bounds

    figsize = _grid_figsize(bounds_3857, n_rows=1, n_cols=2, panel_height=8.0)
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.02)

    for idx, (label, col) in enumerate([("A", "max_depth"), ("B", "corrected_max_depth")]):
        ax = axes[idx]

        not_flooded = gdf_mercator[gdf_mercator[col] == 0]
        if not not_flooded.empty:
            not_flooded.plot(ax=ax, color="#c1121f", linewidth=1.5, zorder=2)
        flooded = gdf_mercator[gdf_mercator[col] > 0]
        if not flooded.empty:
            flooded.plot(ax=ax, color="#2171b5", linewidth=1.5, zorder=3)

        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=20, fontweight="bold",
                verticalalignment="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    _set_common_extent(axes, bounds_3857)
    for ax in axes:
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="#2171b5", lw=2, label="Flooded"),
        Line2D([0], [0], color="#c1121f", lw=1, label="Not flooded"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=12,
                   frameon=True, fancybox=True, framealpha=0.9,
                   facecolor="white", edgecolor="#cccccc")

    plt.savefig(Path(figure_path) / "flood_exposure_correction.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_vhl_flooded_map(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """Map of vehicle-hours-lost for flood-exposed roads (bias-corrected)."""
    bins = [0, 1000, 5000, 10000, 25000, np.inf]
    labels = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    gdf_vhl_flooded["vhl_class"] = pd.cut(gdf_vhl_flooded["vhl"], bins=bins, labels=labels, include_lowest=True)

    linewidth_map = {"0-1K": 0.5, "1K-5K": 1.0, "5K-10K": 2.0, "10K-25K": 3.5, "25K+": 5.0}
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor="white")
    for i, (class_name, width) in enumerate(linewidth_map.items()):
        subset = gdf_vhl_flooded[gdf_vhl_flooded["vhl_class"] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(ax=ax, color=colors[i], linewidth=width, alpha=0.8, label=class_name)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [Line2D([0], [0], color=colors[i], lw=width, label=f"{class_name} vehicle hours")
                       for i, (class_name, width) in enumerate(linewidth_map.items())]
    ax.legend(handles=legend_elements, title="Vehicle Hours Lost", loc="upper right",
              fontsize=9, title_fontsize=10, frameon=True, fancybox=True, shadow=True,
              framealpha=0.9, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(Path(figure_path) / "vehicle_hours_lost_map_flooded.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


# ---------------------------------------------------------------------------
# 5a — Other hazard overlays
# ---------------------------------------------------------------------------

# Attribute columns carried over from the criticality results into the
# per-hazard overlays
_OVERLAY_COLS = [
    "from_id", "to_id", "objectid", "oznaka_deo", "smer_gdf1", "kategorija",
    "oznaka_put", "oznaka_poc", "naziv_poce", "oznaka_zav", "naziv_zavr",
    "duzina_deo", "pocetna_st", "zavrsna_st", "stanje", "geometry", "id",
    "passenger_cars", "buses", "light_trucks", "medium_trucks",
    "heavy_trucks", "articulated_vehicles", "total_aadt", "road_length",
    "speed", "fft", "edge_no", "vhl", "phl", "thl", "pkl", "tkl",
]


def calculate_vhl_snowdrift(
    gdf_results: gpd.GeoDataFrame, snow_drift_path: Path
) -> gpd.GeoDataFrame:
    """Spatially join road segments with snow-drift locations."""
    snow_drift = gpd.read_file(snow_drift_path)
    return gdf_results[_OVERLAY_COLS].sjoin(snow_drift)


def calculate_vhl_landslides(
    gdf_results: gpd.GeoDataFrame, landslide_path: Path
) -> gpd.GeoDataFrame:
    """Spatially join road segments with landslides (tip=='Klizište', 10 m buffer)."""
    landslides = gpd.read_file(landslide_path)
    landslides = landslides[landslides["tip"] == "Klizište"]
    landslides.geometry = landslides.geometry.buffer(10)
    return gdf_results[_OVERLAY_COLS].sjoin(landslides)


def calculate_wildfire_exposure(
    gdf_results: gpd.GeoDataFrame,
    wildfire_path: Path,
    world_path: Path,
) -> gpd.GeoDataFrame:
    """Binary wildfire exposure via DamageScanner (any susceptibility class > 0).

    Adds ``wildfire_susc`` (=1 for all exposed segments) plus coverage columns.
    """
    wildfire_map = xr.open_dataset(wildfire_path, engine="rasterio")

    wildfire_binary = wildfire_map.where(wildfire_map > 0, other=np.nan)
    wildfire_binary = wildfire_binary.where(wildfire_binary.isnull(), other=1.0)

    world = gpd.read_file(world_path)
    country_bounds = world.loc[world.SOV_A3 == "SRB"].to_crs(epsg=32634).bounds

    wildfire_country = wildfire_binary.rio.clip_box(
        minx=country_bounds.minx.values[0],
        miny=country_bounds.miny.values[0],
        maxx=country_bounds.maxx.values[0],
        maxy=country_bounds.maxy.values[0],
    ).load()

    gdf_results_32634 = gdf_results.to_crs(epsg=32634)
    exposed_wildfire = DamageScanner(
        wildfire_country, gdf_results_32634, curves=pd.DataFrame(), maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False, return_full=False)

    exposed_wildfire["wildfire_coverage_m"] = exposed_wildfire["coverage"].apply(lambda c: sum(c))
    exposed_wildfire["wildfire_coverage_pct"] = (
        exposed_wildfire["wildfire_coverage_m"] / (exposed_wildfire["road_length"] * 1000) * 100
    )
    exposed_wildfire["wildfire_susc"] = 1

    return exposed_wildfire.to_crs(gdf_results.crs).copy()


def calculate_heat_exposure(
    gdf_results: gpd.GeoDataFrame, pavement_temperature_path: Path
) -> gpd.GeoDataFrame:
    """Max future pavement temperature (UHI raster) per road via zonal statistics."""
    gdf_results_4326 = gdf_results.to_crs("EPSG:4326")
    stats = zonal_stats(
        gdf_results_4326, str(pavement_temperature_path),
        stats=["max"], all_touched=True, nodata=np.nan,
    )
    exposed_heat = gdf_results_4326.copy()
    exposed_heat["max_pavement_temp"] = [s["max"] for s in stats]
    return exposed_heat.to_crs(gdf_results.crs).copy()


def combine_hazard_exposure(
    gdf_results: gpd.GeoDataFrame,
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Combine the per-hazard overlays into one main-network exposure GeoDataFrame.

    Duplicates per base index are aggregated with max; landslide dates are
    formatted as dd/mm/yyyy strings; only segments exposed to at least one
    hazard with a valid VHL value are kept.  Mixed-type object columns are
    cast to string so the result can be written to parquet/GDB/Excel.
    """
    s_depth = gdf_vhl_flooded["corrected_max_depth"].groupby(level=0).max()
    s_snow = gdf_vhl_snowdrift["dužina_sn"].groupby(level=0).max()
    s_date = gdf_vhl_landslides["datum_evid"].groupby(level=0).max()
    s_wildfire = gdf_vhl_wildfire["wildfire_susc"].groupby(level=0).max()
    s_heat = gdf_vhl_heat["max_pavement_temp"].groupby(level=0).max()

    gdf_hazards = pd.concat(
        [gdf_results, s_depth.rename("max_depth"), s_snow.rename("dužina_sn"),
         s_date.rename("datum_evid"), s_wildfire, s_heat],
        axis=1,
    )
    gdf_hazards["datum_evid"] = gdf_hazards["datum_evid"].dt.strftime("%d/%m/%Y")

    keep_attrs = [
        "oznaka_deo", "smer_gdf1", "kategorija", "oznaka_put", "oznaka_poc",
        "naziv_poce", "oznaka_zav", "naziv_zavr", "duzina_deo", "pocetna_st",
        "zavrsna_st", "stanje", "geometry", "passenger_cars", "buses",
        "light_trucks", "medium_trucks", "heavy_trucks", "articulated_vehicles",
        "total_aadt", "road_length", "average_time_disruption",
        "vhl", "phl", "thl", "pkl", "tkl",
        "max_depth", "dužina_sn", "datum_evid", "wildfire_susc", "max_pavement_temp",
    ]
    gdf_hazards = gdf_hazards[keep_attrs]
    gdf_hazards = gdf_hazards.loc[
        gdf_hazards[["max_depth", "dužina_sn", "datum_evid", "wildfire_susc", "max_pavement_temp"]].any(axis=1)
    ]
    gdf_hazards = gdf_hazards.loc[gdf_hazards["vhl"].notna()]

    # Normalize mixed-type object columns so parquet/GDB/Excel writers accept them
    for col in gdf_hazards.select_dtypes(include="object").columns:
        if col == gdf_hazards.geometry.name:
            continue
        has_non_string = gdf_hazards[col].dropna().apply(lambda x: not isinstance(x, str)).any()
        if has_non_string:
            print(f"  Mixed types in '{col}' (dtype=object) — casting to string")
            gdf_hazards[col] = gdf_hazards[col].astype(str)

    return gdf_hazards


def plot_hazard_comparison_grid(
    datasets: dict[str, tuple[str, gpd.GeoDataFrame]],
    column: str,
    bins: list,
    labels: list[str],
    linewidth_map: dict[str, float],
    legend_title: str,
    legend_unit: str,
    country_plot: gpd.GeoDataFrame,
    figure_path: Path,
    file_name: str,
    basemap_alpha: float = 1.0,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """2×2 hazard-comparison maps (A–D) for a disruption metric + legend strip.

    ``datasets`` maps panel letter -> (title, GeoDataFrame); the metric in
    *column* is binned into *labels* and styled via *linewidth_map*.
    """
    # Lightest tint (former lowest category) omitted: the four categories now
    # use the upper four colours so the darkest/thickest styling stays on the
    # open-ended top category.
    colors = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    class_col = f"{column}_class"

    for _, (_, gdf) in datasets.items():
        gdf[class_col] = pd.cut(gdf[column], bins=bins, labels=labels, include_lowest=True)

    bounds_3857 = country_plot.to_crs(3857).total_bounds
    figw, figh = _grid_figsize(bounds_3857, n_rows=2, n_cols=2, panel_height=6.5)

    fig = plt.figure(figsize=(figw, figh + 1.0), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.05], hspace=0.08, wspace=0.04,
                          left=0.01, right=0.99, top=0.96, bottom=0.02)
    axes_map = {
        "A": fig.add_subplot(gs[0, 0:2]), "B": fig.add_subplot(gs[0, 2:4]),
        "C": fig.add_subplot(gs[1, 0:2]), "D": fig.add_subplot(gs[1, 2:4]),
    }
    ax_legend = fig.add_subplot(gs[2, :])

    for letter, (title, gdf) in datasets.items():
        ax = axes_map[letter]
        gdf_mercator = gdf.to_crs(3857)

        for i, (class_name, width) in enumerate(linewidth_map.items()):
            subset = gdf_mercator[gdf_mercator[class_col] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
        ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
                verticalalignment="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    _set_common_extent(list(axes_map.values()), bounds_3857)
    for ax in axes_map.values():
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=basemap_alpha, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")

    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f"{class_name} {legend_unit}")
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]
    ax_legend.legend(handles=legend_elements, title=legend_title, loc="center",
                     fontsize=10, title_fontsize=11, frameon=True, fancybox=True,
                     shadow=True, framealpha=0.9, facecolor="white", ncols=5,
                     edgecolor="#cccccc")

    plt.savefig(Path(figure_path) / file_name, dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_all_hazard_comparisons(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 300,
    show_figures: bool = True,
) -> None:
    """VHL / PHL / TKL 2×2 hazard-comparison figures (Floods, Snow, Landslides, Wildfires)."""
    datasets = {
        "A": ("Floods", gdf_vhl_flooded),
        "B": ("Snow Drift", gdf_vhl_snowdrift),
        "C": ("Landslides", gdf_vhl_landslides),
        "D": ("Wildfires", gdf_vhl_wildfire),
    }

    plot_hazard_comparison_grid(
        datasets, column="vhl",
        bins=[0, 1000, 5000, 10000, np.inf],
        labels=["0-1K", "1K-5K", "5K-10K", "10K+"],
        linewidth_map={"0-1K": 1.5, "1K-5K": 2.0, "5K-10K": 3.5, "10K+": 5.0},
        legend_title="Vehicle Hours Lost", legend_unit="vehicle hours",
        country_plot=country_plot, figure_path=figure_path,
        file_name="vhl_hazards_comparison.png", basemap_alpha=0.4, dpi=dpi,
        show_figures=show_figures,
    )

    plot_hazard_comparison_grid(
        datasets, column="phl",
        bins=[0, 1000, 5000, 10000, np.inf],
        labels=["0-1K", "1K-5K", "5K-10K", "10K+"],
        linewidth_map={"0-1K": 1.5, "1K-5K": 2.0, "5K-10K": 2.5, "10K+": 3.0},
        legend_title="Passenger Hours Lost", legend_unit="hours",
        country_plot=country_plot, figure_path=figure_path,
        file_name="phl_hazards_comparison.png", basemap_alpha=1.0, dpi=dpi,
        show_figures=show_figures,
    )

    plot_hazard_comparison_grid(
        datasets, column="tkl",
        bins=[10000, 25000, 50000, 100000, np.inf],
        labels=["10K-25K", "25K-50K", "50K-100K", "100K+"],
        linewidth_map={"10K-25K": 1.5, "25K-50K": 2.0, "50K-100K": 2.5, "100K+": 3.0},
        legend_title="Tonnage Kilometers Lost", legend_unit="ton-km",
        country_plot=country_plot, figure_path=figure_path,
        file_name="tkl_hazards_comparison.png", basemap_alpha=1.0, dpi=dpi,
        show_figures=show_figures,
    )


def print_hazard_analysis_summary(
    gdf_results: gpd.GeoDataFrame,
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
) -> None:
    """Print disruption-metric summaries, overlap analysis, and national comparison."""
    print("=" * 70)
    print("BASELINE RESULTS - ALL METRICS")
    print("=" * 70)
    print(f"\nTotal road segments analyzed: {len(gdf_results):,}")
    print("\nAll Metrics Summary:")
    print(gdf_results[["phl", "thl", "pkl", "tkl"]].describe().round(2))
    print("\nTotals across network:")
    print(f"  Total PHL: {gdf_results['phl'].sum():,.0f} passenger hours")
    print(f"  Total THL: {gdf_results['thl'].sum():,.0f} ton hours")
    print(f"  Total PKL: {gdf_results['pkl'].sum():,.0f} passenger km")
    print(f"  Total TKL: {gdf_results['tkl'].sum():,.0f} ton km")

    gdf_vhl_snowdrift = gdf_vhl_snowdrift.rename(columns={"oznaka_deo_left": "oznaka_deo"})
    gdf_vhl_landslides = gdf_vhl_landslides.rename(columns={"oznaka_deo_left": "oznaka_deo"})

    hazard_datasets = {
        "Floods": gdf_vhl_flooded,
        "Snow Drift": gdf_vhl_snowdrift,
        "Landslides": gdf_vhl_landslides,
        "Wildfires": gdf_vhl_wildfire,
    }

    bins_phl = [0, 1000, 5000, 10000, 25000, np.inf]
    labels_phl = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    bins_tkl = [0, 10000, 50000, 100000, 250000, np.inf]
    labels_tkl = ["0-10K", "10K-50K", "50K-100K", "100K-250K", "250K+"]

    # --- PHL ---
    print("\n" + "=" * 70)
    print("PASSENGER HOURS LOST (PHL) BY HAZARD")
    print("=" * 70)
    phl_summary = []
    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}\n{hazard_name.upper()}\n{'─' * 50}")
        print(f"Exposed road segments: {len(gdf):,}")
        print(gdf["phl"].describe().round(2))
        gdf["phl_class"] = pd.cut(gdf["phl"], bins=bins_phl, labels=labels_phl, include_lowest=True)
        phl_counts = gdf["phl_class"].value_counts().sort_index()
        phl_pcts = (gdf["phl_class"].value_counts(normalize=True).sort_index() * 100).round(1)
        for label in labels_phl:
            print(f"  {label}: {phl_counts.get(label, 0)} ({phl_pcts.get(label, 0)}%)")
        if "kategorija" in gdf.columns:
            road_summary = gdf.groupby("kategorija")["phl"].agg(["count", "sum", "mean", "max"]).round(2)
            road_summary.columns = ["Count", "Total PHL", "Mean PHL", "Max PHL"]
            print(road_summary.sort_values("Total PHL", ascending=False))
        cols = [c for c in ["oznaka_deo", "kategorija", "naziv_poce", "naziv_zavr", "total_aadt", "phl"] if c in gdf.columns]
        print(gdf.nlargest(5, "phl")[cols].to_string())
        phl_summary.append({
            "Hazard": hazard_name, "Exposed Segments": len(gdf),
            "Total PHL": gdf["phl"].sum(), "Mean PHL": gdf["phl"].mean(),
            "Median PHL": gdf["phl"].median(), "Max PHL": gdf["phl"].max(),
            "Segments >25K": len(gdf[gdf["phl"] >= 25000]),
            "Segments >10K": len(gdf[gdf["phl"] >= 10000]),
        })
    print("\n" + "─" * 50 + "\nPHL HAZARD COMPARISON SUMMARY\n" + "─" * 50)
    print(pd.DataFrame(phl_summary).to_string(index=False))

    # --- TKL ---
    print("\n" + "=" * 70)
    print("TONNAGE KILOMETERS LOST (TKL) BY HAZARD")
    print("=" * 70)
    tkl_summary = []
    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}\n{hazard_name.upper()}\n{'─' * 50}")
        print(f"Exposed road segments: {len(gdf):,}")
        print(gdf["tkl"].describe().round(2))
        gdf["tkl_class"] = pd.cut(gdf["tkl"], bins=bins_tkl, labels=labels_tkl, include_lowest=True)
        tkl_counts = gdf["tkl_class"].value_counts().sort_index()
        tkl_pcts = (gdf["tkl_class"].value_counts(normalize=True).sort_index() * 100).round(1)
        for label in labels_tkl:
            print(f"  {label}: {tkl_counts.get(label, 0)} ({tkl_pcts.get(label, 0)}%)")
        if "kategorija" in gdf.columns:
            road_summary = gdf.groupby("kategorija")["tkl"].agg(["count", "sum", "mean", "max"]).round(2)
            road_summary.columns = ["Count", "Total TKL", "Mean TKL", "Max TKL"]
            print(road_summary.sort_values("Total TKL", ascending=False))
        cols = [c for c in ["oznaka_deo", "kategorija", "naziv_poce", "naziv_zavr", "total_aadt", "tkl"] if c in gdf.columns]
        print(gdf.nlargest(5, "tkl")[cols].to_string())
        tkl_summary.append({
            "Hazard": hazard_name, "Exposed Segments": len(gdf),
            "Total TKL": gdf["tkl"].sum(), "Mean TKL": gdf["tkl"].mean(),
            "Median TKL": gdf["tkl"].median(), "Max TKL": gdf["tkl"].max(),
            "Segments >250K": len(gdf[gdf["tkl"] >= 250000]),
            "Segments >100K": len(gdf[gdf["tkl"] >= 100000]),
        })
    print("\n" + "─" * 50 + "\nTKL HAZARD COMPARISON SUMMARY\n" + "─" * 50)
    print(pd.DataFrame(tkl_summary).to_string(index=False))

    # --- THL & PKL ---
    for metric, label, threshold in [("thl", "TONNAGE HOURS LOST (THL)", 10000),
                                     ("pkl", "PASSENGER KILOMETERS LOST (PKL)", 250000)]:
        print("\n" + "=" * 70)
        print(f"{label} BY HAZARD")
        print("=" * 70)
        summary = []
        for hazard_name, gdf in hazard_datasets.items():
            summary.append({
                "Hazard": hazard_name, "Exposed Segments": len(gdf),
                f"Total {metric.upper()}": gdf[metric].sum(),
                f"Mean {metric.upper()}": gdf[metric].mean(),
                f"Max {metric.upper()}": gdf[metric].max(),
                f"Segments >{threshold // 1000}K": len(gdf[gdf[metric] >= threshold]),
            })
        print(pd.DataFrame(summary).to_string(index=False))

    # --- Overlap analysis ---
    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS - MULTI-HAZARD EXPOSURE")
    print("=" * 70)
    seg_sets = {name: set(gdf["oznaka_deo"].dropna()) for name, gdf in hazard_datasets.items()}
    for name, s in seg_sets.items():
        print(f"Total unique segments exposed to {name}: {len(s)}")
    all_names = list(seg_sets.keys())
    all_exposed = set().union(*seg_sets.values())
    print(f"\nTotal unique segments exposed to any hazard: {len(all_exposed)}")

    print("\nExclusive exposure:")
    for name, s in seg_sets.items():
        others = set().union(*(seg_sets[n] for n in all_names if n != name))
        print(f"  {name} only: {len(s - others)}")

    print("\nDual exposure (all pairs):")
    for i, n1 in enumerate(all_names):
        for n2 in all_names[i + 1:]:
            others = set().union(*(seg_sets[n] for n in all_names if n not in (n1, n2)))
            print(f"  {n1} AND {n2}: {len(seg_sets[n1] & seg_sets[n2] - others)}")

    print(f"\nAll four hazards: {len(set.intersection(*seg_sets.values()))}")

    from collections import Counter
    segment_hazard_count = Counter()
    for seg in all_exposed:
        n = sum(1 for s in seg_sets.values() if seg in s)
        segment_hazard_count[n] += 1
    print("\nSegments by number of hazards:")
    for n in sorted(segment_hazard_count.keys()):
        print(f"  {n} hazard(s): {segment_hazard_count[n]} segments")

    # --- National comparison ---
    print("\n" + "=" * 70)
    print("HAZARD EXPOSURE AS % OF NATIONAL DAILY TRANSPORT")
    print("=" * 70)
    national_pkm_daily = 2069 * 1e6 / 180
    national_tkm_daily = 4677 * 1e6 / 180
    print("\nNational daily averages (road transport, H1 2025):")
    print(f"  Passenger-km/day: {national_pkm_daily:,.0f}")
    print(f"  Ton-km/day: {national_tkm_daily:,.0f}")
    print("\nTotal exposed PKL as % of national daily:")
    for name, gdf in hazard_datasets.items():
        total = gdf["pkl"].sum()
        print(f"  {name}: {total:,.0f} PKL ({total / national_pkm_daily * 100:.1f}% of national daily)")
    print("\nTotal exposed TKL as % of national daily:")
    for name, gdf in hazard_datasets.items():
        total = gdf["tkl"].sum()
        print(f"  {name}: {total:,.0f} TKL ({total / national_tkm_daily * 100:.1f}% of national daily)")

    # --- Key examples ---
    print("\n" + "=" * 70)
    print("KEY EXAMPLES FOR TEXT")
    print("=" * 70)
    for hazard_name, gdf in hazard_datasets.items():
        print(f"\n{'─' * 50}\n{hazard_name.upper()} - TOP CRITICAL SEGMENTS\n{'─' * 50}")
        print("\nTop 3 by Passenger Hours Lost:")
        cols = [c for c in ["oznaka_deo", "kategorija", "oznaka_put", "naziv_poce", "naziv_zavr", "total_aadt", "phl", "pkl"] if c in gdf.columns]
        for _, row in gdf.nlargest(3, "phl")[cols].iterrows():
            print(f"  {row.get('oznaka_put', 'N/A')} ({row.get('kategorija', 'N/A')}): "
                  f"{row.get('naziv_poce', 'N/A')} → {row.get('naziv_zavr', 'N/A')}")
            print(f"    AADT: {row.get('total_aadt', 0):,.0f}, PHL: {row['phl']:,.0f}, PKL: {row.get('pkl', 0):,.0f}")
        print("\nTop 3 by Tonnage Kilometers Lost:")
        cols = [c for c in ["oznaka_deo", "kategorija", "oznaka_put", "naziv_poce", "naziv_zavr", "total_aadt", "thl", "tkl"] if c in gdf.columns]
        for _, row in gdf.nlargest(3, "tkl")[cols].iterrows():
            print(f"  {row.get('oznaka_put', 'N/A')} ({row.get('kategorija', 'N/A')}): "
                  f"{row.get('naziv_poce', 'N/A')} → {row.get('naziv_zavr', 'N/A')}")
            print(f"    AADT: {row.get('total_aadt', 0):,.0f}, THL: {row.get('thl', 0):,.0f}, TKL: {row['tkl']:,.0f}")

    # --- Master summary table ---
    print("\n" + "=" * 70)
    print("MASTER SUMMARY TABLE FOR TEXT")
    print("=" * 70)
    master = []
    for name, gdf in hazard_datasets.items():
        master.append({
            "Hazard": name, "Segments": len(gdf),
            "Total PHL": f"{gdf['phl'].sum():,.0f}", "Total THL": f"{gdf['thl'].sum():,.0f}",
            "Total PKL": f"{gdf['pkl'].sum():,.0f}", "Total TKL": f"{gdf['tkl'].sum():,.0f}",
            "Max PHL": f"{gdf['phl'].max():,.0f}", "Max TKL": f"{gdf['tkl'].max():,.0f}",
            "PHL >25K": len(gdf[gdf["phl"] >= 25000]),
            "TKL >250K": len(gdf[gdf["tkl"] >= 250000]),
        })
    print(pd.DataFrame(master).to_string(index=False))


# ===========================================================================
# 5c — Flood-scenario accessibility plotting
# ===========================================================================

# Common styling for travel-time impact classes
IMPACT_BINS = [0.167, 0.333, 0.5, 0.667, 1.0, np.inf]
IMPACT_LABELS = ["10-20 min", "20-30 min", "30-40 min", "40-60 min", "60+ min"]
IMPACT_COLORS = ["#fcbba1", "#fc9272", "#ef3b2c", "#cb181d", "#a50f15"]
IMPACT_WIDTHS = {"10-20 min": 1.0, "20-30 min": 1.5, "30-40 min": 2.0, "40-60 min": 2.5, "60+ min": 3.0}


def load_basins(basins_csv_path: Path, basins_shapefile_path: Path) -> gpd.GeoDataFrame:
    """Merge basin-level flood statistics with HydroBASINS geometries."""
    basins = pd.read_csv(basins_csv_path)
    all_basins = gpd.read_file(basins_shapefile_path)
    return gpd.GeoDataFrame(basins.merge(all_basins, left_on="basinID", right_on="HYBAS_ID"))


def plot_basin_water_depths(basins: gpd.GeoDataFrame, figure_path: Path, dpi: int = 300, show_figures: bool = True) -> None:
    """Map of mean water depth per basin in five depth classes."""
    bins = [0, 1, 2, 3, 4, np.inf]
    labels = ["0-1m", "1-2m", "2-3m", "3-4m", "4m+"]
    basins["depth_class"] = pd.cut(basins["mean water depth (m)"], bins=bins, labels=labels, include_lowest=True)

    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor="white")
    basins_mercator = basins.to_crs(3857)

    for category, color in zip(labels, colors):
        category_data = basins_mercator[basins_mercator["depth_class"] == category]
        if len(category_data) > 0:
            category_data.plot(ax=ax, color=color, linewidth=0.1, edgecolor="navy", alpha=0.8, legend=False)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [Patch(facecolor=colors[i], label=f"{labels[i]} depth", edgecolor="navy", linewidth=0.5)
                       for i in range(len(labels))]
    ax.legend(handles=legend_elements, title="Mean Water Depth", loc="upper right",
              fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
              framealpha=0.9, facecolor="white", edgecolor="#cccccc")

    total_basins = len(basins)
    mean_depth = basins["mean water depth (m)"].mean()
    ax.text(0.02, 0.02, f"Total Basins: {total_basins:,} | Average Depth: {mean_depth:.2f}m",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(Path(figure_path) / "basin_water_depths.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def load_base_network(network_parquet_path: Path) -> gpd.GeoDataFrame:
    """Load the basin-scenario base network (giant component) in EPSG:3857."""
    base_network = gpd.read_parquet(network_parquet_path)

    edges = base_network.reindex(
        ["from_id", "to_id"] + [x for x in list(base_network.columns) if x not in ["from_id", "to_id"]],
        axis=1,
    )
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:], directed=True)
    graph.vs["id"] = graph.vs["name"]
    graph = graph.connected_components().giant()
    edges = edges[edges["id"].isin(graph.es["id"])]

    return base_network.to_crs(3857)


def calculate_service_criticality(
    results_path: Path, base_network: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Exposed edges with mean positive travel-time impact for one service.

    Loads per-basin disruption outcomes, assigns the basin's mean positive
    delay to its removed edges, keeps exposed edges with ≥ 10 min impact, and
    bins them into ``impact_class``.
    """
    with open(results_path, "rb") as f:
        save_new_results = pickle.load(f)

    pd.options.mode.chained_assignment = None
    pd.set_option("future.no_silent_downcasting", True)

    collect_removed_edges = []
    for basin in save_new_results.keys():
        scenario_outcome = save_new_results[basin]["scenario_outcome"]
        if isinstance(scenario_outcome, pd.DataFrame):
            subset_edges = base_network.loc[
                base_network.osm_id.astype(str).isin(save_new_results[basin]["real_edges_to_remove"])
            ].copy()
            subset_edges["travel_time_impact"] = (
                scenario_outcome.loc[scenario_outcome.Delta > 0]
                .replace([np.inf, -np.inf], 1)
                .Delta.mean()
            )
            collect_removed_edges.append(subset_edges)

    exposed_edges = gpd.GeoDataFrame(pd.concat(collect_removed_edges))[
        ["osm_id", "from_id", "to_id", "highway", "exposed", "geometry", "travel_time_impact"]
    ]
    exposed_edges = (
        exposed_edges.loc[exposed_edges.exposed]
        .reset_index(drop=True)
        .set_crs(3857, allow_override=True)
    )
    exposed_edges["travel_time_impact"] = np.where(
        np.isinf(exposed_edges["travel_time_impact"]), 1, exposed_edges["travel_time_impact"]
    )
    exposed_edges = exposed_edges.to_crs(3857)
    exposed_edges = exposed_edges[exposed_edges["travel_time_impact"] >= 0.167].copy()

    exposed_edges["impact_class"] = pd.cut(
        exposed_edges["travel_time_impact"], bins=IMPACT_BINS, labels=IMPACT_LABELS, include_lowest=True
    )
    return exposed_edges


def plot_service_criticality(
    exposed_edges: gpd.GeoDataFrame,
    base_network: gpd.GeoDataFrame,
    figure_path: Path,
    file_name: str,
    dpi: int = 200,
    show_figures: bool = True,
) -> None:
    """Single map of increased travel time for one service over the base network."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor="white")

    base_network.plot(ax=ax, linewidth=0.1, color="lightgrey", alpha=0.5)

    edges_mercator = exposed_edges.to_crs(3857)
    for i, (category, color) in enumerate(zip(IMPACT_LABELS, IMPACT_COLORS)):
        category_data = edges_mercator[edges_mercator["impact_class"] == category]
        if len(category_data) > 0:
            category_data.plot(ax=ax, color=color, linewidth=IMPACT_WIDTHS[category],
                               alpha=0.9, zorder=5 + i)

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [Patch(facecolor=IMPACT_COLORS[i], label=f"{IMPACT_LABELS[i]} delay",
                             edgecolor="darkred", linewidth=0.5)
                       for i in range(len(IMPACT_LABELS))]
    ax.legend(handles=legend_elements, title="Increased Travel Time", loc="upper right",
              fontsize=10, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
              framealpha=0.9, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(Path(figure_path) / file_name, dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)
    plt.close()


def calculate_agri_criticality(
    results_path: Path,
    base_network: gpd.GeoDataFrame,
    delta_prefix: str = "delta_avg",
) -> dict[str, gpd.GeoDataFrame | None]:
    """Agriculture criticality per sink type ('road', 'port', 'rail').

    ``delta_prefix`` selects the population delta columns:
    'delta_avg' (average over all sinks) or 'delta_nearest' (nearest sink).
    """
    with open(results_path, "rb") as f:
        save_new_results = pickle.load(f)

    pd.options.mode.chained_assignment = None
    pd.set_option("future.no_silent_downcasting", True)
    river_basins = list(save_new_results.keys())

    sink_types = ["road", "port", "rail"]
    exposed_edges_by_type: dict[str, gpd.GeoDataFrame | None] = {}

    for sink_type in sink_types:
        print(f"Processing: {sink_type}...")
        collect_removed_edges = []

        for basin in river_basins:
            result = save_new_results[basin]
            if result.get("status") == "Error" or result.get("df_population") is None:
                continue

            df_population = result["df_population"]
            delta_col = f"{delta_prefix}_{sink_type}"
            if delta_col not in df_population.columns:
                continue

            delta_values = df_population[delta_col]
            valid_deltas = delta_values[(delta_values > 0) & (~np.isinf(delta_values))]
            if len(valid_deltas) == 0:
                continue

            removed_osm_ids = result["real_edges_to_remove"]
            subset_edges = base_network.loc[
                base_network.osm_id.astype(str).isin([str(x) for x in removed_osm_ids])
            ].copy()
            if len(subset_edges) == 0:
                continue

            subset_edges["travel_time_impact"] = valid_deltas.mean()
            collect_removed_edges.append(subset_edges)

        if len(collect_removed_edges) == 0:
            print(f"  No exposed edges found for {sink_type}")
            exposed_edges_by_type[sink_type] = None
            continue

        exposed_edges = gpd.GeoDataFrame(pd.concat(collect_removed_edges))[
            ["osm_id", "from_id", "to_id", "highway", "exposed", "geometry", "travel_time_impact"]
        ]
        exposed_edges = exposed_edges.loc[exposed_edges.exposed].reset_index(drop=True)
        exposed_edges = exposed_edges.set_crs(3857, allow_override=True)
        exposed_edges["travel_time_impact"] = np.where(
            np.isinf(exposed_edges["travel_time_impact"]), 1, exposed_edges["travel_time_impact"]
        )
        exposed_edges = exposed_edges[exposed_edges["travel_time_impact"] >= 0.167].copy()
        if len(exposed_edges) == 0:
            print(f"  No edges with impact >= 10 min for {sink_type}")
            exposed_edges_by_type[sink_type] = None
            continue

        exposed_edges = exposed_edges.to_crs(3857)
        exposed_edges["impact_class"] = pd.cut(
            exposed_edges["travel_time_impact"], bins=IMPACT_BINS, labels=IMPACT_LABELS, include_lowest=True
        )
        exposed_edges_by_type[sink_type] = exposed_edges
        print(f"  {len(exposed_edges)} edges")

    return exposed_edges_by_type


def _plot_impact_panel(ax: Any, gdf: gpd.GeoDataFrame | None, letter: str,
                       base_network: gpd.GeoDataFrame, basemap_alpha: float = 1.0,
                       bounds_3857=None) -> None:
    """One panel of exposed edges by impact class over the muted base network."""
    base_network.plot(ax=ax, linewidth=0.1, color="lightgrey", alpha=0.5)

    if gdf is not None and len(gdf) > 0:
        for category, color in zip(IMPACT_LABELS, IMPACT_COLORS):
            subset = gdf[gdf["impact_class"] == category]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, linewidth=IMPACT_WIDTHS[category], alpha=0.9)

    ax.axis("off")
    ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
            verticalalignment="top", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if bounds_3857 is not None:
        _set_common_extent(ax, bounds_3857)
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=basemap_alpha, attribution=False)
    ax.set_aspect("equal")


def plot_agri_criticality_3x1(
    exposed_edges_by_type: dict[str, gpd.GeoDataFrame | None],
    base_network: gpd.GeoDataFrame,
    figure_path: Path,
    file_name: str,
    legend_title: str,
    dpi: int = 150,
    show_figures: bool = True,
) -> None:
    """3×1 figure: agriculture criticality for road borders (A), ports (B), rail (C)."""
    bounds_3857 = base_network.total_bounds
    figw, figh = _grid_figsize(bounds_3857, n_rows=1, n_cols=3, panel_height=6.0)

    fig, axes = plt.subplots(1, 3, figsize=(figw, figh + 0.9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.9 / (figh + 0.9), wspace=0.02)

    _plot_impact_panel(axes[0], exposed_edges_by_type["road"], "A", base_network, bounds_3857=bounds_3857)
    _plot_impact_panel(axes[1], exposed_edges_by_type["port"], "B", base_network, bounds_3857=bounds_3857)
    _plot_impact_panel(axes[2], exposed_edges_by_type["rail"], "C", base_network, bounds_3857=bounds_3857)

    legend_handles = [Patch(facecolor=IMPACT_COLORS[i], label=IMPACT_LABELS[i],
                            edgecolor="darkred", linewidth=0.5)
                      for i in range(len(IMPACT_LABELS))]
    fig.legend(handles=legend_handles, title=legend_title, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(IMPACT_LABELS), fontsize=14,
               title_fontsize=16, frameon=True, fancybox=True, framealpha=0.9)

    plt.savefig(Path(figure_path) / file_name, dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def plot_criticality_2x2(
    hospital_exposed_edges: gpd.GeoDataFrame,
    factory_exposed_edges: gpd.GeoDataFrame,
    police_exposed_edges: gpd.GeoDataFrame,
    fire_exposed_edges: gpd.GeoDataFrame,
    base_network: gpd.GeoDataFrame,
    figure_path: Path,
    dpi: int = 150,
    show_figures: bool = True,
) -> None:
    """Combined 2×2 figure: hospitals (A), factories (B), police (C), fire (D)."""
    bounds_3857 = base_network.total_bounds
    figw, figh = _grid_figsize(bounds_3857, n_rows=2, n_cols=2, panel_height=6.5)

    fig, axes = plt.subplots(2, 2, figsize=(figw, figh + 0.9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.9 / (figh + 0.9),
                        wspace=0.02, hspace=0.02)

    _plot_impact_panel(axes[0, 0], hospital_exposed_edges, "A", base_network, basemap_alpha=0.4, bounds_3857=bounds_3857)
    _plot_impact_panel(axes[0, 1], factory_exposed_edges, "B", base_network, basemap_alpha=0.4, bounds_3857=bounds_3857)
    _plot_impact_panel(axes[1, 0], police_exposed_edges, "C", base_network, basemap_alpha=0.4, bounds_3857=bounds_3857)
    _plot_impact_panel(axes[1, 1], fire_exposed_edges, "D", base_network, basemap_alpha=0.4, bounds_3857=bounds_3857)

    legend_handles = [Patch(facecolor=IMPACT_COLORS[i], label=f"{IMPACT_LABELS[i]} delay",
                            edgecolor="darkred", linewidth=0.5)
                      for i in range(len(IMPACT_LABELS))]
    fig.legend(handles=legend_handles, title="Increased travel time", loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(IMPACT_LABELS), fontsize=10,
               title_fontsize=12, frameon=True, fancybox=True, framealpha=0.9)

    plt.savefig(Path(figure_path) / "criticality_2x2.png", dpi=dpi, bbox_inches="tight")
    _show_or_close(show_figures)


def save_impact_layers(
    impact_layers: dict[str, gpd.GeoDataFrame | None],
    parquet_dir: Path,
    gdb_path: Path,
    output_crs: str = "EPSG:6316",
) -> None:
    """Save each accessibility impact layer to parquet + GDB layer + Excel."""
    for layer_name, gdf in impact_layers.items():
        if gdf is None or len(gdf) == 0:
            print(f"Skipping {layer_name}: no data")
            continue
        save_criticality_vector(gdf, parquet_dir, gdb_path, layer_name, output_crs)
