# Standard library
import warnings

# Third-party - Data and scientific computing
import contextily as cx
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import pandas as pd
import xarray as xr
from typing import Tuple
from damagescanner.core import DamageScanner

# Matplotlib-specific imports for figures
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config.network_config import NetworkConfig

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def load_data(config: NetworkConfig) -> Tuple[gpd.GeoDataFrame, xr.Dataset]:
    """
    Load criticality results and clip the flood hazard raster to the target country's extent.
    """
    gdf_results = gpd.read_parquet(config.Path_criticality_results)

    countries = gpd.read_file(config.world_boundaries)
    country = countries.loc[countries.SOV_A3 == "SRB"]
    country_plot = countries.loc[countries.SOV_A3 == "SRB"]

    minx, miny, maxx, maxy = country.total_bounds

    hazard = xr.open_dataset(config.flood_map_RP100, engine="rasterio")
    hazard_clipped = hazard.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    ).load()

    return gdf_results, hazard_clipped, country_plot


def flagged_exposed_segments(row: pd.Series):
    """
    Return True if any flood depth on a road segment exceeds 0.25 meters.
    """
    return any(val > 0.25 for val in row["values"])


def max_depth(row):
    """
    Return the maximum flood depth for a given road segment.
    """
    values = row["values"]
    if values is None or len(values) == 0:
        return 0
    return np.max(values)


def calculate_flood_impact(
    gdf_results: gpd.GeoDataFrame, flood_data: xr.Dataset
) -> gpd.GeoDataFrame:
    """
    Calculate flood impact on roads and return results for segments where
    vehicle-hours-lost (VHL) occur due to impassable flooding.
    """
    exposed_roads = DamageScanner(
        flood_data, gdf_results, curves=pd.DataFrame(), maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False)

    exposed_roads["exposed"] = exposed_roads.progress_apply(
        flagged_exposed_segments, axis=1
    )
    exposed_roads["max_depth"] = exposed_roads.progress_apply(max_depth, axis=1)

    gdf_vhl_flooded = gdf_results.merge(
        exposed_roads.loc[
            exposed_roads["exposed"], ["coverage", "values", "max_depth"]
        ],
        left_index=True,
        right_index=True,
    )

    return gdf_vhl_flooded


def plot_flood_impact_map(
    gdf_vhl_flooded: gpd.GeoDataFrame, config: NetworkConfig
) -> None:
    """
    Plot vehicle-hours-lost (VHL) through flood impact on roads and save figure as PNG.
    """
    bins = [0, 1000, 5000, 10000, 25000, np.inf]
    labels = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]

    gdf_vhl_flooded["vhl_class"] = pd.cut(
        gdf_vhl_flooded["vhl"], bins=bins, labels=labels, include_lowest=True
    )

    linewidth_map = {
        "0-1K": 0.5, "1K-5K": 1.0, "5K-10K": 2.0, "10K-25K": 3.5, "25K+": 5.0,
    }

    gdf_vhl_flooded["linewidth"] = gdf_vhl_flooded["vhl_class"].map(linewidth_map)

    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor="white")

    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]

    for i, (class_name, width) in enumerate(linewidth_map.items()):
        subset = gdf_vhl_flooded[gdf_vhl_flooded["vhl_class"] == class_name]
        if not subset.empty:
            subset.to_crs(3857).plot(
                ax=ax, color=colors[i], linewidth=width, alpha=0.8, label=class_name
            )

    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)

    ax.set_aspect("equal")
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width, label=f"{class_name} vehicle hours")
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]

    ax.legend(
        handles=legend_elements, title="Vehicle Hours Lost", loc="upper right",
        fontsize=9, title_fontsize=10, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
    plt.savefig(
        config.figure_path / "vehicle_hours_lost_map_flooded.png",
        dpi=300, bbox_inches="tight",
    )
    if config.show_figures:
        plt.show()


def read_snowdrift_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Read snow drift data shapefile.
    """
    snow_drift = gpd.read_file(config.Path_snow_drift_data)
    return snow_drift


def calculate_vhl_snow_drift(
    gdf_results: gpd.GeoDataFrame, snow_drift: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Spatially join road segments with snow drift polygons.
    """
    gdf_vhl_snowdrift = gdf_results[
        [
            "from_id", "to_id", "objectid", "oznaka_deo", "smer_gdf1", "kategorija",
            "oznaka_put", "oznaka_poc", "naziv_poce", "oznaka_zav", "naziv_zavr",
            "duzina_deo", "pocetna_st", "zavrsna_st", "stanje", "geometry", "id",
            "passenger_cars", "buses", "light_trucks", "medium_trucks", "heavy_trucks",
            "articulated_vehicles", "total_aadt", "road_length", "speed", "fft",
            "edge_no", "vhl", "phl", "thl", "pkl", "tkl",
        ]
    ].sjoin(snow_drift)

    return gdf_vhl_snowdrift


def read_landslide_data(config: NetworkConfig) -> gpd.GeoDataFrame:
    """
    Read landslide data shapefile and exclude all other unstable occurrences.
    """
    landslides = gpd.read_file(config.Path_landslide_data)
    landslides = landslides[landslides["tip"] == "Klizište"]
    landslides.geometry = landslides.geometry.buffer(10)
    return landslides


def calculate_vhl_landslides(
    gdf_results: gpd.GeoDataFrame, landslides: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Spatially join road segments with landslide polygons.
    """
    gdf_vhl_landslides = gdf_results[
        [
            "from_id", "to_id", "objectid", "oznaka_deo", "smer_gdf1", "kategorija",
            "oznaka_put", "oznaka_poc", "naziv_poce", "oznaka_zav", "naziv_zavr",
            "duzina_deo", "pocetna_st", "zavrsna_st", "stanje", "geometry", "id",
            "passenger_cars", "buses", "light_trucks", "medium_trucks", "heavy_trucks",
            "articulated_vehicles", "total_aadt", "road_length", "speed", "fft",
            "edge_no", "vhl", "phl", "thl", "pkl", "tkl",
        ]
    ].sjoin(landslides)

    return gdf_vhl_landslides


def read_heat_data(config: NetworkConfig) -> xr.Dataset:
    """
    Read heat data .tif file.
    """
    heat_da = rioxarray.open_rasterio(
        config.Future_pavement_temperatures, masked=True
    ).squeeze()

    heat_ds = heat_da.to_dataset(name="temperature")
    return heat_ds


def calculate_heat_impact(
    gdf_results: gpd.GeoDataFrame,
    heat_data: xr.Dataset,
) -> gpd.GeoDataFrame:
    """
    Calculate heat impact on roads based on pavement temperature exposure.
    """
    exposed_roads = DamageScanner(
        heat_data, gdf_results, curves=pd.DataFrame(), maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False)

    exposed_roads["exposed"] = exposed_roads["values"].apply(
        lambda vals: any(v > 50.0 for v in vals if v == v)
    )

    exposed_roads["max_pavement_temp"] = exposed_roads["values"].apply(
        lambda vals: max((v for v in vals if v == v), default=float("nan"))
    )

    gdf_vhl_heat = gdf_results.merge(
        exposed_roads.loc[
            exposed_roads["exposed"], ["coverage", "values", "max_pavement_temp"]
        ],
        left_index=True,
        right_index=True,
    )

    return gdf_vhl_heat


def read_wildfire_data(config: NetworkConfig) -> xr.Dataset:
    """
    Read wildfire risk .tif file (categorical, values 1–6).
    """
    wildfire_da = rioxarray.open_rasterio(
        config.wildfire_risk, masked=True
    ).squeeze()

    wildfire_ds = wildfire_da.to_dataset(name="wildfire_risk")
    return wildfire_ds


def calculate_wildfire_impact(
    gdf_results: gpd.GeoDataFrame,
    wildfire_data: xr.Dataset,
) -> gpd.GeoDataFrame:
    """
    Calculate wildfire impact on roads based on categorical risk exposure (1–6).
    All segments intersecting any risk category > 0 are flagged as exposed (binary).
    """
    exposed_roads = DamageScanner(
        wildfire_data, gdf_results, curves=pd.DataFrame(), maxdam=pd.DataFrame()
    ).exposure(asset_type="roads", disable_progress=False)

    exposed_roads["exposed"] = exposed_roads["values"].apply(
        lambda vals: any(v > 0 for v in vals if v == v)
    )

    # Binary: 1 if any wildfire risk present, 0 otherwise
    exposed_roads["wildfire_risk"] = exposed_roads["values"].apply(
        lambda vals: 1 if any(v > 0 for v in vals if v == v) else 0
    )

    gdf_vhl_wildfire = gdf_results.merge(
        exposed_roads.loc[
            exposed_roads["exposed"], ["coverage", "values", "wildfire_risk"]
        ],
        left_index=True,
        right_index=True,
    )

    return gdf_vhl_wildfire


def calculate_combined_hazard(
    gdf_results: gpd.GeoDataFrame,
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    config: NetworkConfig,
) -> gpd.GeoDataFrame:
    """
    Combine flood, snowdrift, landslide, heat, and wildfire hazard attributes
    into a unified hazard exposure GeoDataFrame and save to parquet.
    """
    s_depth = gdf_vhl_flooded["max_depth"].groupby(level=0).max()
    s_snow = gdf_vhl_snowdrift["dužina_sn"].groupby(level=0).max()
    s_date = gdf_vhl_landslides["datum_evid"].groupby(level=0).max()
    s_temp = gdf_vhl_heat["max_pavement_temp"].groupby(level=0).max()
    s_fire = (gdf_vhl_wildfire["wildfire_risk"].groupby(level=0).max() > 0).astype(int)

    gdf_hazards = pd.concat(
        [
            gdf_results,
            s_depth.rename("max_depth"),
            s_snow.rename("dužina_sn"),
            s_date.rename("datum_evid"),
            s_temp.rename("max_pavement_temp"),
            s_fire.rename("wildfire_risk"),
        ],
        axis=1,
    )

    gdf_hazards["datum_evid"] = gdf_hazards["datum_evid"].dt.strftime("%d/%m/%Y")

    keep_attrs = [
        "oznaka_deo", "smer_gdf1", "kategorija", "oznaka_put",
        "oznaka_poc", "naziv_poce", "oznaka_zav", "naziv_zavr",
        "duzina_deo", "pocetna_st", "zavrsna_st", "stanje", "geometry",
        "passenger_cars", "buses", "light_trucks", "medium_trucks",
        "heavy_trucks", "articulated_vehicles", "total_aadt", "road_length",
        "average_time_disruption", "vhl", "phl", "thl", "pkl", "tkl",
        "max_depth", "dužina_sn", "datum_evid", "max_pavement_temp", "wildfire_risk",
    ]
    gdf_hazards = gdf_hazards[keep_attrs]
    gdf_hazards = gdf_hazards.loc[
        gdf_hazards[["max_depth", "dužina_sn", "datum_evid", "max_pavement_temp", "wildfire_risk"]].any(axis=1)
    ]
    gdf_hazards = gdf_hazards.loc[gdf_hazards["vhl"].notna()]

    gdf_hazards.to_parquet(config.Path_main_network_hazard_exposure)
    gdf_hazards.to_file(
        config.Path_main_network_hazard_exposure.with_suffix(".gpkg"), driver="GPKG"
    )
    gdf_hazards.to_file(
        config.results_path / config.Path_main_network_hazard_exposure.with_suffix(".gpkg").name,
        driver="GPKG",
    )

    return gdf_hazards


def plot_vehicle_hours_lost_per_hazard(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    config: NetworkConfig,
) -> None:
    """
    Plot vehicle-hours-lost (VHL) for all five hazards in a 3×2 grid (2+2+1+legend).
    """
    bins = [0, 1000, 5000, 10000, 25000, float("inf")]
    labels = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    linewidth_map = {"0-1K": 1.0, "1K-5K": 1.5, "5K-10K": 2.0, "10K-25K": 3.5, "25K+": 5.0}

    datasets = {
        "A": ("Floods", gdf_vhl_flooded),
        "B": ("Snow Drift", gdf_vhl_snowdrift),
        "C": ("Landslides", gdf_vhl_landslides),
        "D": ("Heat", gdf_vhl_heat),
        "E": ("Wildfire", gdf_vhl_wildfire),
    }

    for _, (_, gdf) in datasets.items():
        gdf["vhl_class"] = pd.cut(gdf["vhl"], bins=bins, labels=labels, include_lowest=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 20), facecolor="white")
    axes = axes.flatten()

    country_mercator = country_plot.to_crs(3857)

    for idx, (letter, (title, gdf)) in enumerate(datasets.items()):
        ax = axes[idx]
        gdf_merc = gdf.to_crs(3857)

        country_mercator.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.5, zorder=1)

        for i, (class_name, width) in enumerate(linewidth_map.items()):
            subset = gdf_merc[gdf_merc["vhl_class"] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)

        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.05, 0.95, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax_legend = axes[5]
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f"{class_name} vehicle hours")
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]
    ax_legend.legend(
        handles=legend_elements, title="Vehicle Hours Lost", loc="center",
        fontsize=14, title_fontsize=16, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(config.figure_path / "vhl_hazards_comparison.png", dpi=300, bbox_inches="tight")
    if config.show_figures:
        plt.show()


def plot_passenger_hours_lost_per_hazard(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    config: NetworkConfig,
) -> None:
    """
    Plot passenger-hours-lost (PHL) for all five hazards in a 3×2 grid (2+2+1+legend).
    """
    bins_phl = [0, 1000, 5000, 10000, 25000, np.inf]
    labels_phl = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    linewidth_map = {"0-1K": 1, "1K-5K": 1.5, "5K-10K": 2.0, "10K-25K": 2.5, "25K+": 3.0}

    datasets_phl = {
        "A": ("Floods", gdf_vhl_flooded),
        "B": ("Snow Drift", gdf_vhl_snowdrift),
        "C": ("Landslides", gdf_vhl_landslides),
        "D": ("Heat", gdf_vhl_heat),
        "E": ("Wildfire", gdf_vhl_wildfire),
    }

    for _, (_, gdf) in datasets_phl.items():
        gdf["phl_class"] = pd.cut(gdf["phl"], bins=bins_phl, labels=labels_phl, include_lowest=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 20), facecolor="white")
    axes = axes.flatten()

    serbia_mercator = country_plot.to_crs(3857)

    for idx, (letter, (title, gdf)) in enumerate(datasets_phl.items()):
        ax = axes[idx]
        gdf_mercator = gdf.to_crs(3857)

        serbia_mercator.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.5, zorder=1)

        for i, (class_name, width) in enumerate(linewidth_map.items()):
            subset = gdf_mercator[gdf_mercator["phl_class"] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)

        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.05, 0.95, f"{letter}", transform=ax.transAxes, fontsize=20, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax_legend = axes[5]
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f"{class_name} hours")
        for i, (class_name, width) in enumerate(linewidth_map.items())
    ]
    ax_legend.legend(
        handles=legend_elements, title="Passenger Hours Lost", loc="center",
        fontsize=14, title_fontsize=16, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(config.figure_path / "phl_hazards_comparison.png", dpi=300, bbox_inches="tight")
    if config.show_figures:
        plt.show()


def plot_tonnage_kilometers_lost_per_hazard(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    country_plot: gpd.GeoDataFrame,
    config: NetworkConfig,
) -> None:
    """
    Plot tonnage-kilometers-lost (TKL) for all five hazards in a 3×2 grid (2+2+1+legend).
    """
    serbia_mercator = country_plot.to_crs(3857)
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    bins_tkl = [10000, 25000, 50000, 100000, 250000, np.inf]
    labels_tkl = ["5K-25K", "25K-50K", "50K-100K", "100K-250K", "250K+"]
    linewidth_map_tkl = {
        "10-25K": 0.5, "25K-50K": 1.5, "50K-100K": 2.0, "100K-250K": 2.5, "250K+": 3.0,
    }

    datasets_tkl = {
        "A": ("Floods", gdf_vhl_flooded),
        "B": ("Snow Drift", gdf_vhl_snowdrift),
        "C": ("Landslides", gdf_vhl_landslides),
        "D": ("Heat", gdf_vhl_heat),
        "E": ("Wildfire", gdf_vhl_wildfire),
    }

    for _, (_, gdf) in datasets_tkl.items():
        gdf["tkl_class"] = pd.cut(gdf["tkl"], bins=bins_tkl, labels=labels_tkl, include_lowest=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 20), facecolor="white")
    axes = axes.flatten()

    for idx, (letter, (title, gdf)) in enumerate(datasets_tkl.items()):
        ax = axes[idx]
        gdf_mercator = gdf.to_crs(3857)

        serbia_mercator.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=1.5, zorder=1)

        for i, (class_name, width) in enumerate(linewidth_map_tkl.items()):
            subset = gdf_mercator[gdf_mercator["tkl_class"] == class_name]
            if not subset.empty:
                subset.plot(ax=ax, color=colors[i], linewidth=width, zorder=2)

        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, alpha=0.4, attribution=False)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.text(
            0.05, 0.95, f"{letter}", transform=ax.transAxes, fontsize=20, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax_legend = axes[5]
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f"{class_name} ton-km")
        for i, (class_name, width) in enumerate(linewidth_map_tkl.items())
    ]
    ax_legend.legend(
        handles=legend_elements, title="Tonnage Kilometers Lost", loc="center",
        fontsize=14, title_fontsize=16, frameon=True, fancybox=True, shadow=True,
        framealpha=0.9, facecolor="white", edgecolor="#cccccc",
    )

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(config.figure_path / "tkl_hazards_comparison.png", dpi=300, bbox_inches="tight")
    if config.show_figures:
        plt.show()


def print_analysis_summary(
    gdf_vhl_flooded: gpd.GeoDataFrame,
    gdf_vhl_snowdrift: gpd.GeoDataFrame,
    gdf_vhl_landslides: gpd.GeoDataFrame,
    gdf_vhl_heat: gpd.GeoDataFrame,
    gdf_vhl_wildfire: gpd.GeoDataFrame,
    gdf_results: gpd.GeoDataFrame,
) -> None:
    """
    Print hazard-specific summaries of disruption metrics across the road network.
    """
    hazard_datasets = {
        "Floods": gdf_vhl_flooded,
        "Snow Drift": gdf_vhl_snowdrift,
        "Landslides": gdf_vhl_landslides,
        "Heat": gdf_vhl_heat,
        "Wildfire": gdf_vhl_wildfire,
    }

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

    bins_phl = [0, 1000, 5000, 10000, 25000, np.inf]
    labels_phl = ["0-1K", "1K-5K", "5K-10K", "10K-25K", "25K+"]
    bins_tkl = [0, 10000, 50000, 100000, 250000, np.inf]
    labels_tkl = ["0-10K", "10K-50K", "50K-100K", "100K-250K", "250K+"]

    # PHL
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

    # TKL
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

    # THL & PKL
    for metric, label, threshold in [("thl", "TONNAGE HOURS LOST (THL)", 10000), ("pkl", "PASSENGER KILOMETERS LOST (PKL)", 250000)]:
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

    # Overlap
    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS - MULTI-HAZARD EXPOSURE")
    print("=" * 70)
    if "oznaka_deo" in gdf_vhl_flooded.columns:
        seg_sets = {}
        for name, gdf in hazard_datasets.items():
            col = "oznaka_deo" if "oznaka_deo" in gdf.columns else "oznaka_deo_left"
            seg_sets[name] = set(gdf[col].dropna())
            print(f"Total unique segments exposed to {name}: {len(seg_sets[name])}")

        all_names = list(seg_sets.keys())
        print("\nExclusive exposure:")
        for name, s in seg_sets.items():
            others = set().union(*(seg_sets[n] for n in all_names if n != name))
            print(f"  {name} only: {len(s - others)}")

        print("\nDual exposure (all pairs):")
        for i, n1 in enumerate(all_names):
            for n2 in all_names[i + 1:]:
                others = set().union(*(seg_sets[n] for n in all_names if n not in (n1, n2)))
                print(f"  {n1} AND {n2}: {len(seg_sets[n1] & seg_sets[n2] - others)}")

        print(f"\nAll five hazards: {len(set.intersection(*seg_sets.values()))}")

    # National comparison
    print("\n" + "=" * 70)
    print("HAZARD EXPOSURE AS % OF NATIONAL DAILY TRANSPORT")
    print("=" * 70)
    national_pkm_daily = 2069 * 1e6 / 180
    national_tkm_daily = 4677 * 1e6 / 180
    print(f"\nNational daily averages:\n  PKM/day: {national_pkm_daily:,.0f}\n  TKM/day: {national_tkm_daily:,.0f}")
    print("\nPKL as % of national daily:")
    for name, gdf in hazard_datasets.items():
        total = gdf["pkl"].sum()
        print(f"  {name}: {total:,.0f} ({total / national_pkm_daily * 100:.1f}%)")
    print("\nTKL as % of national daily:")
    for name, gdf in hazard_datasets.items():
        total = gdf["tkl"].sum()
        print(f"  {name}: {total:,.0f} ({total / national_tkm_daily * 100:.1f}%)")

    # Master summary
    print("\n" + "=" * 70)
    print("MASTER SUMMARY TABLE")
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


def main():
    """
    Run the full multi-hazard road network exposure analysis.
    """
    config = NetworkConfig()
    config.show_figures = True

    criticality_results, flood_data, country_plot = load_data(config)

    flood_results = calculate_flood_impact(criticality_results, flood_data)

    snow_drift_data = read_snowdrift_data(config)
    vhl_snow_drift = calculate_vhl_snow_drift(criticality_results, snow_drift_data)

    landslide_data = read_landslide_data(config)
    vhl_landslides = calculate_vhl_landslides(criticality_results, landslide_data)

    heat_data = read_heat_data(config)
    vhl_heat = calculate_heat_impact(criticality_results, heat_data)

    wildfire_data = read_wildfire_data(config)
    vhl_wildfire = calculate_wildfire_impact(criticality_results, wildfire_data)

    calculate_combined_hazard(
        criticality_results, flood_results, vhl_snow_drift, vhl_landslides,
        vhl_heat, vhl_wildfire, config
    )

    plot_vehicle_hours_lost_per_hazard(
        flood_results, vhl_snow_drift, vhl_landslides, vhl_heat, vhl_wildfire, country_plot, config
    )
    plot_passenger_hours_lost_per_hazard(
        flood_results, vhl_snow_drift, vhl_landslides, vhl_heat, vhl_wildfire, country_plot, config
    )
    plot_tonnage_kilometers_lost_per_hazard(
        flood_results, vhl_snow_drift, vhl_landslides, vhl_heat, vhl_wildfire, country_plot, config
    )

    if config.print_statistics:
        print_analysis_summary(
            flood_results, vhl_snow_drift, vhl_landslides, vhl_heat, vhl_wildfire, criticality_results
        )


if __name__ == "__main__":
    main()