# Converted from 4_Hazard_maps.ipynb


# ===== Cell 1 =====
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import shapely
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import re
from exactextract import exact_extract
from tqdm import tqdm
import damagescanner.download as download

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
from lonboard import viz
from damagescanner.vector import _get_cell_area_m2
from pyproj import Geod
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import contextily as cx
import geopandas as gpd
import pandas as pd
from matplotlib.lines import Line2D
import rioxarray as rxr
import pyproj
from typing import Tuple
from config.network_config import NetworkConfig

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning) # exactextract gives a warning that is invalid




def load_data(config: NetworkConfig) -> Tuple[xr.Dataset, gpd.GeoDataFrame]:
    """
    Load a country's boundary and clip a European flood hazard raster to its extent.

    Parameters
    ----------
    config : NetworkConfig
        Must provide `data_path` pointing to the directory containing:
        - ne_10m_admin_0_countries.shp
        - Europe_RP100_filled_depth.tif

    Returns
    -------
    xr.Dataset
        Flood depth dataset clipped to the selected country's extent.
    gpd.GeoDataFrame
        Boundary geometry of the selected country.
    """
    # Load world countries shapefile
    countries = gpd.read_file(config.data_path / "ne_10m_admin_0_countries.shp")

    # Select target country (hardcoded SOV_A3, but can be made parameterizable)
    country = countries.loc[countries.SOV_A3 == "SRB"]
    minx, miny, maxx, maxy = country.total_bounds

    # Load hazard raster
    hazard = xr.open_dataset(
        config.data_path / "Europe_RP100_filled_depth.tif",
        engine="rasterio"
    )

    # Clip raster to country bounding box
    hazard_clipped = hazard.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    ).load()

    return hazard_clipped, country



def plot_flood_hazard_map(config: NetworkConfig, hazard: xr.Dataset) -> None:
    """
    Plot a flood hazard raster with a custom blue colormap, basemap overlay,
    legend, and export to file.

    Parameters
    ----------
    config : NetworkConfig
        Must provide:
        - figure_path : Path to save the output figure
        - show_figures : bool, whether to display the plot

    hazard : xr.Dataset
        Flood hazard dataset containing a `band_data` variable suitable for plotting.

    Returns
    -------
    None
        Saves a PNG map and optionally displays it.
    """
    fig, ax = plt.subplots(figsize=(20, 8), facecolor="white")

    # Custom blue colormap
    flood_cmap = LinearSegmentedColormap.from_list(
        "flood_blue",
        [
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
            "#6baed6", "#4292c6", "#2171b5", "#084594"
        ],
        N=256
    )

    # Ensure CRS is defined
    if hazard.rio.crs is None:
        hazard = hazard.rio.write_crs("EPSG:4326")

    # Reproject to Web Mercator
    hazard_mercator = hazard.rio.reproject("EPSG:3857")

    # Plot the raster
    hazard_mercator.band_data.plot(
        ax=ax,
        cmap=flood_cmap,
        alpha=0.7,
        vmin=0,
        vmax=6,
        add_colorbar=False
    )

    # Basemap
    cx.add_basemap(
        ax=ax,
        source=cx.providers.OpenStreetMap.Mapnik,
        alpha=0.4,
        attribution=False
    )

    ax.axis("off")

    # Legend labels and colors
    flood_labels = ['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5m+']
    legend_colors = [flood_cmap(i / 5) for i in range(len(flood_labels))]

    legend_elements = [
        Patch(
            facecolor=legend_colors[i],
            label=flood_labels[i],
            edgecolor="navy",
            linewidth=0.5,
            alpha=0.8
        )
        for i in range(len(flood_labels))
    ]

    ax.legend(
        handles=legend_elements,
        title="Flood Depth (meters)",
        loc="lower left",
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc"
    )

    # Clean data for potential stats
    values = hazard.band_data.values
    _ = values[~np.isnan(values)]

    fig.suptitle("")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.88)

    # Save figure
    output_path = config.figure_path / "flood_depth_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if config.show_figures:
        plt.show()


def plot_snowdrift(config: NetworkConfig, country_geometry)->None:
    """
    Plot snow drift segments over a country's road network with length-based
    styling, basemap overlay, and legend.

    Parameters
    ----------
    config : NetworkConfig
        Must provide:
        - data_path : Path containing input snow drift shapefile
        - intermediate_results_path : Path containing the road network Parquet
        - figure_path : Output directory for saved figures
        - show_figures : Whether to display the plot

    country_geometry : GeoDataFrame
        Boundary geometry of the country to plot behind the road and snow drift data.

    Returns
    -------
    None
        Saves a PNG figure and optionally displays it.
    """
    # Load inputs
    snow_drift = gpd.read_file(config.data_path / "snezni_nanosi_studije.shp")
    baseline_roads = gpd.read_parquet(
        config.intermediate_results_path / "PERS_directed_final.parquet"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")

    # Reproject to Web Mercator for basemap compatibility
    country_mercator = country_geometry.to_crs(3857)
    roads_mercator = baseline_roads.to_crs(3857)
    drift_mercator = snow_drift.to_crs(3857)

    # Length bins (km)
    bins = [0, 0.5, 1, 2, 5, float("inf")]
    labels = ["< 0.5 km", "0.5-1 km", "1-2 km", "2-5 km", "> 5 km"]

    # Classify drift lengths
    drift_mercator["length_class"] = pd.cut(
        drift_mercator["dužina_sn"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Colors and line widths for each class
    colors = {
        "< 0.5 km": "#deebf7",
        "0.5-1 km": "#9ecae1",
        "1-2 km": "#4292c6",
        "2-5 km": "#2171b5",
        "> 5 km": "#084594",
    }
    linewidths = {
        "< 0.5 km": 1.0,
        "0.5-1 km": 1.5,
        "1-2 km": 2.0,
        "2-5 km": 3.0,
        "> 5 km": 4.0,
    }

    # Base layers
    country_mercator.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#333333",
        linewidth=1.5,
        zorder=2
    )

    roads_mercator.plot(
        ax=ax,
        color="black",
        linewidth=0.4,
        alpha=0.5,
        zorder=2
    )

    # Snow drift lines per length class
    for label in labels:
        subset = drift_mercator[drift_mercator["length_class"] == label]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=colors[label],
                linewidth=linewidths[label],
                alpha=0.9,
                zorder=3
            )

    # Basemap
    cx.add_basemap(
        ax=ax,
        source=cx.providers.OpenStreetMap.Mapnik,
        alpha=0.3,
        attribution=False
    )

    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1, label="Road Network", alpha=0.6)
    ] + [
        Line2D(
            [0], [0],
            color=colors[label],
            linewidth=linewidths[label],
            label=f"{label} (n={len(drift_mercator[drift_mercator['length_class'] == label])})"
        )
        for label in labels
    ]

    ax.legend(
        handles=legend_elements,
        title="Snow Drift Length",
        loc="upper right",
        fontsize=12,
        title_fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc"
    )

    # Save output
    output_path = config.figure_path / "snow_drift_map.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if config.show_figures:
        plt.show()

    


def plot_landslides(config: NetworkConfig, country_geometry)->None:
    """
    Plot landslide events classified by year, with a basemap, visual styling,
    and legend.

    Parameters
    ----------
    config : NetworkConfig
        Must provide:
        - data_path : Directory containing 'Nestabilne_pojave.shp'
        - figure_path : Directory to save the output figure
        - show_figures : Whether to display the plot

    country_geometry : GeoDataFrame
        Boundary geometry of the country used as a background layer.

    Returns
    -------
    None
        Saves a PNG figure and optionally displays it.
    """
    # Load landslide point data
    landslides = gpd.read_file(config.data_path / "Nestabilne_pojave.shp")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")

    # Reproject to Web Mercator
    country_mercator = country_geometry.to_crs(3857)
    landslides_mercator = landslides.to_crs(3857)

    # Convert date → year
    landslides_mercator["year"] = pd.to_datetime(
        landslides_mercator["datum_evid"], errors="coerce"
    ).dt.year

    # Year bins
    bins = [0, 2000, 2010, 2015, 2020, 2030]
    labels = ["< 2000", "2000-2010", "2010-2015", "2015-2020", "2020-2025"]

    landslides_mercator["year_class"] = pd.cut(
        landslides_mercator["year"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Color scheme for age classes
    colors = {
        "< 2000": "#fde0dd",
        "2000-2010": "#fa9fb5",
        "2010-2015": "#f768a1",
        "2015-2020": "#c51b8a",
        "2020-2025": "#7a0177",
    }

    # Draw background country outline
    country_mercator.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#333333",
        linewidth=1.5,
        zorder=2
    )

    # Plot landslide points by class
    for label in labels:
        subset = landslides_mercator[
            landslides_mercator["year_class"] == label
        ]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=colors[label],
                markersize=40,
                edgecolor="#333333",
                linewidth=0.3,
                alpha=0.8,
                zorder=3
            )

    # Add basemap
    cx.add_basemap(
        ax=ax,
        source=cx.providers.OpenStreetMap.Mapnik,
        alpha=0.3,
        attribution=False
    )

    ax.set_aspect("equal")
    ax.axis("off")

    # Legend entries
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=colors[label],
            markersize=10,
            markeredgecolor="#333333",
            markeredgewidth=0.3,
            label=f"{label} (n={len(landslides_mercator[landslides_mercator['year_class'] == label])})"
        )
        for label in labels
    ]

    ax.legend(
        handles=legend_elements,
        title="Year Recorded",
        loc="upper right",
        fontsize=12,
        title_fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc"
    )

    # Save figure
    output_path = config.figure_path / "landslides_map_by_year.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if config.show_figures:
        plt.show()




def main():
    """
    Main function to orchestrate all accessibilty calculations.
    """
    # Initialize configuration
    config = NetworkConfig()

    #load flood hazard data and country outline
    hazard_country, country_geometry = load_data(config)

    #plot and save flood hazard map
    plot_flood_hazard_map(config, hazard_country)

    #plot and save snowdrift hazard map
    plot_snowdrift(config, country_geometry)

    #plot and save landslide hazard map
    plot_landslides(config, country_geometry)


if __name__ == "__main__":
    main()