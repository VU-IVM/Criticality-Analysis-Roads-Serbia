import os,sys
import pickle
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import wkt
from shapely.geometry import Point
#import cftime
from pathlib import Path
from tqdm import tqdm
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For colormap
from matplotlib.lines import Line2D  # For custom legend
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator,LinearLocator
import osm_flex.extract as ex
import igraph as ig
import networkx as nx
from rasterio.enums import Resampling
from exactextract import exact_extract
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterstats import zonal_stats
import matplotlib.colors as mcolors

import string

# Define scenarios and bins
scenarios = ["15", "20", "30", "40"]
scenario_labels = {"15": "1.5", "20": "2.0", "30": "3.0", "40": "4.0"}  # for titles
bins = [10, 25, 50, 100, 150, np.inf]
labels = ["10-25", "25-50", "50-100", "100-150", "150+"]
colors = ["#b2182b", "#ef8a62", "#fddbc7","#d1e5f0", "#67a9cf"]
color_dict = dict(zip(labels, colors))
color_dict["No data"] = "white"




# ===== Cell 2 =====
data_path = Path('input_files')
figure_path = Path('figures')
intermediate_path = Path('intermediate_results')
accessibility_analysis_path = Path('accessibility_analysis')

# ===== Cell 3 =====
basins = pd.read_csv(intermediate_path / "SRB_flood_statistics_per_Basin_basins_scenario.csv")

all_basins = gpd.read_file(data_path / "hybas_eu_lev09_v1c.shp")
basins = gpd.GeoDataFrame(basins.merge(all_basins,left_on='basinID',right_on='HYBAS_ID'))

# --- Open NetCDF dataset ---
ds = xr.open_dataset(r"C:\Users\yma794\Downloads\disEnsemble_highExtremes.nc")

# Attach CRS (Lambert Azimuthal Equal Area)
ds = ds.rio.write_crs("EPSG:3035", inplace=False)
print("Raster CRS:", ds.rio.crs)

roads = gpd.read_parquet(
    r"C:\Users\yma794\Documents\Serbia\Analysis - Copy2\intermediate_results\PERS_directed_final.parquet"
)

if roads.crs != "EPSG:3035":
    roads = roads.to_crs("EPSG:3035")

# --- Prepare a dictionary to hold all scenarios ---
# --- Prepare a dictionary to hold all scenarios (both % change and RP shift) ---
rl_dict = {}

for scenario in scenarios:
    # Percent change (already done)
    da = ds[f"return_level_perc_chng_{scenario}"]
    sig = ds[f"significant_{scenario}"]
    da = da.where(sig == 1)
    da = da.transpose("y", "x").sortby("y")
    da = da.rio.write_crs("EPSG:3035", inplace=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    rl_dict[f"perc_chng_{scenario}"] = da

    # Return period shift
    rp = ds[f"baseline_rp_shift_{scenario}"]
    rp = rp.transpose("y", "x").sortby("y")
    rp = rp.rio.write_crs("EPSG:3035", inplace=False)
    rp = rp.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    rl_dict[f"rp_shift_{scenario}"] = rp


for scenario in scenarios:
    da = ds[f"return_level_perc_chng_{scenario}"]
    sig = ds[f"significant_{scenario}"]

    # Mask non-significant values
    da = da.where(sig == 1)

    # Transpose and sort to match typical raster orientation
    da = da.transpose("y", "x").sortby("y")

    # Attach CRS and spatial dims
    da = da.rio.write_crs("EPSG:3035", inplace=False)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    rl_dict[scenario] = da

# --- Reproject basins to raster CRS ---
basins_3035 = basins.to_crs(rl_dict["30"].rio.crs)  # use any scenario's CRS, all are the same

print("Basins reprojected to raster CRS:", basins_3035.crs)

# Loop over scenarios and calculate basin means
for scenario in scenarios:
    print(f"Processing scenario {scenario}...")

    # Select the DataArray for this scenario
    da = rl_dict[f"rp_shift_{scenario}"]


    # Clip and calculate mean per basin
    mean_vals = []
    for _, basin in basins_3035.iterrows():
        masked = da.rio.clip([basin.geometry], basins_3035.crs, drop=False)
        vals = masked.values
        vals = vals[~np.isnan(vals)]
        mean_vals.append(vals.mean() if vals.size else np.nan)

    # Save results in basins_3035
    basins_3035[f"rp{scenario}_mean"] = mean_vals

    # Print min/max to adjust bins
    min_val = np.nanmin(mean_vals)
    max_val = np.nanmax(mean_vals)
    print(f"Scenario {scenario}: min={min_val:.2f}, max={max_val:.2f}")


for scenario in scenarios:
    col_mean = f"rp{scenario}_mean"

    # Categorize into bins
    basins_3035[f"rp{scenario}_bin"] = pd.cut(
        basins_3035[col_mean],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Count basins per bin
    bin_counts = basins_3035[f"rp{scenario}_bin"].value_counts().sort_index()
    no_data_count = basins_3035[f"rp{scenario}_bin"].isna().sum()

    print(f"Scenario {scenario} ({scenario_labels[scenario]}°C):")
    print("Number of basins in each bin:")
    print(bin_counts)
    print(f"Number of basins with no data: {no_data_count}\n")


# Drop leftover spatial join columns if they exist
roads = roads.drop(
    columns=[c for c in ["index_right", "index_left"] if c in roads.columns],
    errors="ignore"
)

basins_3035 = basins_3035.drop(
    columns=[c for c in ["index_right", "index_left"] if c in basins_3035.columns],
    errors="ignore"
)

# Start from roads
roads_rp = roads.copy()

# One spatial join (no scenario-specific columns yet)
roads_rp = gpd.sjoin(
    roads_rp,
    basins_3035[["geometry"] + [f"rp{s}_mean" for s in scenarios]],
    how="left",
    predicate="intersects"
)

# Now create bins for each scenario
for scenario in scenarios:
    roads_rp[f"rp{scenario}_bin"] = pd.cut(
        roads_rp[f"rp{scenario}_mean"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print(f"Scenario {scenario} — roads mean:")
    print(roads_rp[f"rp{scenario}_mean"].describe())
    print("ROADS BIN")
    print(roads_rp[f"rp{scenario}_bin"].value_counts(dropna=False))


#############################################
# Plot basins
#############################################


# Reproject basins to Web Mercator for basemap
basins_3857 = basins_3035.to_crs(epsg=3857)

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 14.5))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    col_mean = f"rp{scenario}_mean"

    # Categorize into bins
    basins_3857[f"rp{scenario}_bin"] = pd.cut(
        basins_3857[col_mean],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Create plot color column, assign white to NaN
    basins_3857[f"plot_color_{scenario}"] = basins_3857[f"rp{scenario}_bin"].astype(str)
    basins_3857.loc[basins_3857[f"plot_color_{scenario}"] == "nan", f"plot_color_{scenario}"] = "No data"
    basins_3857[f"plot_color_{scenario}"] = basins_3857[f"plot_color_{scenario}"].map(color_dict)

    # Plot basins
    basins_3857.plot(
        color=basins_3857[f"plot_color_{scenario}"],
        edgecolor="black",
        linewidth=0.5,
        ax=axes[i]
    )

    # Add basemap
    cx.add_basemap(
        ax=axes[i],
        source=cx.providers.CartoDB.Positron,
        alpha=1.0,
        attribution=False
    )

    axes[i].set_title(f"Future warming scenario: {scenario_labels[scenario]}°C", fontsize=14)
    axes[i].set_axis_off()

# Create shared legend
handles = [mpatches.Patch(facecolor=color_dict[label], edgecolor="black", label=label) for label in labels + ["No data"]]
fig.legend(
    handles=handles,
    title="Future return periods of floods with a current 100 year return period",
    loc="lower center",
    ncol=len(handles),
    frameon=True,                # box around legend
    edgecolor="black",
    facecolor="white"
)



labels = string.ascii_uppercase
N = 4  # number of plots used

# Flatten axes in any order
axes_flat = axes.flatten()  

# Sort axes **top-to-bottom, left-to-right**
axes_sorted = sorted(
    axes_flat[:N],
    key=lambda ax: (-ax.get_position().y0, ax.get_position().x0)  # negative y0 for top-to-bottom
)

# Add labels
for i, ax in enumerate(axes_sorted):
    ax.text(
        0.05, 0.95, labels[i],
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
    )


plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave space at bottom for legend
plt.savefig(r"C:\Users\yma794\Documents\Serbia\Change in return period.png", dpi=300, bbox_inches="tight")
basins_3857.to_parquet(r"C:\Users\yma794\Documents\Serbia\Future Floods change in RP.parquet")
#plt.show()






#############################################
# Plot effect on roads
#############################################


# Reproject basins to Web Mercator for basemap
roads_rp = roads_rp.to_crs(epsg=3857)
# Reproject basins to Web Mercator for basemap
roads = roads.to_crs(epsg=3857)

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 14.5))
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    col_mean = f"rp{scenario}_mean"

    """
    # Categorize into bins
    basins_3857[f"rp{scenario}_bin"] = pd.cut(
        basins_3857[col_mean],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    """

    # Create plot color column, assign white to NaN
    roads_rp[f"plot_color_{scenario}"] = roads_rp[f"rp{scenario}_bin"].astype(str)
    roads_rp.loc[roads_rp[f"plot_color_{scenario}"] == "nan", f"plot_color_{scenario}"] = "No data"
    roads_rp[f"plot_color_{scenario}"] = roads_rp[f"plot_color_{scenario}"].map(color_dict)

    #plot roads in the background in grey 
    roads.plot(
            ax=axes[i],
            color="grey",
            linewidth=0.8
        )

    # Plot roads that experience change
    roads_rp.plot(
        color=roads_rp[f"plot_color_{scenario}"],
        edgecolor="black",
        linewidth=1.2,
        ax=axes[i]
    )

    # Add basemap
    cx.add_basemap(
        ax=axes[i],
        source=cx.providers.CartoDB.Positron,
        alpha=1.0,
        attribution=False
    )

    axes[i].set_title(f"Future warming scenario: {scenario_labels[scenario]}°C", fontsize=14)
    axes[i].set_axis_off()

# Create shared legend
#legend_labels = ["-10-0", "0-10", "10-25", "25-50", "50+", "No data"]
legend_labels = ["10-25", "25-50", "50-100", "100-150", "150+"]

handles = [
    mpatches.Patch(
        facecolor=color_dict[label],
        edgecolor="black",
        label=label
    )
    for label in legend_labels
]

fig.legend(
    handles=handles,
    title="Future return periods of floods with a current 100 year return period",
    loc="lower center",
    ncol=len(handles),
    frameon=True,                # box around legend
    edgecolor="black",
    facecolor="white"
)


labels = string.ascii_uppercase
N = 4  # number of plots used

# Flatten axes in any order
axes_flat = axes.flatten()  

# Sort axes **top-to-bottom, left-to-right**
axes_sorted = sorted(
    axes_flat[:N],
    key=lambda ax: (-ax.get_position().y0, ax.get_position().x0)  # negative y0 for top-to-bottom
)

# Add labels
for i, ax in enumerate(axes_sorted):
    ax.text(
        0.05, 0.95, labels[i],
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
    )


plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave space at bottom for legend
plt.savefig(r"C:\Users\yma794\Documents\Serbia\Change in return period experienced by roads.png", dpi=300, bbox_inches="tight")
roads_rp.to_parquet(r"C:\Users\yma794\Documents\Serbia\Future Floods change in RP experienced by roads.parquet")
plt.show()