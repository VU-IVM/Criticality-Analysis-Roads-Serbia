# Converted from 5a_MainNetwork_Hazard_Criticality.ipynb


# ===== Cell 1 =====
# Standard library
import os
import re
import sys
import warnings
from pathlib import Path

# Third-party - Data and scientific computing
import contextily as cx
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from tqdm import tqdm
from damagescanner.core import DamageScanner

#  Shapely-specific imports for spatial analysis
import shapely
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, snap

# Matplotlib-specific imports for figures
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ===== Cell 2 =====
data_path = Path('input_files')
figure_path = Path('figures')
intermediate_path = Path('intermediate_results')

# ===== Cell 3 =====
gdf_results = gpd.read_parquet(intermediate_path / "criticality_results.parquet")

# ===== Cell 4 =====
# Load country outline
world_path =  data_path / "ne_10m_admin_0_countries.shp"
world = gpd.read_file(world_path)
country_plot = world.loc[world.SOV_A3 == 'SRB']

# read flood data
flood_path = data_path / "Europe_RP100_filled_depth.tif"

country_bounds = world.loc[world.SOV_A3 == 'SRB'].bounds
country_geom = world.loc[world.SOV_A3 == 'SRB'].geometry

hazard_map = xr.open_dataset(flood_path, engine="rasterio")
hazard_country = hazard_map.rio.clip_box(minx=country_bounds.minx.values[0],
                 miny=country_bounds.miny.values[0],
                 maxx=country_bounds.maxx.values[0],
                 maxy=country_bounds.maxy.values[0]
                ).load()

exposed_roads = DamageScanner(
    hazard_country, 
    gdf_results, 
    curves=pd.DataFrame(), 
    maxdam=pd.DataFrame()
).exposure(asset_type='roads',disable_progress=False)


# ===== Cell 5 =====
def flagged_exposed_segments(row):
        return any(val > 0.25 for val in row['values'])

def max_depth(row):
        return np.max(row['values'])

exposed_roads['exposed'] = exposed_roads.progress_apply(flagged_exposed_segments, axis=1)
exposed_roads['max_depth'] =  exposed_roads.progress_apply(max_depth, axis=1)

# ===== Cell 6 =====
gdf_vhl_flooded = gdf_results.merge(exposed_roads.loc[exposed_roads.exposed][['coverage','values','max_depth']],left_index=True,right_index=True)

# ===== Cell 7 =====
gdf_vhl_flooded

# ===== Cell 8 =====
# Define bins based on vehicle hours lost distribution - heavily skewed toward 0
bins = [0, 1000, 5000, 10000, 25000, np.inf]
labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']

# Create binned column
gdf_vhl_flooded['vhl_class'] = pd.cut(
    gdf_vhl_flooded['vhl'], 
    bins=bins, 
    labels=labels, 
    include_lowest=True
)

# Define line widths for each class (higher VHL = thicker lines)
linewidth_map = {
    '0-1K': 0.5,
    '1K-5K': 1.0,
    '5K-10K': 2.0,
    '10K-25K': 3.5,
    '25K+': 5.0
}

# Create a linewidth column
gdf_vhl_flooded['linewidth'] = gdf_vhl_flooded['vhl_class'].map(linewidth_map)

# Create figure with high DPI for crisp visuals
fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='white')

# Plot each class separately with both width and color variation
# Using red-orange color progression for impact severity
colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']

for i, (class_name, width) in enumerate(linewidth_map.items()):
    subset = gdf_vhl_flooded[gdf_vhl_flooded['vhl_class'] == class_name]
    if not subset.empty:
        subset.to_crs(3857).plot(
            ax=ax,
            color=colors[i],
            linewidth=width,
            alpha=0.8,
            label=class_name
        )

# Add basemap with optimal styling
cx.add_basemap(ax=ax,
    source=cx.providers.CartoDB.Positron,
                alpha=0.4, 
                attribution=False)


# Enhance the plot styling
ax.set_aspect('equal')
ax.axis('off')

# Create custom legend with line samples that show both width and color
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=width, label=f'{class_name} vehicle hours')
                  for i, (class_name, width) in enumerate(linewidth_map.items())]

legend = ax.legend(handles=legend_elements, 
                  title='Vehicle Hours Lost', 
                  loc='upper right',
                  fontsize=9,
                  title_fontsize=10,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  framealpha=0.9,
                  facecolor='white',
                  edgecolor='#cccccc')

# Enhance overall plot appearance
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.94)
plt.savefig(figure_path / 'vehicle_hours_lost_map_flooded.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== Cell 9 =====
snow_drift = gpd.read_file(data_path / "snezni_nanosi_studije.shp")

# ===== Cell 10 =====
gdf_vhl_snowdrift = gdf_results[['from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
       'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
       'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry', 'id',
       'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
       'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'speed', 'fft','edge_no', 'vhl']].sjoin(snow_drift)

# ===== Cell 11 =====
landslides = gpd.read_file(data_path / "Nestabilne_pojave.shp")
landslides.geometry = landslides.geometry.buffer(10)

# ===== Cell 12 =====
gdf_vhl_landslides = gdf_results[['from_id', 'to_id', 'objectid', 'oznaka_deo', 'smer_gdf1', 'kategorija',
       'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
       'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry', 'id',
       'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
       'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'speed', 'fft','edge_no', 'vhl']].sjoin(landslides)

# ===== Cell 13 =====
gdf_vhl_landslides

# ===== Cell 14 =====
# pick an aggregation rule for duplicates; here we use 'max' as an example
s_depth = gdf_vhl_flooded['max_depth'].groupby(level=0).max()
s_snow  = gdf_vhl_snowdrift['dužina_sn'].groupby(level=0).max()
s_date  = gdf_vhl_landslides['datum_evid'].groupby(level=0).max()

# align everything to the base index
gdf_hazards = pd.concat([gdf_results, s_depth.rename('max_depth'),
                         s_snow.rename('dužina_sn'),
                         s_date.rename('datum_evid')], axis=1)

# convert date to string dd/mm/yyyy (NaT -> NaN -> optional empty string)
gdf_hazards['datum_evid'] = gdf_hazards['datum_evid'].dt.strftime('%d/%m/%Y')


# ===== Cell 15 =====
keep_attrs = ['oznaka_deo', 'smer_gdf1', 'kategorija',
       'oznaka_put', 'oznaka_poc', 'naziv_poce', 'oznaka_zav', 'naziv_zavr',
       'duzina_deo', 'pocetna_st', 'zavrsna_st', 'stanje', 'geometry',
       'passenger_cars', 'buses', 'light_trucks', 'medium_trucks',
       'heavy_trucks', 'articulated_vehicles', 'total_aadt', 'road_length', 'average_time_disruption', 'vhl','max_depth', 'dužina_sn', 'datum_evid']
gdf_hazards = gdf_hazards[keep_attrs]
gdf_hazards = gdf_hazards.loc[gdf_hazards[['max_depth', 'dužina_sn', 'datum_evid']].any(axis=1)]
gdf_hazards = gdf_hazards.loc[gdf_hazards['vhl'].notna()]

# ===== Cell 16 =====
mask = (
    (gdf_hazards['max_depth'].fillna(0) > 0) &
    (gdf_hazards['dužina_sn'].fillna(0) > 0) &
    (gdf_hazards['datum_evid'].notna()))

affected_all = gdf_hazards.loc[mask]

# ===== Cell 17 =====
affected_all

# ===== Cell 18 =====
### save outputs

# ===== Cell 19 =====
gdf_hazards.to_parquet(intermediate_path / "main_network_hazard_exposure.parquet")

# ===== Cell 20 =====
# Define bins and styling (shared across all three)
bins = [0, 1000, 5000, 10000, 25000, np.inf]
labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']
colors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
linewidth_map = {'0-1K': 1, '1K-5K': 1.5, '5K-10K': 2.0, '10K-25K': 3.5, '25K+': 5.0}

# Prepare all three datasets
datasets = {
    'A': ('Floods', gdf_vhl_flooded),
    'B': ('Snow Drift', gdf_vhl_snowdrift),
    'C': ('Landslides', gdf_vhl_landslides)
}

# Bin all datasets
for letter, (title, gdf) in datasets.items():
    gdf['vhl_class'] = pd.cut(gdf['vhl'], bins=bins, labels=labels, include_lowest=True)

# Create figure with 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='white')
axes = axes.flatten()  # Flatten to [ax_A, ax_B, ax_C, ax_legend]

# Convert country boundary once
serbia_mercator = country_plot.to_crs(3857)

# Plot the three hazard maps
for idx, (letter, (title, gdf)) in enumerate(datasets.items()):
    ax = axes[idx]
    gdf_mercator = gdf.to_crs(3857)
    
    # Plot country outline
    serbia_mercator.plot(ax=ax, facecolor='none', edgecolor='#333333', 
                         linewidth=1.5, zorder=1)
    
    # Plot each VHL class
    for i, (class_name, width) in enumerate(linewidth_map.items()):
        subset = gdf_mercator[gdf_mercator['vhl_class'] == class_name]
        if not subset.empty:
            subset.plot(ax=ax, color=colors[i], linewidth=width, alpha=0.8, zorder=2)
    
    # Add basemap
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, 
                   alpha=0.4, attribution=False)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add letter label
    ax.text(0.05, 0.95, f'{letter}', transform=ax.transAxes, fontsize=20, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Use the 4th panel (lower right) for the legend
ax_legend = axes[3]
ax_legend.axis('off')

# Create legend elements
legend_elements = [
    Line2D([0], [0], color=colors[i], lw=width * 1.5, label=f'{class_name} vehicle hours')
    for i, (class_name, width) in enumerate(linewidth_map.items())
]

# Add legend to the empty panel
ax_legend.legend(handles=legend_elements, 
                 title='Vehicle Hours Lost',
                 loc='center',
                 fontsize=14,
                 title_fontsize=16,
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 framealpha=0.9,
                 facecolor='white',
                 edgecolor='#cccccc')


plt.tight_layout()
plt.savefig('vhl_hazards_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== Cell 21 =====
# ============ BASELINE RESULTS ============
print("=" * 60)
print("BASELINE RESULTS - AVERAGE TIME DELAY")
print("=" * 60)

print(f"\nTotal road segments analyzed: {len(gdf_results):,}")
print(f"\nAverage Time Disruption (hours):")
print(gdf_hazards['average_time_disruption'].describe())

# Convert to minutes for interpretation
gdf_results['avg_delay_minutes'] = gdf_results['average_time_disruption'] * 60
print(f"\nAverage Time Disruption (minutes):")
print(gdf_results['avg_delay_minutes'].describe())

# Segments by delay category
print("\nSegments by delay category:")
delay_bins = [0, 0.01, 0.25, 0.5, 1, np.inf]
delay_labels = ['No Delay', '1-15 min', '15-30 min', '30-60 min', '60+ min']
gdf_results['delay_class'] = pd.cut(gdf_results['average_time_disruption'], 
                                     bins=delay_bins, labels=delay_labels, include_lowest=True)
print(gdf_results['delay_class'].value_counts().sort_index())
print("\nPercentages:")
print((gdf_results['delay_class'].value_counts(normalize=True).sort_index() * 100).round(2))

print("\n" + "=" * 60)
print("BASELINE RESULTS - VEHICLE HOURS LOST")
print("=" * 60)

print(f"\nTotal road segments: {len(gdf_results):,}")
print(f"\nVehicle Hours Lost (VHL):")
print(gdf_results['vhl'].describe())

print(f"\nTotal VHL across network: {gdf_results['vhl'].sum():,.0f} vehicle hours")
print(f"Mean VHL per segment: {gdf_results['vhl'].mean():,.2f} vehicle hours")
print(f"Median VHL per segment: {gdf_results['vhl'].median():,.2f} vehicle hours")

# Segments by VHL category
print("\nSegments by VHL category:")
vhl_bins = [0, 1000, 5000, 10000, 25000, np.inf]
vhl_labels = ['0-1K', '1K-5K', '5K-10K', '10K-25K', '25K+']
gdf_results['vhl_class'] = pd.cut(gdf_results['vhl'], bins=vhl_bins, labels=vhl_labels, include_lowest=True)
print(gdf_results['vhl_class'].value_counts().sort_index())
print("\nPercentages:")
print((gdf_results['vhl_class'].value_counts(normalize=True).sort_index() * 100).round(2))

# Top 10 most critical segments
print("\nTop 10 segments by VHL:")
print(gdf_results.nlargest(10, 'vhl')[['oznaka_deo', 'kategorija', 'vhl', 'total_aadt']].to_string())


# ============ HAZARD-SPECIFIC RESULTS ============
print("\n" + "=" * 60)
print("HAZARD-SPECIFIC RESULTS - VEHICLE HOURS LOST")
print("=" * 60)

hazard_datasets = {
    'Floods': gdf_vhl_flooded,
    'Snow Drift': gdf_vhl_snowdrift,
    'Landslides': gdf_vhl_landslides
}

# Summary comparison table
summary_data = []

for hazard_name, gdf in hazard_datasets.items():
    print(f"\n{'─' * 40}")
    print(f"{hazard_name.upper()}")
    print(f"{'─' * 40}")
    
    print(f"Affected road segments: {len(gdf):,}")
    print(f"\nVHL Statistics:")
    print(gdf_results['vhl'].describe())
    
    print(f"\nTotal VHL: {gdf_results['vhl'].sum():,.0f} vehicle hours")
    print(f"Mean VHL: {gdf_results['vhl'].mean():,.2f} vehicle hours")
    print(f"Median VHL: {gdf_results['vhl'].median():,.2f} vehicle hours")
    print(f"Max VHL: {gdf_results['vhl'].max():,.2f} vehicle hours")
    
    # By category
    gdf['vhl_class'] = pd.cut(gdf_results['vhl'], bins=vhl_bins, labels=vhl_labels, include_lowest=True)
    print(f"\nSegments by VHL category:")
    print(gdf['vhl_class'].value_counts().sort_index())
    
    # By road category if available
    if 'kategorija' in gdf_results.columns:
        print(f"\nVHL by road category:")
        print(gdf_results.groupby('kategorija')['vhl'].agg(['count', 'sum', 'mean']).round(2))
    
    # Top 5 critical segments
    print(f"\nTop 5 most critical segments:")
    cols_to_show = ['oznaka_deo', 'kategorija', 'vhl', 'total_aadt'] if 'total_aadt' in gdf_results.columns else ['oznaka_deo', 'kategorija', 'vhl']
    cols_to_show = [c for c in cols_to_show if c in gdf_results.columns]
    print(gdf_results.nlargest(5, 'vhl')[cols_to_show].to_string())
    
    # Collect for summary table
    summary_data.append({
        'Hazard': hazard_name,
        'Affected Segments': len(gdf_results),
        'Total VHL': gdf_results['vhl'].sum(),
        'Mean VHL': gdf_results['vhl'].mean(),
        'Median VHL': gdf_results['vhl'].median(),
        'Max VHL': gdf_results['vhl'].max(),
        'Segments >25K VHL': len(gdf_results[gdf_results['vhl'] >= 25000])
    })

# Create comparison summary table
print("\n" + "=" * 60)
print("HAZARD COMPARISON SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame(summary_data)
summary_df['Total VHL'] = summary_df['Total VHL'].apply(lambda x: f"{x:,.0f}")
summary_df['Mean VHL'] = summary_df['Mean VHL'].apply(lambda x: f"{x:,.2f}")
summary_df['Median VHL'] = summary_df['Median VHL'].apply(lambda x: f"{x:,.2f}")
summary_df['Max VHL'] = summary_df['Max VHL'].apply(lambda x: f"{x:,.2f}")
print(summary_df.to_string(index=False))

# ============ ADDITIONAL ANALYSIS ============
print("\n" + "=" * 60)
print("ADDITIONAL ANALYSIS")
print("=" * 60)

# Which road categories are most affected by each hazard?
print("\nRoad categories most affected (by total VHL):")
for hazard_name, gdf in hazard_datasets.items():
    if 'kategorija' in gdf_results.columns:
        top_cat = gdf_results.groupby('kategorija')['vhl'].sum().sort_values(ascending=False)
        print(f"\n{hazard_name}:")
        print(top_cat)

# Overlap analysis - are the same segments affected by multiple hazards?
print("\n" + "─" * 40)
print("OVERLAP ANALYSIS")
print("─" * 40)

# Get unique segment identifiers (assuming 'oznaka_deo' or similar exists)
if 'oznaka_deo' in gdf_vhl_flooded.columns:
    flooded_segments = set(gdf_vhl_flooded['oznaka_deo'].dropna())
    snow_segments = set(gdf_vhl_snowdrift['oznaka_deo_left'].dropna())
    landslide_segments = set(gdf_vhl_landslides['oznaka_deo_left'].dropna())
    
    print(f"Segments affected by floods only: {len(flooded_segments - snow_segments - landslide_segments)}")
    print(f"Segments affected by snow only: {len(snow_segments - flooded_segments - landslide_segments)}")
    print(f"Segments affected by landslides only: {len(landslide_segments - flooded_segments - snow_segments)}")
    print(f"Segments affected by floods AND snow: {len(flooded_segments & snow_segments)}")
    print(f"Segments affected by floods AND landslides: {len(flooded_segments & landslide_segments)}")
    print(f"Segments affected by snow AND landslides: {len(snow_segments & landslide_segments)}")
    print(f"Segments affected by ALL three hazards: {len(flooded_segments & snow_segments & landslide_segments)}")
