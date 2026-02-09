# Converted from 5c_CombinedClimateCriticality.ipynb


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
from pyproj import Geod
from tqdm import tqdm

# Shapely-specific imports for spatial analysis
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
gdf_hazards = gpd.read_parquet(intermediate_path / "main_network_hazard_exposure.parquet") 
base_network = gpd.read_parquet(intermediate_path / 'PERS_directed_final.parquet')

# ===== Cell 4 =====
hospital_exposed_edges = gpd.read_parquet(intermediate_path / 'hospital_impacts.parquet').to_crs(gdf_hazards.crs)
factory_exposed_edges = gpd.read_parquet(intermediate_path / 'factory_impacts.parquet').to_crs(gdf_hazards.crs)
police_exposed_edges = gpd.read_parquet(intermediate_path / 'police_impacts.parquet').to_crs(gdf_hazards.crs)
fire_exposed_edges = gpd.read_parquet(intermediate_path / 'fire_impacts.parquet').to_crs(gdf_hazards.crs)

# ===== Cell 5 =====
# Ensure gdf_hazards has a stable index to aggregate on
gdf_hazards = gdf_hazards.copy()
gdf_hazards.index.name = 'hazard_id'  # name the index for clarity

def add_impact_column(base_gdf, edges_gdf, impact_col_name, predicate="intersects", agg="mean"):
    """
    Spatially join edges to base_gdf, aggregate travel_time_impact per base index, and add as a new column.
    - predicate: 'intersects', 'within', 'contains', 'touches' (choose based on geometry semantics)
    - agg: 'mean', 'max', 'min', 'median' etc.
    """
    # Keep only needed columns to avoid bloat
    edges = edges_gdf[['travel_time_impact', 'geometry']].copy()

    # Spatial join (left frame is base), this creates potential many-to-one matches
    joined = base_gdf.sjoin(edges, how='left', predicate=predicate)

    # Aggregate per left index (hazard_id)
    # Note: joined.index is the left index; the sjoin adds right index as 'index_right'
    agg_series = joined.groupby(joined.index)['travel_time_impact'].agg(agg)

    # Attach to base_gdf with a clear name
    base_gdf[impact_col_name] = agg_series.reindex(base_gdf.index)

    return base_gdf

# Add each impact column (choose your aggregator: 'mean' or 'max')
gdf_hazards = add_impact_column(gdf_hazards, hospital_exposed_edges, 'hospital_impact', predicate='intersects', agg='mean')
gdf_hazards = add_impact_column(gdf_hazards, factory_exposed_edges,  'factory_impact',  predicate='intersects', agg='mean')
gdf_hazards = add_impact_column(gdf_hazards, police_exposed_edges,   'police_impact',   predicate='intersects', agg='mean')
gdf_hazards = add_impact_column(gdf_hazards, fire_exposed_edges,     'fire_impact',     predicate='intersects', agg='mean')

# Optional: replace NaN (no intersecting edges) with 0 hours or leave as NaN
gdf_hazards[['hospital_impact','factory_impact','police_impact','fire_impact']] = \
    gdf_hazards[['hospital_impact','factory_impact','police_impact','fire_impact']].fillna(np.nan)

# Quick sanity checks
print(gdf_hazards[['hospital_impact','factory_impact','police_impact','fire_impact']].describe())


# ===== Cell 6 =====
gdf_hazards = gdf_hazards.rename(columns = {'max_depth' : 'flood_depth', 
                              'dužina_sn' : 'snow_drift', 
                              'datum_evid' : 'landslide_date', 
                              'hospital_impact' : 'hospital_delay',
                               'factory_impact' : 'factory_delay', 
                              'police_impact' : 'police_delay',
                              'fire_impact' : 'fire_delay'})

# ===== Cell 7 =====
gdf_hazards

# ===== Cell 8 =====
# Columns after your rename
hazard_cols = ['flood_depth', 'snow_drift', 'landslide_date']            # hazard exposure
delay_cols  = ['vhl','hospital_delay', 'factory_delay', 'police_delay', 'fire_delay']  # service delays

# 1) Prepare inputs
gdf_hazards = gdf_hazards.copy()

# If landslide_date is a string date, convert to a presence indicator [0,1]
# 1 if there is a date present, else 0. If you prefer to ignore landslides, skip this step and remove from hazard_cols.
gdf_hazards['landslide_presence'] = np.where(gdf_hazards['landslide_date'].astype(str).str.len() > 0, 1.0, 0.0)

# Choose which columns to normalize
cols_to_normalize = ['flood_depth', 'snow_drift', 'landslide_presence'] + delay_cols

# Ensure columns exist and fill NaNs with 0
for c in cols_to_normalize:
    if c not in gdf_hazards.columns:
        gdf_hazards[c] = 0.0
    gdf_hazards[c] = gdf_hazards[c].astype(float).fillna(0.0)

# 2) Safe min–max normalization to [0, 1]
def safe_minmax(s):
    s = s.astype(float)
    s = s.fillna(0.0)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        # constant or all zeros, return zeros
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)

for c in cols_to_normalize:
    gdf_hazards[f'{c}_norm'] = safe_minmax(gdf_hazards[c])

# 3) Build aggregate criticality
# Option A simple unweighted average across normalized hazards and delays
norm_hazards = ['flood_depth_norm', 'snow_drift_norm', 'landslide_presence_norm']
norm_delays  = [f'{c}_norm' for c in delay_cols]
all_norm     = norm_hazards + norm_delays

# Simple average
gdf_hazards['criticality_sum'] = gdf_hazards[all_norm].sum(axis=1)

# ===== Cell 9 =====
gdf_hazards[['hospital_delay', 'factory_delay', 'police_delay', 'fire_delay']] = gdf_hazards[['hospital_delay', 'factory_delay', 'police_delay', 'fire_delay']]*60

# ===== Cell 10 =====
gdf_hazards.to_excel(intermediate_path / 'VUA_Climate_Criticality_PERS.xlsx')

# ===== Cell 11 =====
gdf_hazards.sort_values('criticality_sum',ascending=False)

# ===== Cell 12 =====
# 1) Prepare data: drop zero or missing criticality
gdf_plot = gdf_hazards.copy().to_crs(3857)
gdf_plot = gdf_plot[gdf_plot['criticality_sum'].fillna(0) > 0.1]

# 2) Define bins and labels
bins = [0, 1, 2, 3, np.inf]
labels = ['0–1', '1–2', '2–3', '>3']

# 3) Assign classes
gdf_plot['criticality_class'] = pd.cut(
    gdf_plot['criticality_sum'],
    bins=bins,
    labels=labels,
    include_lowest=False,  # (0–1], (1–2], (2–3], (3, inf)
    right=True
)

# 4) Colors (low to high) — swap to your preferred palette if needed
colors = ['#fcae91','#fb6a4a','#de2d26','#a50f15']
color_map = dict(zip(labels, colors))

# 5) Figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 9), facecolor='white')

# Base network (light grey) — plotted above basemap, below criticality
base_network.to_crs(3857).plot(ax=ax, linewidth=0.15, color='lightgrey', alpha=0.6, zorder=2)

# Plot each class
for cls in labels:
    sub = gdf_plot[gdf_plot['criticality_class'] == cls]
    if len(sub) == 0:
        continue
    sub.plot(
        ax=ax,
        color=color_map[cls],
        linewidth=1.5,
        alpha=0.95,
        zorder=4
    )

# Legend
legend_handles = [
    Patch(facecolor=color_map[lbl], edgecolor='black', linewidth=0.3, label=lbl)
    for lbl in labels
]
ax.legend(
    handles=legend_handles,
    title='Criticality (sum)',
    loc='upper right',
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    fontsize=12,
    title_fontsize=14
)

cx.add_basemap(
    ax=ax,
    source=cx.providers.CartoDB.Positron,
    alpha=0.8,       # light background
    attribution=False,
    zorder=1)


# Cosmetics
ax.set_aspect('equal')
ax.axis('off')


plt.savefig(figure_path / 'criticality_sum_map.png', dpi=300, bbox_inches='tight')


# ===== Cell 13 =====
# --- Settings ---
TOP_METHOD = "quantile"    # "quantile" or "topN" or "cutoff"
TOP_Q = 0.90               # top 10% by criticality_sum
TOP_N = 1000               # if TOP_METHOD == "topN"
CUTOFF_VAL = 3.0           # if TOP_METHOD == "cutoff" (e.g., >3 in your legend)

# --- Prep ---
gdf = gdf_hazards.copy()

gdf["len_km"] = gdf['road_length']

# Identify normalized submetrics used in the sum
norm_cols = sorted([c for c in gdf.columns if c.endswith("_norm")])
if "criticality_sum" not in gdf.columns:
    # If not present, create it as sum of all _norm columns
    gdf["criticality_sum"] = gdf[norm_cols].sum(axis=1)

# --- Select top critical segments ---
if TOP_METHOD == "quantile":
    thr = gdf["criticality_sum"].quantile(TOP_Q)
    top_mask = gdf["criticality_sum"] >= thr
elif TOP_METHOD == "topN":
    top_idx = gdf["criticality_sum"].nlargest(TOP_N).index
    top_mask = gdf.index.isin(top_idx)
elif TOP_METHOD == "cutoff":
    top_mask = gdf["criticality_sum"] > CUTOFF_VAL
else:
    raise ValueError("TOP_METHOD must be one of: quantile, topN, cutoff")

top = gdf.loc[top_mask].copy()
rest = gdf.loc[~top_mask].copy()

print(f"Total segments: {len(gdf):,}")
print(f"Top subset size: {len(top):,} ({len(top)/len(gdf)*100:.1f}%)")
print(f"Top subset total length: {top['len_km'].sum():,.1f} km")
print(f"Mean criticality_sum (top): {top['criticality_sum'].mean():.2f}")
print(f"Mean criticality_sum (rest): {rest['criticality_sum'].mean():.2f}")

# --- 1) Which road categories dominate in the top subset? ---
cat_summary_top = (top.groupby("kategorija")
                      .agg(n_segments=("kategorija", "size"),
                           length_km=("len_km", "sum"),
                           avg_crit=("criticality_sum", "mean"))
                      .sort_values(["n_segments", "length_km"], ascending=False))
cat_summary_all = (gdf.groupby("kategorija")
                      .agg(n_segments=("kategorija", "size"),
                           length_km=("len_km", "sum"))
                      .sort_values(["n_segments", "length_km"], ascending=False))

# Add shares vs. network totals
cat_summary_top["share_count_%"] = (cat_summary_top["n_segments"] / len(top) * 100).round(1)
cat_summary_top["share_length_%"] = (cat_summary_top["length_km"] / top["len_km"].sum() * 100).round(1)

print("\nTop critical subset by road category (count, length, and avg criticality):")
print(cat_summary_top.round(2).to_string())

print("\nAll segments by road category (for baseline context):")
print(cat_summary_all.round(2).to_string())

# --- 2) Which submetrics contributed most to top rankings (overall)? ---
# Contribution here is the sum of normalized values per column in the top subset.
submetric_contrib_top = top[norm_cols].sum().sort_values(ascending=False)
submetric_contrib_top_pct = (submetric_contrib_top / submetric_contrib_top.sum() * 100).round(1)
submetric_contrib_df = pd.DataFrame({
    "total_contribution": submetric_contrib_top.round(2),
    "share_%": submetric_contrib_top_pct
}).sort_values("total_contribution", ascending=False)

print("\nSubmetric contributions in top subset (overall):")
print(submetric_contrib_df.to_string())

# --- 3) Which submetric dominates within each road category (top subset)? ---
# For each category, compute contribution shares across submetrics
by_cat_contrib = (top.groupby("kategorija")[norm_cols].sum())
by_cat_contrib_share = by_cat_contrib.div(by_cat_contrib.sum(axis=1), axis=0) * 100.0
# Identify dominant submetric per category
dominant_metric_per_cat = by_cat_contrib_share.idxmax(axis=1).to_frame("dominant_submetric")
dominant_share_per_cat = by_cat_contrib_share.max(axis=1).round(1).to_frame("dominant_share_%")
dominant_cat_df = (pd.concat([dominant_metric_per_cat, dominant_share_per_cat], axis=1)
                     .sort_values("dominant_share_%", ascending=False))

print("\nDominant submetric per road category in the top subset:")
print(dominant_cat_df.to_string())

# --- 4) Quick list of top segments (IDs) with their category and contributions ---
# Show top 20 by criticality_sum with breakdown across submetrics
top_breakdown = top.loc[top["criticality_sum"].nlargest(20).index,
                        ["kategorija", "criticality_sum"] + norm_cols].round(3)
print("\nTop 20 segments by criticality_sum with submetric breakdown:")
print(top_breakdown.to_string())

# --- 5) Optional: overall shares by band used in your map legend ---
bands = pd.cut(gdf["criticality_sum"], bins=[0,1,2,3,np.inf], labels=["0–1","1–2","2–3",">3"], right=True)
band_share = bands.value_counts(normalize=True).sort_index() * 100
print("\nNetwork share by criticality_sum bands (0–1, 1–2, 2–3, >3):")
print(band_share.round(1).to_string())

# --- 6) Optional: per-category median criticality and length in the top subset ---
cat_medians = top.groupby("kategorija").agg(
    median_crit=("criticality_sum", "median"),
    length_km=("len_km", "sum")
).sort_values("median_crit", ascending=False)
print("\nPer-category median criticality and total length (top subset):")
print(cat_medians.round(2).to_string())


# ===== Cell 14 =====
# --- Vehicle Hours Lost (VHL) fun facts for the top subset ---

def _find_col(df, candidates):
    """Return the first column found in df among candidates, case-insensitive."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# 1) Identify or construct raw VHL
# Try existing raw VHL first
vhl_col = _find_col(gdf, ["VHL", "vhl", "vehicle_hours_lost", "vehicle_hours_lost_daily", "vhl_daily"])

# Else compute from delay and AADT if available
delay_col = _find_col(gdf, ["avg_delay_min", "avg_time_disruption_min", "delay_min", "avg_delay_minutes"])
aadt_col  = _find_col(gdf, ["AADT", "aadt"])

if vhl_col is None and (delay_col is not None and aadt_col is not None):
    vhl_col = "VHL_computed_daily"
    gdf[vhl_col] = (gdf[delay_col].astype(float) / 60.0) * gdf[aadt_col].astype(float)

# Fallback to normalized proxy only if nothing else is available
using_normalized_proxy = False
if vhl_col is None:
    vhl_norm_col = _find_col(gdf, ["vhl_norm"])
    if vhl_norm_col is None:
        raise ValueError("Could not find raw VHL nor ingredients to compute it, and vhl_norm is also missing.")
    vhl_col = vhl_norm_col
    using_normalized_proxy = True

# 2) Prepare series for top and rest
vhl_top = top[vhl_col].astype(float)
vhl_rest = rest[vhl_col].astype(float)
vhl_all = gdf[vhl_col].astype(float)

# 3) Core fun facts
def pct(x): 
    return f"{100.0 * x:.1f}%"

if not using_normalized_proxy:
    # Thresholds in daily vehicle-hours
    thresholds = [1000, 5000, 10000]

    share_total_vhl_top = (vhl_top.sum() / vhl_all.sum()) if vhl_all.sum() > 0 else np.nan
    med_top = np.nanmedian(vhl_top)
    p90_top = np.nanpercentile(vhl_top, 90)
    p95_top = np.nanpercentile(vhl_top, 95)
    max_top = np.nanmax(vhl_top)

    print("\n--- Vehicle hours lost: ready-to-paste statements ---")
    print(f"The top critical subset accounts for {pct(share_total_vhl_top)} of total daily vehicle hours lost on the network.")
    print(f"Within the top subset the median daily vehicle hours lost per segment is {med_top:,.0f}, with the 90th and 95th percentiles at {p90_top:,.0f} and {p95_top:,.0f}, and a maximum of {max_top:,.0f}.")

    # Counts above thresholds in top vs rest
    for thr in thresholds:
        n_top = int((vhl_top >= thr).sum())
        n_rest = int((vhl_rest >= thr).sum())
        print(f"{n_top} segments in the top subset exceed {thr:,.0f} daily vehicle hours lost "
              f"(compared with {n_rest} in the rest of the network).")

    # Which road classes most often exceed thresholds
    top_with_vhl = top.assign(_vhl=vhl_top.values).copy()
    for thr in thresholds:
        exceed = (top_with_vhl[top_with_vhl["_vhl"] >= thr]
                  .groupby("kategorija")
                  .size()
                  .sort_values(ascending=False))
        if exceed.empty:
            continue
        print(f"\nAmong top segments exceeding {thr:,.0f} daily vehicle hours lost, the most frequent categories are:")
        for cat, cnt in exceed.items():
            print(f"  {cat}: {cnt} segments")

else:
    # Normalized proxy mode: use quantiles rather than absolute thresholds
    share_total_vhl_top = (vhl_top.sum() / vhl_all.sum()) if vhl_all.sum() > 0 else np.nan
    med_top = np.nanmedian(vhl_top)
    p90_top = np.nanpercentile(vhl_top, 90)
    p95_top = np.nanpercentile(vhl_top, 95)
    max_top = np.nanmax(vhl_top)

    print("\n--- Vehicle hours lost (normalized proxy): ready-to-paste statements ---")
    print(f"The top critical subset accounts for {pct(share_total_vhl_top)} of the summed normalized vehicle hours lost metric.")
    print(f"Within the top subset the median normalized value is {med_top:,.3f}, with the 90th and 95th percentiles at {p90_top:,.3f} and {p95_top:,.3f}, and a maximum of {max_top:,.3f}.")

    # Report which categories dominate the top decile of the proxy
    q_thr = np.nanpercentile(vhl_all, 90)
    exceed = (top[vhl_col] >= q_thr)
    by_cat = top.loc[exceed].groupby("kategorija").size().sort_values(ascending=False)
    if not by_cat.empty:
        print("\nWithin the top subset, categories most represented in the top decile of the normalized vehicle hours lost proxy are:")
        for cat, cnt in by_cat.items():
            print(f"  {cat}: {cnt} segments")

# 4) Optional: Top segments by VHL for quick cross-checking
n_show = 20
cols_show = ["kategorija", "criticality_sum"]
if vhl_col not in cols_show:
    cols_show = ["kategorija", "criticality_sum", vhl_col]
top_by_vhl = top.sort_values(by=vhl_col, ascending=False).head(n_show)[cols_show]
print(f"\nTop {n_show} segments by {vhl_col}:")
print(top_by_vhl.to_string(index=True))

