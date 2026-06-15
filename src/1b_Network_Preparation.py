"""
1b — Network Preparation
========================

Purpose
-------
Build the directed, AADT-attributed road network used in all downstream criticality
analyses.  Takes raw road geometry and traffic counts as input and produces a
topologically clean, directed network with travel-time attributes.

Inputs
------
- Road network parquet  (NetworkConfig.Network_PERS_Corr)
  Pre-processed Serbian road network: geometry + road attributes.
- AADT traffic data shapefile  (NetworkConfig.AADT_data)
  Official PGDS counts per road section, keyed by ``oznaka_deo``.
- Country boundaries shapefile  (NetworkConfig.world_boundaries)
  Used to exclude Kosovo road segments.

Outputs
-------
- intermediate_results/PERS_directed_final.parquet   Primary output: directed network
  with AADT, speed, and free-flow travel time.  Used by all downstream scripts.
- intermediate_results/giant_component_dropped_roads.parquet  Roads excluded by the
  giant component filter (always written, for inspection).
- figures/{traffic_type}_aadt_map.png  Individual AADT map per traffic type (6 files).
- figures/AADT_categories_combined.png  Six-panel AADT map by vehicle category.
- figures/giant_component_dropped_roads.png  Included vs. excluded roads map
  (written only when > 5% of total length is dropped).

Key Processing Steps
--------------------
1. Load & deduplicate — read road network parquet; retain relevant attribute
   columns; drop exact duplicate features.
2. Iterative snapping — snap road endpoints within 2 m of nearby endpoints or
   segments (up to 20 iterations).
3. Topology build (pass 1) — add endpoints, split edges at shared nodes
   (``split_edges_at_nodes``), assign node and edge IDs.
4. AADT merge — join traffic counts by ``oznaka_deo`` (attribute match first, then
   spatial intersection with >= 50% overlap); fill remaining gaps via neighbour
   interpolation (pass 1) and road-category median with neighbour cap (pass 2).
5. Kosovo filter — remove all segments intersecting the Kosovo boundary polygon.
6. Visualise AADT — save per-traffic-type maps and combined 6-panel figure.
7. Topology build (pass 2) — rebuild endpoints, IDs, and topology on the
   Kosovo-filtered network.
8. Directed network — halve AADT on bidirectional roads (originals are the
   two-direction sum); add reversed edges for all non-oneway roads
   (``smer_gdf1`` not in {L, D}).
9. Speed & travel time — assign speed limits by ``kategorija``; compute
   ``road_length`` (km) and ``fft`` (free-flow travel time in hours).
   Any NaN/inf fft values are flagged with a diagnostic report — no automatic
   correction is applied.
10. Giant component check — extract strongly-connected giant component; report
    dropped edges by count and length; assert >= 95% length retained; save
    dropped roads to parquet for inspection.
"""

# Standard library
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party - Data and scientific computing
import contextily as cx
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from tqdm import tqdm

# from exactextract import exact_extract
# Shapely-specific imports for spatial analysis
import shapely
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import snap

# Matplotlib-specific imports for figures
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Project root (repository root, one level above this src folder)
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd().resolve()
)
print(f"Project root set to: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local/Project imports
from src.simplify import *

from config.network_config import NetworkConfig

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# TODO: Add section for base_network_SRB_basins process and add to config file


@dataclass
class NetworkPrepConfig:
    # TODO: rename LocalConfig class consistently
    """Configuration for network preparation and analysis."""

    # Output paths
    output_path = NetworkConfig.intermediate_results_path
    figures_path = NetworkConfig.figure_path

    # Input file path
    network_input_layer = NetworkConfig.Network_PERS_Corr

    # Network snapping parameters in meters for topology errors (e.g., small gaps at intersections)
    snap_tolerance: float = 2.0
    snap_search_buffer: float = 30.0  # radius
    snap_max_iterations: int = (
        20  # maximum number of iterations to prevent infinite loops
    )

    # AADT merge parameters for spatial joins between AADT segments and road segments.
    overlap_threshold: float = 0.5  # 50% overlap required to consider a match valid
    endpoint_buffer: float = 1.0  # Buffer distance to consider roads as touching endpoints when filling missing AADT values

    # Country filtering
    exclude_country_code: str = "KOS"  # Kosovo

    # Road attributes
    road_attributes: List[str] = field(
        default_factory=lambda: [
            "objectid",
            "oznaka_deo",
            "smer_gdf1",
            "kategorija",
            "oznaka_put",
            "oznaka_poc",
            "naziv_poce",
            "oznaka_zav",
            "naziv_zavr",
            "duzina_deo",
            "pocetna_st",
            "zavrsna_st",
            "stanje",
        ]
    )

    # Traffic types
    traffic_types: List[str] = field(
        default_factory=lambda: [
            "passenger_cars",
            "buses",
            "light_trucks",
            "medium_trucks",
            "heavy_trucks",
            "articulated_vehicles",
            "total_aadt",
        ]
    )

    aadt_original_columns: List[str] = field(
        default_factory=lambda: ["PA", "BUS", "LT", "ST", "TT", "AV", "Ukupno"]
    )

    # Speed limits by road category (km/h)
    speed_limits: Dict[str, int] = field(
        default_factory=lambda: {"IM": 100, "IA": 100, "IB": 100, "IIA": 80, "IIB": 80}
    )
    default_speed: int = 80

    # Visualization parameters
    figure_dpi: int = 300
    traffic_colors: List[str] = field(
        default_factory=lambda: [
            "#005f73",
            "#9b2226",
            "#a53860",
            "#283618",
            "#2a9d8f",
            "#582f0e",
            "#001219",
        ]
    )

    # Traffic visualization breaks and labels
    breaks_labels: Dict[str, Tuple[List, List]] = field(
        default_factory=lambda: {
            "passenger_cars": (
                [0, 5000, 10000, 20000, 30000, float("inf")],
                [
                    "< 5,000",
                    "5,000-10,000",
                    "10,000-20,000",
                    "20,000-30,000",
                    "> 30,000",
                ],
            ),
            "buses": (
                [0, 50, 100, 200, 400, float("inf")],
                ["< 50", "50-100", "100-200", "200-400", "> 400"],
            ),
            "light_trucks": (
                [0, 100, 200, 400, 600, float("inf")],
                ["< 100", "100-200", "200-400", "400-600", "> 600"],
            ),
            "medium_trucks": (
                [0, 100, 200, 400, 600, float("inf")],
                ["< 100", "100-200", "200-400", "400-600", "> 600"],
            ),
            "heavy_trucks": (
                [0, 50, 100, 200, 300, float("inf")],
                ["< 50", "50-100", "100-200", "200-300", "> 300"],
            ),
            "articulated_vehicles": (
                [0, 1000, 2000, 4000, 6000, float("inf")],
                ["< 1,000", "1,000-2,000", "2,000-4,000", "4,000-6,000", "> 6,000"],
            ),
            "total_aadt": (
                [0, 5000, 10000, 20000, 40000, float("inf")],
                [
                    "< 5,000",
                    "5,000-10,000",
                    "10,000-20,000",
                    "20,000-40,000",
                    "> 40,000",
                ],
            ),
        }
    )

    @property
    def aadt_path(self) -> Path:
        """Full path to AADT data file."""
        return NetworkConfig.AADT_data

    @property
    def world_path(self) -> Path:
        """Full path to world boundaries file."""
        return NetworkConfig.world_boundaries

    def __post_init__(self):
        """Ensure output directories exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)

    def get_width_mapping(self, traffic_type: str) -> Dict[str, float]:
        """Generate width mapping for a traffic type."""
        _, labels = self.breaks_labels[traffic_type]
        return {cat: 0.5 + i * 0.75 for i, cat in enumerate(labels)}

    def get_legend_title(self, traffic_type: str) -> str:
        """Get formatted legend title for traffic type."""
        titles = {
            "passenger_cars": "Passenger Cars\n(vehicles/day)",
            "buses": "Buses\n(vehicles/day)",
            "light_trucks": "Light Trucks\n(vehicles/day)",
            "medium_trucks": "Medium Trucks\n(vehicles/day)",
            "heavy_trucks": "Heavy Trucks\n(vehicles/day)",
            "articulated_vehicles": "Articulated Vehicles\n(vehicles/day)",
            "total_aadt": "Total AADT\n(vehicles/day)",
        }
        return titles.get(traffic_type, traffic_type)


def get_endpoints(geom) -> Tuple[Optional[Point], Optional[Point]]:
    """
    Get start and end points of a linestring or multilinestring.

    Args:
        geom: Shapely geometry (LineString or MultiLineString)

    Returns:
        Tuple of (start_point, end_point), or (None, None) if invalid
    """
    if geom is None or geom.is_empty:
        return None, None

    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        if len(lines) == 0:
            return None, None
        first_line = lines[0]
        last_line = lines[-1]
        start_coords = list(first_line.coords)[0]
        end_coords = list(last_line.coords)[-1]
        return Point(start_coords), Point(end_coords)

    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        return Point(coords[0]), Point(coords[-1])

    else:
        return None, None


def load_network(config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Load Serbian road network from a GIS file path using GeoPandas.

    The file path is specified in ``config.network_input_layer`` and is read using
    :func:`geopandas.read_file`.

    Args:
        config: Network configuration
    """

    # Read into GeoDataFrame
    gdf = gpd.read_parquet(config.network_input_layer)
    print("Successfully loaded feature layer.")

    # Select relevant attributes
    attributes = config.road_attributes + ["geometry"]
    gdf = gdf[attributes]

    # Check and remove exact duplicates (including geometry)
    total_rows = len(gdf)
    duplicate_mask = gdf.duplicated(keep=False)
    unique_duplicated = gdf[duplicate_mask].drop_duplicates().shape[0]
    total_duplicates = duplicate_mask.sum()

    gdf = gdf.drop_duplicates()
    gdf = gdf.reset_index(drop=True)

    print(f"  Total rows before deduplication         : {total_rows}")
    print(f"  Unique rows that have a duplicate       : {unique_duplicated}")
    print(f"  Total rows involved in duplication      : {total_duplicates}")
    print(f"  Total rows after deduplication          : {len(gdf)}")

    return gdf


def snap_network_iteratively(
    gdf: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> gpd.GeoDataFrame:
    """
    Iteratively snap road endpoints to nearby endpoints or road segments.
    Uses spatial index for efficiency.

    Args:
        gdf: Road network GeoDataFrame
        config: Network configuration

    Returns:
        Snapped GeoDataFrame
    """
    gdf = gdf.copy()
    total_snaps = 0
    indices = list(gdf.index)
    already_snapped_pairs = set()

    iteration = 0
    while True:
        iteration += 1
        snaps_this_round = 0
        snapped_this_round = set()

        # Build spatial index fresh each iteration
        geometries = gdf.geometry.tolist()
        tree = STRtree(geometries)
        {idx: i for i, idx in enumerate(indices)}
        pos_to_idx = {i: idx for i, idx in enumerate(indices)}

        for idx1 in tqdm(indices, desc=f"Iteration {iteration}"):
            if idx1 in snapped_this_round:
                continue

            geom1 = gdf.loc[idx1, "geometry"]
            if geom1 is None or geom1.is_empty:
                continue

            start1, end1 = get_endpoints(geom1)
            if start1 is None:
                continue

            # Find candidate roads within buffer
            buffer_geom = geom1.buffer(config.snap_search_buffer)
            candidate_positions = tree.query(buffer_geom)
            candidate_indices = [pos_to_idx[pos] for pos in candidate_positions]

            for pt1, pos1 in [(start1, "start"), (end1, "end")]:
                if idx1 in snapped_this_round:
                    break

                for idx2 in candidate_indices:
                    if idx1 == idx2:
                        continue

                    # Skip if this pair was already snapped
                    pair = tuple(sorted([idx1, idx2]))
                    if pair in already_snapped_pairs:
                        continue

                    geom2 = gdf.loc[idx2, "geometry"]
                    if geom2 is None or geom2.is_empty:
                        continue

                    dist = pt1.distance(geom2)

                    if 0 < dist <= config.snap_tolerance:
                        gdf.loc[idx1, "geometry"] = snap(
                            gdf.loc[idx1, "geometry"], geom2, config.snap_tolerance
                        )
                        snapped_this_round.add(idx1)
                        already_snapped_pairs.add(pair)
                        snaps_this_round += 1

                        start1, end1 = get_endpoints(gdf.loc[idx1, "geometry"])
                        break

        print(f"Iteration {iteration} complete: {snaps_this_round} snaps")
        total_snaps += snaps_this_round

        if snaps_this_round == 0:
            break
        if iteration > config.snap_max_iterations:
            print("Max iterations reached")
            break
        print("Snapped network iteratively.")
        print(f"\nTotal snaps made: {total_snaps}")
    return gdf


def prepare_network_topology(
    pers_network: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> gpd.GeoDataFrame:
    """
    Prepare network topology by adding endpoints, splitting edges, and adding IDs.

    Args:
        pers_network: Road network
        config: Network configuration

    Returns:
        Network with proper topology
    """
    # #JDP
    # # Add osm_id column if it doesn't exist (required by simplify.py functions)
    # if 'osm_id' not in pers_network.columns:
    #     pers_network['osm_id'] = range(len(pers_network))

    # Create a Network object from the input DataFrame
    net = Network(edges=pers_network)
    net = add_endpoints(net)
    split_attributes = [attr for attr in config.road_attributes if attr != "geometry"]
    net = split_edges_at_nodes(net, attributes=split_attributes)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)

    pers_network = net.edges.set_crs(pers_network.crs)

    print(f"After topology prep, columns: {list(pers_network.columns)}")

    return pers_network


def load_aadt_data(config: NetworkPrepConfig) -> gpd.GeoDataFrame:
    """
    Load AADT (Average Annual Daily Traffic) data.

    Args:
        config: Network configuration

    Returns:
        GeoDataFrame with AADT data
    """
    aadt_network = gpd.read_file(config.aadt_path)

    print(f"Loaded AADT data with columns: {list(aadt_network.columns)}")

    # Remove rows with missing AADT data
    aadt_network.dropna(subset=config.aadt_original_columns, inplace=True)
    return aadt_network


def _nunique_deo(gdf) -> str:
    try:
        return f"{gdf['oznaka_deo'].nunique()} unique oznaka_deo"
    except (KeyError, AttributeError):
        return "? unique oznaka_deo (column missing)"


def merge_aadt_with_network(
    pers_network: gpd.GeoDataFrame,
    aadt_network: gpd.GeoDataFrame,
    config: NetworkPrepConfig,
) -> gpd.GeoDataFrame:
    """
    Merge AADT data with road network using attribute and spatial joins.

    Args:
        pers_network: Road network
        aadt_network: AADT data
        config: Network configuration

    Returns:
        Network with AADT values merged
    """
    aadt_cols = config.aadt_original_columns

    # ── STEP 0: Inputs ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 0 — Inputs")
    print("="*60)
    print(f"  pers_network : {len(pers_network)} rows | {_nunique_deo(pers_network)}")
    print(f"  aadt_network : {len(aadt_network)} rows | {_nunique_deo(aadt_network)}")
    n_input = len(pers_network)

    # ── STEP 1: Resolve oznaka_deo ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Resolve 'oznaka_deo' column")
    print("="*60)
    if "oznaka_deo" not in pers_network.columns:
        if "oznaka_deo_left" in pers_network.columns:
            pers_network = pers_network.rename(columns={"oznaka_deo_left": "oznaka_deo"})
            print("  Renamed 'oznaka_deo_left' → 'oznaka_deo'")
        elif "oznaka_deo_right" in pers_network.columns:
            pers_network = pers_network.rename(columns={"oznaka_deo_right": "oznaka_deo"})
            print("  Renamed 'oznaka_deo_right' → 'oznaka_deo'")
        else:
            raise KeyError(
                "Required column 'oznaka_deo' not found in pers_network. "
                f"Available columns: {list(pers_network.columns)}"
            )
    else:
        print("  'oznaka_deo' already present — no rename needed")

    #assing crs
    if pers_network.crs is None:
        print(f"  ⚠ pers_network has no CRS — assuming same as aadt_network ({aadt_network.crs})")
        pers_network = pers_network.set_crs(aadt_network.crs)
    elif pers_network.crs != aadt_network.crs:
        print(f"  ⚠ CRS mismatch — reprojecting pers_network from {pers_network.crs} to {aadt_network.crs}")
        pers_network = pers_network.to_crs(aadt_network.crs)

    # ── STEP 2: Attribute join + immediate deduplication ─────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Attribute join on 'oznaka_deo' (with deduplication)")
    print("="*60)
    first_merger = pers_network.merge(
        aadt_network[aadt_cols + ["oznaka_deo"]],
        how="left",
        on="oznaka_deo",
    )
    print(f"  Rows after merge (before dedup) : {len(first_merger):>6} | {_nunique_deo(first_merger)}  (input was {n_input})")

    # Deduplicate: for each original network row, keep max AADT values
    agg_dict = {col: "first" for col in pers_network.columns if col in first_merger.columns}
    agg_dict.update({col: "max" for col in aadt_cols})
    first_merger = first_merger.groupby(level=0).agg(agg_dict)
    first_merger = gpd.GeoDataFrame(first_merger, geometry="geometry")

    print(f"  Rows after dedup                : {len(first_merger):>6} | {_nunique_deo(first_merger)}")
    if len(first_merger) != n_input:
        print(f"  ⚠ Still {len(first_merger) - n_input:+d} rows vs input — investigate!")

    attr_matched_mask   = first_merger["PA"].notna()
    attr_unmatched_mask = first_merger["PA"].isna()
    n_attr_matched      = attr_matched_mask.sum()
    n_attr_unmatched    = attr_unmatched_mask.sum()
    print(f"  ✓ Matched (PA not NaN): {n_attr_matched:>6} ({100*n_attr_matched/len(first_merger):.1f}%) | {_nunique_deo(first_merger.loc[attr_matched_mask])}")
    print(f"  ✗ Unmatched (PA NaN)  : {n_attr_unmatched:>6} ({100*n_attr_unmatched/len(first_merger):.1f}%) | {_nunique_deo(first_merger.loc[attr_unmatched_mask])}")

    if n_attr_unmatched > 0:
        try:
            unmatched_keys = first_merger.loc[attr_unmatched_mask, "oznaka_deo"].unique()
            print(f"  Sample unmatched oznaka_deo values: {unmatched_keys[:5].tolist()}")
        except KeyError:
            print("  Sample unmatched oznaka_deo values: (column missing)")

    # ── STEP 3: Spatial join for unmatched rows ───────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Spatial join (intersects) on unmatched rows")
    print("="*60)
    unmatched_gdf = first_merger.loc[attr_unmatched_mask][pers_network.columns]
    overlap = unmatched_gdf.sjoin(
        aadt_network[aadt_cols + ["oznaka_deo", "geometry"]],
        how="left",
        predicate="intersects",
    )

    overlap = unmatched_gdf.sjoin(
        aadt_network[aadt_cols + ["oznaka_deo", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # Restore oznaka_deo from the left side after sjoin suffix collision
    if "oznaka_deo_left" in overlap.columns:
        overlap = overlap.rename(columns={"oznaka_deo_left": "oznaka_deo"})

    n_into_sjoin   = len(unmatched_gdf)
    n_spatial_any  = overlap["index_right"].notna().sum()
    n_spatial_none = overlap["index_right"].isna().sum()
    print(f"  Rows into sjoin          : {n_into_sjoin:>6} | {_nunique_deo(unmatched_gdf)}")
    print(f"  Rows out of sjoin        : {len(overlap):>6} | {_nunique_deo(overlap)}  (can be > input due to 1-to-many matches)")
    print(f"  ✓ Got a spatial match    : {n_spatial_any:>6} ({100*n_spatial_any/n_into_sjoin:.1f}%) | {_nunique_deo(overlap.loc[overlap['index_right'].notna()])}")
    print(f"  ✗ No spatial match at all: {n_spatial_none:>6} ({100*n_spatial_none/n_into_sjoin:.1f}%) | {_nunique_deo(overlap.loc[overlap['index_right'].isna()])}")

    # ── STEP 4: Overlap ratio filter ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 4 — Overlap ratio filter (threshold = {config.overlap_threshold})")
    print("="*60)
    overlap_with_aadt_geom = overlap.dropna(subset=["index_right"]).copy()
    overlap_with_aadt_geom["aadt_geometry"] = aadt_network.loc[
        overlap_with_aadt_geom["index_right"].astype(int), "geometry"
    ].values

    # Calculate intersection and overlap ratio
    overlap_with_aadt_geom["intersection_geom"] = overlap_with_aadt_geom.apply(
        lambda row: row["geometry"].intersection(row["aadt_geometry"]), axis=1
    )
    overlap_with_aadt_geom["overlap_ratio"] = (
        overlap_with_aadt_geom["intersection_geom"].length
        / overlap_with_aadt_geom["geometry"].length
    )

    below_threshold_mask = overlap_with_aadt_geom["overlap_ratio"] < config.overlap_threshold
    above_threshold_mask = overlap_with_aadt_geom["overlap_ratio"] >= config.overlap_threshold
    n_evaluated = len(overlap_with_aadt_geom)
    n_below = below_threshold_mask.sum()
    n_above = above_threshold_mask.sum()
    print(f"  Rows evaluated             : {n_evaluated:>6} | {_nunique_deo(overlap_with_aadt_geom)}")
    print(f"  ✓ Above threshold          : {n_above:>6} ({100*n_above/n_evaluated:.1f}%) | {_nunique_deo(overlap_with_aadt_geom.loc[above_threshold_mask])}")
    print(f"  ✗ Below threshold (dropped): {n_below:>6} ({100*n_below/n_evaluated:.1f}%) | {_nunique_deo(overlap_with_aadt_geom.loc[below_threshold_mask])}")

    # Print overlap ratio distribution to help tune the threshold
    bins   = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
    ratios = overlap_with_aadt_geom["overlap_ratio"]
    print("\n  Overlap ratio distribution:")
    for lo, hi, label in zip(bins, bins[1:], labels):
        count = ((ratios >= lo) & (ratios < hi)).sum()
        bar = "█" * int(30 * count / max(n_evaluated, 1))
        print(f"    {label:>8}  {bar:<30} {count:>5} ({100*count/n_evaluated:.1f}%)")

    overlap_filtered = overlap_with_aadt_geom[above_threshold_mask].copy()
    overlap_filtered = overlap_filtered.drop(
        columns=["aadt_geometry", "intersection_geom", "overlap_ratio"]
    )

    # ── STEP 5: Aggregate spatial matches ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Aggregate overlapping spatial matches")
    print("="*60)
    first_cols = [
        "oznaka_deo", "smer_gdf1", "kategorija", "oznaka_put",
        "oznaka_poc", "naziv_poce", "oznaka_zav", "naziv_zavr",
        "duzina_deo", "pocetna_st", "geometry",
    ]
    agg_dict = {col: "first" for col in first_cols if col in overlap_filtered.columns}
    agg_dict.update({col: "max" for col in aadt_cols})

    n_before_dropna   = len(overlap_filtered)
    overlap_for_agg   = overlap_filtered.dropna(subset=aadt_cols)
    n_dropped_no_aadt = n_before_dropna - len(overlap_for_agg)
    print(f"  Rows before dropna(aadt_cols)    : {n_before_dropna:>6} | {_nunique_deo(overlap_filtered)}")
    print(f"  ✗ Dropped (all AADT cols NaN)    : {n_dropped_no_aadt:>6} ({100*n_dropped_no_aadt/max(n_before_dropna,1):.1f}%) | {_nunique_deo(overlap_filtered.loc[~overlap_filtered.index.isin(overlap_for_agg.index)])}")

    result = overlap_for_agg.groupby(level=0).agg(agg_dict)
    print(f"  Rows after groupby (deduplicated): {len(result):>6} | {_nunique_deo(result)}")

    # ── STEP 6: Concatenate ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Concatenate attribute matches + spatial matches + leftovers")
    print("="*60)
    attr_matched_rows = first_merger.loc[first_merger.dropna(subset=aadt_cols).index]
    print(f"  Attribute-matched rows   : {len(attr_matched_rows):>6} ({100*len(attr_matched_rows)/n_input:.1f}%) | {_nunique_deo(attr_matched_rows)}")
    print(f"  Spatial-matched rows     : {len(result):>6} ({100*len(result)/n_input:.1f}%) | {_nunique_deo(result)}")

    AADT_connected = pd.concat([attr_matched_rows, result])
    n_dupes = AADT_connected.index.duplicated().sum()
    if n_dupes:
        print(f"  ⚠ Duplicate indices after first concat: {n_dupes}")

    leftovers = pers_network.loc[~pers_network.index.isin(AADT_connected.index)]
    print(f"  Leftover (no AADT at all): {len(leftovers):>6} ({100*len(leftovers)/n_input:.1f}%) | {_nunique_deo(leftovers)}")

    AADT_connected = gpd.GeoDataFrame(pd.concat([AADT_connected, leftovers]), geometry="geometry")
    print(f"  Total rows after concat  : {len(AADT_connected):>6} | {_nunique_deo(AADT_connected)}")

    # ── STEP 7: Rename + cast ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — Rename columns and cast to float64")
    print("="*60)
    column_mapping = {
        "PA": "passenger_cars", "BUS": "buses", "LT": "light_trucks",
        "ST": "medium_trucks", "TT": "heavy_trucks",
        "AV": "articulated_vehicles", "Ukupno": "total_aadt",
    }
    AADT_connected = AADT_connected.rename(columns=column_mapping)
    AADT_connected[config.traffic_types] = AADT_connected[config.traffic_types].astype(np.float64)
    AADT_connected = gpd.GeoDataFrame(AADT_connected, geometry="geometry")

    # ── STEP 8: Final summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8 — Final summary")
    print("="*60)
    print(f"  Input rows  : {n_input:>6} | {_nunique_deo(pers_network)}")
    print(f"  Output rows : {len(AADT_connected):>6} | {_nunique_deo(AADT_connected)}")
    row_diff = len(AADT_connected) - n_input
    print(f"  Row delta   : {row_diff:>+6}  {'✓ preserved' if row_diff == 0 else '⚠ mismatch'}")
    n_final_dupes = AADT_connected.index.duplicated().sum()
    if n_final_dupes:
        print(f"  ⚠ Duplicate indices in output: {n_final_dupes}")
    print()
    for col in config.traffic_types:
        n_filled = AADT_connected[col].notna().sum()
        pct = 100 * n_filled / len(AADT_connected)
        bar = "█" * int(20 * n_filled / len(AADT_connected))
        print(f"  {col:<22}: {bar:<20} {n_filled:>6} ({pct:.1f}%)")
    print("="*60 + "\n")

    return AADT_connected

def find_touching_roads_with_aadt(
    idx: int, gdf: gpd.GeoDataFrame, buffer_dist: float = 1.0
) -> List:
    """
    Find roads that touch the endpoints of a given road and have AADT values.

    Args:
        idx: Index of road to check
        gdf: GeoDataFrame with roads
        buffer_dist: Buffer distance for intersection check

    Returns:
        List of tuples (endpoint_type, road_idx, road_row)
    """
    row = gdf.loc[idx]
    start_pt, end_pt = get_endpoints(row.geometry)

    if start_pt is None:
        return []

    touching_roads = []
    for other_idx, other_row in gdf.iterrows():
        if other_idx == idx:
            continue
        if pd.isna(other_row["total_aadt"]):
            continue

        # Check if endpoints touch the other road
        if other_row.geometry is not None and not other_row.geometry.is_empty:
            if start_pt.buffer(buffer_dist).intersects(other_row.geometry):
                touching_roads.append(("start", other_idx, other_row))
            if end_pt.buffer(buffer_dist).intersects(other_row.geometry):
                touching_roads.append(("end", other_idx, other_row))

    return touching_roads


def fill_missing_aadt(
    AADT_connected: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> gpd.GeoDataFrame:
    """
    Fill missing AADT values using two-pass approach:
    1. From roads touching both endpoints
    2. From category median, capped by touching roads

    Args:
        AADT_connected: Network with partial AADT data
        config: Network configuration

    Returns:
        Network with filled AADT values
    """
    traffic_cols = config.traffic_types

    # ============================================
    # PASS 1: Fill from both endpoints touching roads with AADT
    # ============================================
    print("Pass 1: Filling from roads touching both endpoints...")

    missing_aadt = AADT_connected[AADT_connected["total_aadt"].isna()].index.tolist()
    filled_count_pass1 = 0

    for idx in tqdm(missing_aadt, total=len(missing_aadt)):
        touching = find_touching_roads_with_aadt(
            idx, AADT_connected, config.endpoint_buffer
        )

        # Check if we have at least one touch at start and one at end
        start_touches = [t for t in touching if t[0] == "start"]
        end_touches = [t for t in touching if t[0] == "end"]

        if len(start_touches) > 0 and len(end_touches) > 0:
            # Get AADT values from touching roads
            start_values = {
                col: np.mean([t[2][col] for t in start_touches]) for col in traffic_cols
            }
            end_values = {
                col: np.mean([t[2][col] for t in end_touches]) for col in traffic_cols
            }

            # Take average of start and end
            for col in traffic_cols:
                AADT_connected.loc[idx, col] = (start_values[col] + end_values[col]) / 2

            filled_count_pass1 += 1

    print(f"Pass 1 filled {filled_count_pass1} roads")

    # ============================================
    # PASS 2: Fill with median by kategorija, then cap by touching roads
    # ============================================
    print("Pass 2: Filling with kategorija median...")

    # Calculate median values per kategorija
    kategoria_medians = AADT_connected.groupby("kategorija")[traffic_cols].median()

    missing_aadt = AADT_connected[AADT_connected["total_aadt"].isna()].index.tolist()
    filled_count_pass2 = 0

    for idx in tqdm(missing_aadt, total=len(missing_aadt)):
        row = AADT_connected.loc[idx]
        kategorija = row["kategorija"]

        # Skip if no kategorija
        if pd.isna(kategorija) or kategorija not in kategoria_medians.index:
            continue

        # Fill with median values
        median_values = kategoria_medians.loc[kategorija]
        for col in traffic_cols:
            AADT_connected.loc[idx, col] = median_values[col]

        # Now check touching roads and cap if our value is higher
        touching = find_touching_roads_with_aadt(
            idx, AADT_connected, config.endpoint_buffer
        )

        if len(touching) > 0:
            # Get max AADT from any touching road
            max_touching_values = {
                col: max([t[2][col] for t in touching]) for col in traffic_cols
            }

            # Cap our values if they exceed touching roads
            for col in traffic_cols:
                if AADT_connected.loc[idx, col] > max_touching_values[col]:
                    AADT_connected.loc[idx, col] = max_touching_values[col]

        filled_count_pass2 += 1
    print(f"Pass 2 filled {filled_count_pass2} roads")

    # Summary
    remaining_missing = AADT_connected["total_aadt"].isna().sum()
    print(f"\nRemaining roads without AADT: {remaining_missing}")

    return AADT_connected


def filter_by_country(
    AADT_connected: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> gpd.GeoDataFrame:
    """
    Filter roads to exclude specified country (e.g., Kosovo).

    Args:
        AADT_connected: Network with AADT data
        config: Network configuration

    Returns:
        Filtered network
    """
    # Load country outline
    world = gpd.read_file(config.world_path)
    country = world.loc[world.SOV_A3 == config.exclude_country_code]
    country = country.to_crs(AADT_connected.crs)

    # Dissolve in case there are multiple polygons
    kosovo_geom = country.union_all()

    # Filter roads that are within Serbia (not in Kosovo)
    AADT_Serbia = AADT_connected[
        ~AADT_connected.geometry.intersects(kosovo_geom)
    ].copy()

    return AADT_Serbia


def plot_aadt_categories_combined(
    gdf_aadt: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> None:
    """
    Create combined 3x2 subplot visualization of all AADT categories.

    Args:
        gdf_aadt: Network with AADT data
        config: Network configuration
    """
    # Exclude 'total_aadt' - only 6 categories
    traffic_types = [t for t in config.traffic_types if t != "total_aadt"]
    letters = ["A", "B", "C", "D", "E", "F"]

    # Create figure with 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(12, 24), facecolor="white")
    axes = axes.flatten()

    for i, traffic_type in enumerate(traffic_types):
        ax = axes[i]
        breaks, labels = config.breaks_labels[traffic_type]
        width_mapping = config.get_width_mapping(traffic_type)

        # Create categories for this traffic type
        gdf_aadt[f"{traffic_type}_category"] = pd.cut(
            gdf_aadt[traffic_type], bins=breaks, labels=labels, include_lowest=True
        )

        # Plot each category
        for category in labels:
            subset = gdf_aadt[gdf_aadt[f"{traffic_type}_category"] == category]
            if len(subset) > 0:
                width = width_mapping[category]
                subset.plot(
                    ax=ax, color=config.traffic_colors[i], alpha=0.7, linewidth=width
                )

        cx.add_basemap(
            ax=ax,
            crs=gdf_aadt.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,
            alpha=0.3,
            attribution=False,
        )

        # Create legend with line widths
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=config.traffic_colors[i],
                lw=width_mapping[cat],
                label=cat,
                alpha=0.7,
            )
            for cat in labels
        ]
        ax.legend(
            handles=legend_elements,
            title=config.get_legend_title(traffic_type),
            loc="upper right",
            fontsize=13,
            title_fontsize=15,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="#cccccc",
        )

        ax.axis("off")

        # Add letter label
        ax.text(
            0.05,
            0.95,
            letters[i],
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        config.figures_path / "AADT_categories_combined.png",
        dpi=config.figure_dpi,
        bbox_inches="tight",
    )
    if NetworkConfig.show_figures:
        plt.show()


def plot_total_aadt_map(
    aadt_network: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> None:
    """
    Create individual map visualization for total AADT.

    Args:
        aadt_network: Original AADT network data
        config: Network configuration
    """
    # Prepare data
    gdf_aadt = aadt_network.copy()
    column_mapping = {
        "PA": "passenger_cars",
        "BUS": "buses",
        "LT": "light_trucks",
        "ST": "medium_trucks",
        "TT": "heavy_trucks",
        "AV": "articulated_vehicles",
        "Ukupno": "total_aadt",
    }
    gdf_aadt = gdf_aadt.rename(columns=column_mapping)
    gdf_aadt[config.traffic_types] = gdf_aadt[config.traffic_types].astype(np.float64)

    # Plot total AADT
    traffic_type = "total_aadt"
    breaks, labels = config.breaks_labels[traffic_type]
    width_mapping = config.get_width_mapping(traffic_type)

    # Create categories
    gdf_aadt[f"{traffic_type}_category"] = pd.cut(
        gdf_aadt[traffic_type].astype(np.float64),
        bins=breaks,
        labels=labels,
        include_lowest=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for category in labels:
        subset = gdf_aadt[gdf_aadt[f"{traffic_type}_category"] == category]
        if len(subset) > 0:
            width = width_mapping[category]
            subset.plot(
                ax=ax,
                color=config.traffic_colors[6],
                alpha=0.7,
                linewidth=width,
                label=category,
            )

    cx.add_basemap(
        ax=ax,
        crs=gdf_aadt.crs.to_string(),
        source=cx.providers.OpenStreetMap.Mapnik,
        alpha=0.3,
        attribution=False,
    )
    ax.legend(title=config.get_legend_title(traffic_type), loc="upper right")
    ax.axis("off")
    plt.savefig(
        config.figures_path / f"{traffic_type}_aadt_map_og.png",
        dpi=config.figure_dpi,
        bbox_inches="tight",
    )
    if NetworkConfig.show_figures:
        plt.show()


def plot_individual_aadt_maps(
    gdf_aadt: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> None:
    """
    Save one map per traffic type (6 types, excluding total_aadt).

    Args:
        gdf_aadt: Network with AADT data (columns already renamed to English names)
        config: Network configuration
    """
    traffic_types = [t for t in config.traffic_types if t != "total_aadt"]

    for i, traffic_type in enumerate(traffic_types):
        breaks, labels = config.breaks_labels[traffic_type]
        width_mapping = config.get_width_mapping(traffic_type)

        gdf_aadt[f"{traffic_type}_category"] = pd.cut(
            gdf_aadt[traffic_type], bins=breaks, labels=labels, include_lowest=True
        )

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for category in labels:
            subset = gdf_aadt[gdf_aadt[f"{traffic_type}_category"] == category]
            if len(subset) > 0:
                width = width_mapping[category]
                subset.plot(
                    ax=ax,
                    color=config.traffic_colors[i],
                    alpha=0.7,
                    linewidth=width,
                    label=category,
                )

        cx.add_basemap(
            ax=ax,
            crs=gdf_aadt.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,
            alpha=0.3,
            attribution=False,
        )
        ax.legend(title=config.get_legend_title(traffic_type), loc="upper right")
        ax.axis("off")
        plt.savefig(
            config.figures_path / f"{traffic_type}_aadt_map.png",
            dpi=config.figure_dpi,
            bbox_inches="tight",
        )
        if NetworkConfig.show_figures:
            plt.show()


def create_directed_network(
    AADT_Serbia: gpd.GeoDataFrame, config: NetworkPrepConfig
) -> gpd.GeoDataFrame:
    """
    Create directed network with bidirectional edges for non-oneway roads.

    Args:
        AADT_Serbia: Network filtered for Serbia
        config: Network configuration

    Returns:
        Directed network with speed and travel time attributes
    """
    # Prepare network topology
    net = Network(edges=AADT_Serbia)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)
    base_network = net.edges.set_crs(AADT_Serbia.crs)

    # Filter for roads that are not oneway
    non_oneway_mask = ~base_network["smer_gdf1"].isin(["L", "D"])

    # Diagnostic: print all smer_gdf1 values present on non-oneway roads
    smer_counts = base_network.loc[non_oneway_mask, "smer_gdf1"].value_counts(dropna=False)
    print(f"\nNon-oneway smer_gdf1 values (total {non_oneway_mask.sum()} roads):")
    for val, count in smer_counts.items():
        print(f"  {repr(val)}: {count}")

    # Halve AADT on bidirectional roads — original values are sum of both directions
    base_network.loc[non_oneway_mask, config.traffic_types] = (
        base_network.loc[non_oneway_mask, config.traffic_types] / 2
    )
    non_oneway_roads = base_network[non_oneway_mask]

    # Create reverse edges
    def reverse_road(row):
        reversed_geometry = shapely.LineString(row["geometry"].coords[::-1])
        new_row = row.copy()
        new_row["from_id"], new_row["to_id"] = row["to_id"], row["from_id"]
        new_row["geometry"] = reversed_geometry
        return new_row

    # Apply the reverse function
    reversed_edges = non_oneway_roads.apply(reverse_road, axis=1)

    # Append reversed edges back to the original GeoDataFrame
    base_network = gpd.GeoDataFrame(
        pd.concat([base_network, reversed_edges])
    ).reset_index(drop=True)
    base_network["id"] = base_network.index

    # Calculate speed and travel time
    def fill_speed(x):
        try:
            return config.speed_limits[x.kategorija]
        except Exception:
            return config.default_speed

    base_network["road_length"] = base_network.geometry.apply(
        lambda line_string: shapely.length(line_string) / 1e3
    )
    base_network["speed"] = base_network.apply(lambda x: fill_speed(x), axis=1)
    base_network["fft"] = base_network.apply(lambda x: x.road_length / x.speed, axis=1)

    bad_mask = ~np.isfinite(base_network["fft"])
    if bad_mask.any():
        raw_geom_m = base_network.loc[bad_mask].geometry.length
        detail_cols = [c for c in ["oznaka_deo", "kategorija", "stanje", "duzina_deo"] if c in base_network.columns]
        report = base_network.loc[bad_mask, detail_cols].copy()
        report["geom_length_m"] = raw_geom_m.values
        report["road_length_km"] = base_network.loc[bad_mask, "road_length"].values
        report["fft"] = base_network.loc[bad_mask, "fft"].values
        print(f"WARNING: {bad_mask.sum()} edge(s) with NaN/inf fft — inspect before proceeding:")
        print(report.to_string())
        print("  ^^^ Compare geom_length_m against duzina_deo to diagnose the source of the NaN geometry.")
    else:
        print("fft check: all edges have valid fft values.")

    return base_network


def create_igraph_and_export(
    base_network: gpd.GeoDataFrame,
    AADT_Serbia: gpd.GeoDataFrame,
    config: NetworkPrepConfig,
) -> None:
    """
    Create igraph network and export results.

    Args:
        base_network: Directed network with all attributes
        AADT_Serbia: Original Serbia network (for CRS)
        config: Network configuration
    """
    # Load into igraph
    edges = base_network.reindex(
        ["from_id", "to_id"]
        + [x for x in list(base_network.columns) if x not in ["from_id", "to_id"]],
        axis=1,
    )
    graph = ig.Graph.TupleList(
        edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:], directed=True
    )

    # Giant component check — 95% length threshold
    giant = graph.connected_components().giant()
    giant_ids = set(giant.es["id"])
    dropped_mask = ~edges["id"].isin(giant_ids)
    giant_edges = edges[~dropped_mask]
    dropped_edges = edges[dropped_mask]

    total_length = base_network["road_length"].sum()
    dropped_length = base_network.loc[dropped_mask, "road_length"].sum()
    retained_pct = 100 * (1 - dropped_length / total_length) if total_length > 0 else 100.0

    print(f"\nGiant component: {len(giant_edges)} edges retained, {len(dropped_edges)} dropped")
    print(f"  Total length    : {total_length:.1f} km")
    print(f"  Dropped length  : {dropped_length:.1f} km")
    print(f"  Retained        : {retained_pct:.2f}%")

    if retained_pct < 95.0:

        print(f"\n⚠ Only {retained_pct:.2f}% of road length retained — producing diagnostic map")

        # Save dropped roads for inspection
        dropped_gdf = base_network.loc[dropped_mask].reset_index(drop=True).set_crs(AADT_Serbia.crs)
        dropped_gdf.to_parquet(config.output_path / "giant_component_dropped_roads.parquet")
        print(f"  Saved {len(dropped_gdf)} dropped roads to giant_component_dropped_roads.parquet")

        fig, ax = plt.subplots(figsize=(14, 10))
        giant_gdf = base_network.loc[~dropped_mask].set_crs(AADT_Serbia.crs)
        giant_gdf_wm = giant_gdf.to_crs(epsg=3857)
        dropped_gdf_wm = dropped_gdf.to_crs(epsg=3857)
        giant_gdf_wm.plot(ax=ax, color="blue", linewidth=0.8, alpha=0.7, label="Giant component")
        dropped_gdf_wm.plot(ax=ax, color="red", linewidth=1.5, alpha=0.9, label="Dropped")
        try:
            cx.add_basemap(ax, crs=giant_gdf_wm.crs, source=cx.providers.CartoDB.Positron)
        except Exception:
            pass
        ax.legend()
        ax.set_title(f"Giant component: {retained_pct:.2f}% retained")
        ax.axis("off")
        plt.savefig(config.figures_path / "giant_component_dropped_roads.png", dpi=config.figure_dpi, bbox_inches="tight")
        if NetworkConfig.show_figures:
            plt.show()

    assert retained_pct >= 95.0, (
        f"Giant component retained only {retained_pct:.2f}% of road length "
        f"(threshold: 95%). Check giant_component_dropped_roads.parquet for dropped segments."
    )

    # Export to parquet
    edges_gdf = giant_edges.reset_index(drop=True).set_crs(AADT_Serbia.crs)
    edges_gdf.to_parquet(config.output_path / "PERS_directed_final.parquet")
    print(f"Directed graph saved to {(config.output_path / 'PERS_directed_final.parquet').resolve()}")


def main():
    """
    Main function to orchestrate the network preparation workflow.
    """
    # Initialize configuration
    config = NetworkPrepConfig()

    # Step 1: Load network
    print("Step 1: Loading network...")
    pers_network = load_network(config)

    # Step 2: Snap network iteratively
    print("Step 2: Snapping network...")
    pers_network = snap_network_iteratively(pers_network, config)

    # Step 3: Prepare network topology
    print("Step 3: Preparing network topology...")
    pers_network = prepare_network_topology(pers_network, config)

    # Step 4: Load AADT data
    print("Step 4: Loading AADT data...")
    aadt_network = load_aadt_data(config)

    # Step 5: Merge AADT with network
    print("Step 5: Merging AADT with network...")
    AADT_connected = merge_aadt_with_network(pers_network, aadt_network, config)

    # Step 6: Fill missing AADT values
    print("Step 6: Filling missing AADT values...")
    AADT_connected = fill_missing_aadt(AADT_connected, config)

    # Step 7: Filter by country (exclude Kosovo)
    print("Step 7: Filtering by country...")
    AADT_Serbia = filter_by_country(AADT_connected, config)

    # Step 8: Create visualizations
    print("Step 8: Creating individual AADT maps (6 traffic types)...")
    plot_individual_aadt_maps(AADT_Serbia, config)

    print("Step 9: Creating combined AADT category map...")
    plot_aadt_categories_combined(AADT_Serbia, config)

    print("Step 10: Creating total AADT map on original AADT network...")
    plot_total_aadt_map(aadt_network, config)

    # Step 11: Create directed network
    print("Step 11: Creating directed network...")
    base_network = create_directed_network(AADT_Serbia, config)

    # Step 12: Create igraph and export
    print("Step 12: Creating igraph and exporting...")
    create_igraph_and_export(base_network, AADT_Serbia, config)

    print("Network preparation completed successfully!")


if __name__ == "__main__":
    main()
