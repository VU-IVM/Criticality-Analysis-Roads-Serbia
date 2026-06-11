"""
Fix: change smer_gdf1 of segment A5916 from 'L' to 'O'.

A5916 (petlja Vrnjačka Banja → Vrnjačka Banja start of motorway) is a
bidirectional road section incorrectly marked as one-way ('L').  This causes
node 1501 to become a topological sink in the directed network, disconnecting
the entire dead-end chain from the strongly-connected giant component.

Changing to 'O' (bidirectional) causes the 1b processing code to:
  1. Halve the AADT for this segment (treating the stored value as the
     two-direction sum, consistent with all other 'O' segments).
  2. Create a reverse edge automatically, resolving the dead-end.
"""

from pathlib import Path
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parents[1]
path = BASE_DIR / "input_files" / "roads_serbia_original_full_AADT.parquet"

roads = gpd.read_parquet(path)

mask = roads["oznaka_deo"] == "A5916"
n = mask.sum()
if n == 0:
    print("ERROR: no rows found with oznaka_deo == 'A5916'. Check the segment code.")
else:
    print(f"Found {n} row(s) for A5916:")
    print(roads.loc[mask, ["oznaka_deo", "smer_gdf1"]].to_string())
    roads.loc[mask, "smer_gdf1"] = "O"
    print(f"\nChanged smer_gdf1 to 'O'.")
    roads.to_parquet(path)
    print("Saved.")
