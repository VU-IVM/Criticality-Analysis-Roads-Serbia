from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

def read_road_network(path: Path | str) -> gpd.GeoDataFrame:
    """
    Read a vector road network from a file into a GeoDataFrame.
    Supports .parquet, .gpkg, and .shp formats.

    Parameters
    ----------
    path : Path
        Path to the input file.

    Returns
    -------
    gpd.GeoDataFrame
    """
    path = Path(path)
    
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return gpd.read_parquet(path)
    elif suffix in (".gpkg", ".shp"):
        return gpd.read_file(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported formats are: .parquet, .gpkg, .shp"
        )


def assign_flood_depth_to_roads(
    roads: gpd.GeoDataFrame,
    raster_data: np.ndarray,
    raster_bounds,
    raster_crs,
) -> gpd.GeoDataFrame:
    """
    Assign the maximum flood depth from a raster to each road segment
    and classify into depth bins.

    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Road network.
    raster_data : np.ndarray
        2D array of flood depth values.
    raster_bounds : BoundingBox
        Raster bounds (left, bottom, right, top).
    raster_crs : any
        CRS of the raster.

    Returns
    -------
    gpd.GeoDataFrame
        Roads with added columns: 'flood_depth_max' and 'flood_class'.
    """
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    from rasterstats import zonal_stats

    FLOOD_BINS   = [0, 1, 2, 3, 4, 5, np.inf]
    FLOOD_LABELS = ["0–1m", "1–2m", "2–3m", "3–4m", "4–5m", "5m+"]

    roads = roads.to_crs(raster_crs)

    rows, cols = raster_data.shape
    left, bottom, right, top = raster_bounds
    raster_transform = from_bounds(left, bottom, right, top, cols, rows)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=rows, width=cols,
            count=1, dtype=raster_data.dtype,
            crs=raster_crs, transform=raster_transform,
        ) as dataset:
            dataset.write(raster_data, 1)

        stats = zonal_stats(
            roads, memfile.name,
            stats=["max"], nodata=np.nan, all_touched=True,
        )

    roads["flood_depth_max"] = [s["max"] for s in stats]

    bin_labels = [f"{lo}–{hi}m" if hi != np.inf else "5m+"
                  for lo, hi in zip(FLOOD_BINS[:-1], FLOOD_BINS[1:])]
    roads["flood_class"] = pd.cut(
        roads["flood_depth_max"],
        bins=FLOOD_BINS,
        labels=bin_labels,
        include_lowest=True,
    ).astype(str)
    roads["flood_class"] = roads["flood_class"].where(
        roads["flood_depth_max"].notna() & (roads["flood_depth_max"] > 0),
        other="No flooding",
    )

    n_flooded = (roads["flood_class"] != "No flooding").sum()
    print(f"Roads with flood depth assigned : {n_flooded:,}")
    print(f"Roads with no flooding          : {(roads['flood_class'] == 'No flooding').sum():,}")
    print(f"Flood depth range               : {roads['flood_depth_max'].min():.2f} – {roads['flood_depth_max'].max():.2f} m")

    return roads