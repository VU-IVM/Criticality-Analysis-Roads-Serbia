import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
from rasterio import features
from affine import Affine
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import contextily as cx
import string
import pandas as pd

# -----------------------------
# Load dataset
# -----------------------------

def max_raster_along_line(line, raster, transform):
    """
    Returns the maximum raster value intersecting a line geometry.
    """
    mask = geometry_mask(
        [mapping(line)],
        transform=transform,
        invert=True,
        out_shape=raster.shape
    )

    values = raster[mask]

    if np.all(np.isnan(values)):
        return np.nan

    return np.nanmax(values)

results = {}

for rcp in ("45", "85"):
    results[rcp] = {}
    for period in ("1", "2"):

        file_name = r"C:\Users\yma794\Downloads\BlueSpot\BlueSpot\results\rcp" + rcp + "_rx1d_change" + period + ".nc"
        ds = xr.open_dataset(
            file_name
        )

        file_name_ensamble = r"C:\Users\yma794\Downloads\BlueSpot\BlueSpot\results\rcp" + rcp + "_rx1d_change" + period + "_ensmed.nc"
        ensamble_median = xr.open_dataset(
            file_name_ensamble
        )

        roads = gpd.read_parquet(
            r"C:\Users\yma794\Documents\Serbia\Analysis - Copy2\intermediate_results\PERS_directed_final.parquet"
        )

        if roads.crs != "EPSG:4326":
            roads = roads.to_crs("EPSG:4326")



        if rcp == "45":
            rcp_text = "4.5"
        elif rcp == "85":
            rcp_text = "8.5"

        if period == "1":
            period_text = "2031-2060"
        elif rcp == "2":
            period_text = "2071-2100"

        # -----------------------------
        # Load Serbia boundary
        # -----------------------------
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        serbia = world[world.NAME == "Serbia"]

        # -----------------------------
        # Set CRS for dataset
        # -----------------------------
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        ensamble_median = ensamble_median.rio.write_crs("EPSG:4326")
        ensamble_median = ensamble_median.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        #make variable names consistent with baseline data
        ensamble_median = ensamble_median.rename({
            "Change_ensmed_T20":   "Change_T20",
            "Change_ensmed_T50":   "Change_T50",
            "Change_ensmed_T100":  "Change_T100",
            "Change_ensmed_T500":  "Change_T500",
            "Change_ensmed_T1000": "Change_T1000",
        })


        # -----------------------------
        # Variables to plot
        # -----------------------------

        variables = ["Change_T20",] #in data set: "Change_T20","Change_T50","Change_T100","Change_T500","Change_T1000",


        # -----------------------------
        # Plot setup
        # -----------------------------

        for var in variables:

            data = ds[var]
            ensemble = ensamble_median[var]

            # Count positive & negative models
            n_pos = (data > 0).sum(dim="model")
            n_neg = (data < 0).sum(dim="model")

            # Agreement classification
            agreement = xr.zeros_like(n_pos)

            # Negative agreement
            agreement = xr.where(n_neg == 8, -4, agreement)
            agreement = xr.where(n_neg == 7, -3, agreement)
            agreement = xr.where(n_neg == 6, -2, agreement)
            agreement = xr.where(n_neg == 5, -1, agreement)

            # Positive agreement
            agreement = xr.where(n_pos == 5,  1, agreement)
            agreement = xr.where(n_pos == 6,  2, agreement)
            agreement = xr.where(n_pos == 7,  3, agreement)
            agreement = xr.where(n_pos == 8,  4, agreement)

            # -----------------------------
            # Build affine transform
            # -----------------------------
            lon = agreement.longitude.values
            lat = agreement.latitude.values

            transform = (
                Affine.translation(lon[0], lat[0])
                * Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
            )

            # -----------------------------
            # Rasterize Serbia mask
            # -----------------------------
            mask = features.rasterize(
                [(geom, 1) for geom in serbia.geometry],
                out_shape=(len(lat), len(lon)),
                transform=transform,
                fill=0,
                dtype="uint8",
            )

            mask = xr.DataArray(
                mask,
                coords={"latitude": lat, "longitude": lon},
                dims=("latitude", "longitude"),
            )

            # Apply mask
            agreement_clip = agreement.where(mask == 1)

            # -----------------------------
            # Agreement percentage
            # -----------------------------
            total_cells = agreement_clip.count().item()
            good_cells = agreement_clip.where(agreement_clip != 0).count().item()
            percent_good = good_cells / total_cells * 100

            ensemble_masked = ensemble.where(agreement != 0)

            lon = ensemble.longitude.values
            lat = ensemble.latitude.values

            transform = (
                Affine.translation(lon[0], lat[0])
                * Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
            )
            

            mask = features.rasterize(
                [(geom, 1) for geom in serbia.geometry],
                out_shape=(len(lat), len(lon)),
                transform=transform,
                fill=0,
                dtype="uint8",
            )

            mask = xr.DataArray(
                mask,
                coords={"latitude": lat, "longitude": lon},
                dims=("latitude", "longitude"),
            )

            ensemble_masked = ensemble_masked.where(mask == 1)

            #convert to percent
            ensemble_masked_pct  = ensemble_masked * 100

            # Use percent-scaled, agreement-masked data
            raster = ensemble_masked_pct

            # Extract values
            raster_values = raster.values

            # Build affine transform
            lon = raster.longitude.values
            lat = raster.latitude.values

            transform = Affine.translation(lon[0], lat[0]) * Affine.scale(
                lon[1] - lon[0], lat[1] - lat[0]
            )

            roads["max_rx1day_pct"] = roads.geometry.apply(
                lambda geom: max_raster_along_line(
                    geom,
                    raster_values,
                    transform
                )
            )

            #roads_climate = roads.dropna(subset=["max_rx1day_pct"])
            roads_with_agreement = roads[roads["max_rx1day_pct"].notna()]
            roads_no_agreement = roads[roads["max_rx1day_pct"].isna()]

            results[rcp][period] = {
                "roads_with_agreement": roads_with_agreement,
                "no_agreement": roads_no_agreement,
            }

            #save result per rcp and time period 
            file_text = r"C:\Users\yma794\Documents\Serbia\change in maximum daily precipitation rcp " + rcp + " period "+ period + ".paquet"
            roads_with_agreement.to_parquet(file_text)
            

################################
# Plot the results
###############################

rcps = ["45", "85"]
periods = ["1", "2"] 

fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(6.5, 9.2),
    sharex=True, sharey=True    
)

fig.subplots_adjust(
    left=0.06,    # left margin
    right=0.98,   # right margin
    top=0.98,     # top margin
    bottom=0.10,  # leave room for colorbar
    wspace=0.02,  # horizontal space between subplots
    hspace=0.02   # vertical space between subplots
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
        fontsize=14,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
    )

    

for row, period in enumerate(periods):
    for col, rcp in enumerate(rcps):

        #reproject to match contextily
        results[rcp][period]["roads_with_agreement"] = (
            results[rcp][period]["roads_with_agreement"]
            .to_crs(epsg=3857)
        )

        results[rcp][period]["no_agreement"] = (
            results[rcp][period]["no_agreement"]
            .to_crs(epsg=3857)
        )

        ax = axes[row, col]

        # plot roads without agreement (grey, underneath)
        results[rcp][period]["no_agreement"].plot(
            ax=ax,
            color="grey",
            linewidth=0.6
        )

        print("min value in rcp ", rcp, "period ", period)
        print(results[rcp][period]["roads_with_agreement"]["max_rx1day_pct"].min())
        print("max value in rcp ", rcp, "period ", period)
        print(results[rcp][period]["roads_with_agreement"]["max_rx1day_pct"].max())

        total_roads = (
            len(results[rcp][period]["roads_with_agreement"]) +
            len(results[rcp][period]["no_agreement"])
        )

        print(f"Total roads ({rcp}, {period}):", total_roads)

        bins = [-10, -5, -2, 0, 2, 5, 10, 15, 20, 25]

        gdf = results[rcp][period]["roads_with_agreement"]

        counts = (
            gdf
            .assign(bin=pd.cut(gdf["max_rx1day_pct"], bins=bins))
            .groupby("bin")
            .size()
        )

        print(counts)

        colors = [
            "#08306b",  # strong decrease (dark blue)
            "#2171b5",  # moderate decrease
            "#6baed6",  # slight decrease (light blue)
            "#fcae91",  # small increase (light red)
            "#fb6a4a",  # moderate increase
            "#de2d26",  # strong increase
            "#a50f15",  # very strong increase
            "#770111",   # extreme increase
            "#360108"
        ]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bins, cmap.N)

        # plot roads with signal (coloured)
        results[rcp][period]["roads_with_agreement"].plot(
            ax=ax,
            column="max_rx1day_pct",
            norm=norm,
            cmap = cmap,
            linewidth=1.2,
            legend = False,
            legend_kwds={"label": "Max RX1day change (%)"}
        )

        #colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])


        ax.set_axis_off()

        # Add basemap
        cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Positron, 
                   alpha=0.4, attribution=False)



rcps_tite = ["4.5", "8.5"]
periods_title = ["2031 - 2060", "2071 - 2100"]

# Column titles
for col, rcp in enumerate(rcps_tite):
    # Position: x in figure coordinates, y at top of subplots (1 = top)
    x = 0.25 if col == 0 else 0.75  # tweak depending on layout
    fig.text(
        x, 0.98,  # slightly above top
        f"RCP {rcp}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )

# Row titles
for row, period in enumerate(periods_title):
    # Position: y in figure coordinates, x at left of subplots
    y = 0.73 if row == 0 else 0.28  # tweak depending on layout
    fig.text(
        0.02, y,
        period,
        ha="left",
        va="center",
        fontsize=12,
        rotation=90,
        fontweight="bold"
    )
        
cax = fig.add_axes([0.25, 0.06, 0.5, 0.025])

cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="horizontal"
)

cbar.set_label("Change  (%)")
cbar.set_ticks(bins)
for ax in axes.flat:
    ax.margins(0)


plt.savefig(r"C:\Users\yma794\Documents\Serbia\change in Rx1d.png", dpi=300, bbox_inches="tight")
plt.show()
