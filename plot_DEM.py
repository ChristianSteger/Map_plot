# Description: Plot Digital Elevation Model (DEM) data
#
# Required conda environment:
# conda create -n plot_env numpy matplotlib cartopy xarray cmcrameri shapely
# -c conda-forge
#  -> additionally, the package 'utilities' has to be installed:
#     https://github.com/ChristianSteger/Utilities
#
# Author: Christian R. Steger, June 2023

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import xarray as xr
from cmcrameri import cm
from shapely.geometry import Polygon
from utilities.plot import truncate_colormap
from utilities.miscellaneous import aggregation_1d, aggregation_2d
from utilities.grid import coord_edges, polygon_inters_approx

mpl.style.use("classic")
 
# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths to folders
path_dem = "/Users/csteger/Desktop/GLOBE/"
# path_dem = "/store/c2sm/extpar_raw_data/topo/globe/"  # CSCS
path_plot = "/Users/csteger/Desktop/"

###############################################################################
# Load data and create map plot
###############################################################################

# Settings
tiles_dem = ["GLOBE_B10.nc", "GLOBE_C10.nc",
             "GLOBE_F10.nc", "GLOBE_G10.nc"]  # relevant DEM tiles
sub_dom = {"lon": slice(-50.0, 75.0), "lat": slice(75.0, 25.0)}
agg_num = 5  # e.g. 1 (no spatial aggregation), 2, 5
mask_caspian_sea = True

# Load DEM data
ds = xr.open_mfdataset([path_dem + i for i in tiles_dem])
ds = ds.sel(lon=sub_dom["lon"], lat=sub_dom["lat"])
lon = ds["lon"].values  # [degree]
lat = ds["lat"].values  # [degree]
elevation = ds["altitude"].values  # [m]
ds.close()

# Spatial aggregation (-> accelerates plotting and optional masking)
if agg_num > 1:
    lon = aggregation_1d(lon, agg_num, "mean")
    lat = aggregation_1d(lat, agg_num, "mean")
    elevation = aggregation_2d(elevation, agg_num, agg_num, "mean")

# Mask Caspian Sea
if mask_caspian_sea:
    coastlines = shapereader.natural_earth(resolution="10m",
                                           category="physical",
                                           name="coastline")
    caspian_sea = list(shapereader.Reader(coastlines).records())[957]
    lon_edge, lat_edge = coord_edges(lon, lat)
    area_frac = polygon_inters_approx(lon_edge, lat_edge,
                                      Polygon(caspian_sea.geometry),
                                      num_samp=1)
    elevation[area_frac == 1.0] = np.nan

# Define map projection
# crs_map = ccrs.PlateCarree()
crs_map = ccrs.RotatedPole(pole_latitude=43.0,
                           pole_longitude=-170.0)

# Colormap
cmap = truncate_colormap(cm.bukavu, 0.5, 1.0)
levels = np.arange(0.0, 4250.0, 250.0)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
ticks = np.arange(0.0, 4500.0, 500.0)

# Map plot
fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(3, 2, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.04,
                       width_ratios=[1.0, 0.03],
                       height_ratios=[0.3, 1.0, 0.3])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[:, 0], projection=crs_map)
ax.set_facecolor(cm.bukavu(0.4))
plt.pcolormesh(lon, lat, elevation, cmap=cmap, norm=norm,
               shading="auto", rasterized=True, transform=ccrs.PlateCarree())
ax.coastlines("10m", linewidth=0.5)
ax.set_aspect("auto")
# ax.set_extent([-20.0, 50.0, 20.0, 75.0], crs=ccrs.PlateCarree())
ax.set_extent([-26.0, +29.0, -18.0, +26.0], crs=crs_map)
# -----------------------------------------------------------------------------
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.6, color="gray",
                  draw_labels=True, alpha=0.5, linestyle="-",
                  x_inline=False, y_inline=False, zorder=5)
gl_spac = 5  # grid line spacing for map plot [degree]
gl.xlocator = mticker.FixedLocator(range(-180, 180 + gl_spac, gl_spac))
gl.ylocator = mticker.FixedLocator(range(-90, 90 + gl_spac, gl_spac))
gl.right_labels, gl.top_labels = False, False
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[1:2, 1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                               ticks=ticks, orientation="vertical")
plt.ylabel("Elevation [m]", labelpad=8.0)
# -----------------------------------------------------------------------------
fig.savefig(path_plot + "DEM_map_plot.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
