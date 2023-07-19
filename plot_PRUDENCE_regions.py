# Description: Plot PRUDENCE regions on a map with terrain from COSMO
#              simulation and optionally compute (binary) masks for region(s)
#
# Required conda environment:
# conda create -n plot_env numpy matplotlib cartopy xarray cmcrameri
# -c conda-forge
#  -> additionally, the package 'utilities' has to be installed:
#     https://github.com/ChristianSteger/Utilities
#
# Author: Christian R. Steger, July 2023

# Load modules
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import xarray as xr
from cmcrameri import cm
from utilities.grid import polygon_rectangular
from utilities.plot import truncate_colormap
from utilities.grid import coord_edges
from utilities.grid import polygon_inters_exact
# from shapely.ops import transform
# from pyproj import CRS, Transformer

mpl.style.use("classic")

###############################################################################
# Settings
###############################################################################

# PRUDENCE regions (full name, (west, east, south, north))
regions = {
    "BI": ("British Isles",     (-10.0, 2.0,  50.0, 59.0)),
    "IP": ("Iberian Peninsula", (-10.0, 3.0,  36.0, 44.0)),
    "FR": ("France",            (-5.0,  5.0,  44.0, 50.0)),
    "ME": ("Mid-Europe",        (2.0,   16.0, 48.0, 55.0)),
    "SC": ("Scandinavia",       (5.0,   30.0, 55.0, 70.0)),
    "AL": ("Alps",              (5.0,   15.0, 44.0, 48.0)),
    "MD": ("Mediterranean",     (3.0,   25.0, 36.0, 44.0)),
    "EA": ("Eastern Europe",    (16.0,  30.0, 44.0, 55.0))
    }
# -> based on Christensen and Christensen (2007),
#    https://doi.org/10.1007/s10584-006-9210-7, Fig. 4

# Paths
file_extpar = "/Users/csteger/Dropbox/IAC/Data/Model/COSMO/EXTPAR_files/" \
              + "EURO/extpar_12km_europe_771x771.nc"
# -> CSCS: /project/pr133/extpar_crclim/crclim/extpar_12km_europe_771x771.nc
path_plot = "/Users/csteger/Desktop/PRUDENCE_regions_map.png"

###############################################################################
# Create map plot
###############################################################################

# Create polygons
poly_coords_geo = {}
for i in regions.keys():
    box = (regions[i][1][0], regions[i][1][2],
           regions[i][1][1], regions[i][1][3])
    poly_coords_geo[i] = polygon_rectangular(box, spacing=0.01)
    # grid spacing (0.01 deg is ~1km)

# Load data from EXTPAR file
ds = xr.open_dataset(file_extpar)
rlon = ds["rlon"].values
rlat = ds["rlat"].values
elevation = ds["HSURF"].values
soil_type = ds["SOILTYP"].values
elevation[soil_type == 9] = np.nan  # set water grid cells to NaN
pole_longitude = ds["rotated_pole"].grid_north_pole_longitude
pole_latitude = ds["rotated_pole"].grid_north_pole_latitude
ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pole_latitude,
                                 pole_longitude=pole_longitude)
ds.close()

# Colormap
cmap = truncate_colormap(cm.bukavu, 0.55, 1.0)
levels = np.arange(0.0, 3500.0, 250.0)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
ticks = np.arange(0.0, 3500.0, 500.0)

# Plot
fig = plt.figure(figsize=(9, 9))
gs = gridspec.GridSpec(3, 2, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.04,
                       width_ratios=[1.0, 0.03],
                       height_ratios=[0.3, 1.0, 0.3])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[:, 0], projection=ccrs_rot_pole)
ax.set_facecolor(cm.bukavu(0.4))
plt.pcolormesh(rlon, rlat, elevation, shading="auto", cmap=cmap, norm=norm)
ax.coastlines(resolution="50m", linewidth=0.5)
ax.set_aspect("auto")
for i in regions.keys():
    poly = plt.Polygon(list(zip(*poly_coords_geo[i].exterior.xy)),
                       facecolor="none",
                       edgecolor="black", linewidth=3.0, linestyle="-",
                       zorder=2, transform=ccrs.PlateCarree())
    ax.add_patch(poly)
    x, y = poly_coords_geo[i].centroid.xy
    t = plt.text(x[0], y[0], i, fontsize=11, fontweight="bold",
                 color="black", transform=ccrs.PlateCarree(), zorder=6)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="black",
                    boxstyle="round,pad=0.5"))
ax.set_extent([-18.5, 17.5, -12.5, 25.0], crs=ccrs_rot_pole)
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
fig.savefig(path_plot, dpi=300, bbox_inches="tight")
plt.close(fig)

###############################################################################
# Create (binary) masks for regions
###############################################################################

# # Compute mask for specific regions
# reg = "IP"
# project = Transformer.from_crs(CRS.from_user_input(ccrs.PlateCarree()),
#                                CRS.from_user_input(ccrs_rot_pole),
#                                always_xy=True).transform
# shp_geom_trans = transform(project, poly_coords_geo[reg])
# area_frac_exact = polygon_inters_exact(
#     *np.meshgrid(*coord_edges(rlon, rlat)), shp_geom_trans,
#     agg_cells=np.array([10, 5, 2]))
# mask_bin = (area_frac_exact >= 0.5).astype(bool)  # binary mask
#
# # Test plot
# plt.figure()
# ax = plt.axes(projection=ccrs_rot_pole)
# plt.pcolormesh(rlon, rlat, area_frac_exact, shading="auto", cmap=cm.nuuk_r)
# ax.coastlines(resolution="50m", linewidth=0.5)
# plt.colorbar()
