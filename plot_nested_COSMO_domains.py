# Description: Plot nested COSMO domains
#
# Author: Christian R. Steger, October 2022

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from cmcrameri import cm

mpl.style.use("classic")
 
# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

###############################################################################
# Load data (elevation, coordinates, rotated coordinate system)
###############################################################################

# Notes: load required input data either from EXTPAR NetCDF files or from
#        *c.nc files with constant fields that are created during the
#        COSMO run

# # From EXTPAR file (select relevant subdomain)
# files = {"coarse": {"path": "/Users/csteger/Desktop/HSURF/"
#                             + "extpar_euro_12km_771x771.nc",
#                     "slice_rlon": slice(172, 533),
#                     "slice_rlat": slice(156, 517)},
#          "fine":   {"path": "/Users/csteger/Desktop/HSURF/"
#                             + "extpar_euro_2km_2313x2313.nc",
#                     "slice_rlon": slice(0, 1542),
#                     "slice_rlat": slice(0, 1542)}}

# From COSMO output file (*c.nc)
files = {"coarse": {"path": "/Users/csteger/Desktop/HSURF/"
                            + "lffd20210522000000c_lm_c.nc",
                    "slice_rlon": slice(None, None),
                    "slice_rlat": slice(None, None)},
         "fine":   {"path": "/Users/csteger/Desktop/HSURF/"
                            + "lffd20210522000000c_lm_f.nc",
                    "slice_rlon": slice(None, None),
                    "slice_rlat": slice(None, None)}}

# Load data
data = {}
for i in list(files.keys()):
    ds = xr.open_dataset(files[i]["path"])
    ds = ds.isel(rlon=files[i]["slice_rlon"], rlat=files[i]["slice_rlat"])
    data[i] = {"elev": ds["HSURF"].values.squeeze(),
               "lsm":  ds["FR_LAND"].values.squeeze(),
               "rlon": ds["rlon"].values,
               "rlat": ds["rlat"].values,
               "rot_pole_lat": ds["rotated_pole"].grid_north_pole_latitude,
               "rot_pole_lon": ds["rotated_pole"].grid_north_pole_longitude}
    ds.close()

###############################################################################
# Map plot
###############################################################################

# Load NCL colormap (optional; comment out in case NCL colormap is not used)
file = "/Users/csteger/Desktop/OceanLakeLandSnow.rgb"
# Source of NCL rgb-file:
# https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
rgb = np.loadtxt(file, comments=("#", "ncolors"))
if rgb.max() > 1.0:
    rgb /= 255.0
print("Number of colors: " + str(rgb.shape[0]))
cmap_ncl = mpl.colors.LinearSegmentedColormap.from_list("OceanLakeLandSnow",
                                                        rgb, N=rgb.shape[0])

# Colormap for terrain
# cmap_ter, min_v, max_v = plt.get_cmap("terrain"), 0.25, 1.00  # matplotlib
# cmap_ter, min_v, max_v = cm.fes, 0.50, 1.00                   # crameri
cmap_ter, min_v, max_v = cmap_ncl, 0.05, 1.00                 # NCL
cmap_ter = mpl.colors.LinearSegmentedColormap.from_list(
    "trunc({n},{a:.2f},{b:.2f})".format(
        n=cmap_ter.name, a=min_v, b=max_v),
    cmap_ter(np.linspace(min_v, max_v, 100)))
levels_ter = np.arange(0.0, 3000.0, 200.0)
norm_ter = mpl.colors.BoundaryNorm(levels_ter, ncolors=cmap_ter.N,
                                   extend="max")

# Color for sea/ocean (water)
cmap_sea = mpl.colors.ListedColormap(["lightskyblue"])
bounds_sea = [0.5, 1.5]
norm_sea = mpl.colors.BoundaryNorm(bounds_sea, cmap_sea.N)

# Domain labels
lab = {"coarse": {"txt": r'$\Delta$x = 12 km', "offset": (3.6, -1.2)},
       "fine":   {"txt": r'$\Delta$x = 2.2 km', "offset": (3.6, -1.2)}}

# Inner domain (without boundary relaxation zone) (optional)
brz_w = {"coarse": 15, "fine": 80}
plot_brz = False

# Map plot
crs_map = ccrs.RotatedPole(pole_latitude=data["coarse"]["rot_pole_lat"],
                           pole_longitude=data["coarse"]["rot_pole_lon"])
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 2, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.04,
                       width_ratios=[1.0, 0.025],
                       height_ratios=[0.3, 1.0, 0.3])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[:, 0], projection=crs_map)
for i in list(files.keys()):
    crs_rot = ccrs.RotatedPole(pole_latitude=data[i]["rot_pole_lat"],
                               pole_longitude=data[i]["rot_pole_lon"])
    # -------------------------------------------------------------------------
    land = cfeature.NaturalEarthFeature("physical", "land", scale="50m",
                                        edgecolor="black",
                                        facecolor="lightgray")
    ax.add_feature(land, zorder=1)
    # -------------------------------------------------------------------------
    rlon, rlat = data[i]["rlon"], data[i]["rlat"]
    data_plot = np.ones_like(data[i]["lsm"])
    plt.pcolormesh(rlon, rlat, data_plot, transform=crs_rot, shading="auto",
                   cmap=cmap_sea, norm=norm_sea, zorder=2, rasterized=True)
    data_plot = np.ma.masked_where(data[i]["lsm"] < 0.5, data[i]["elev"])
    plt.pcolormesh(rlon, rlat, data_plot, transform=crs_rot, shading="auto",
                   cmap=cmap_ter, norm=norm_ter, zorder=3, rasterized=True)
    # -------------------------------------------------------------------------
    dx_h = np.diff(rlon).mean() / 2.0
    x = [rlon[0] - dx_h, rlon[-1] + dx_h, rlon[-1] + dx_h, rlon[0] - dx_h]
    dy_h = np.diff(rlat).mean() / 2.0
    y = [rlat[0] - dy_h, rlat[0] - dy_h, rlat[-1] + dy_h, rlat[-1] + dy_h]
    poly = plt.Polygon(list(zip(x, y)), facecolor="none", edgecolor="black",
                       linewidth=2.5, zorder=4)
    ax.add_patch(poly)
    # -------------------------------------------------------------------------
    # Plot inner domain (without boundary relaxation zone)
    # -------------------------------------------------------------------------
    if plot_brz:
        dx_h = np.diff(rlon).mean() / 2.0
        x = [rlon[0 + brz_w[i]] - dx_h, rlon[-1 - brz_w[i]] + dx_h,
             rlon[-1 - brz_w[i]] + dx_h, rlon[0 + brz_w[i]] - dx_h]
        dy_h = np.diff(rlat).mean() / 2.0
        y = [rlat[0 + brz_w[i]] - dy_h, rlat[0 + brz_w[i]] - dy_h,
             rlat[-1 - brz_w[i]] + dy_h, rlat[-1 - brz_w[i]] + dy_h]
        poly = plt.Polygon(list(zip(x, y)), facecolor="none",
                           edgecolor="black", linestyle="--",
                           linewidth=1.0, zorder=4)
        ax.add_patch(poly)
    # -------------------------------------------------------------------------
    t = plt.text(rlon[0] + lab[i]["offset"][0],
                 rlat[-1] + lab[i]["offset"][1], lab[i]["txt"],
                 fontsize=13, fontweight="bold", horizontalalignment="center",
                 verticalalignment="center", transform=crs_rot, zorder=6)
    t.set_bbox(dict(facecolor="white", edgecolor="none"))
# -----------------------------------------------------------------------------
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.6, color="gray",
                  draw_labels=True, alpha=0.5, linestyle="-",
                  x_inline=False, y_inline=False, zorder=5)
gl_spac = 5  # grid line spacing for map plot [degree]
gl.xlocator = mticker.FixedLocator(range(-180, 180 + gl_spac, gl_spac))
gl.ylocator = mticker.FixedLocator(range(-90, 90 + gl_spac, gl_spac))
gl.right_labels, gl.top_labels = False, False
# -----------------------------------------------------------------------------
ext_dom = 2.0  # increase map extent [degree]
ax.set_extent([data["coarse"]["rlon"][0] - ext_dom,
               data["coarse"]["rlon"][-1] + ext_dom,
               data["coarse"]["rlat"][0] - ext_dom,
               data["coarse"]["rlat"][-1] + ext_dom], crs=crs_map)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[1:2, 1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap_ter, norm=norm_ter,
                               ticks=levels_ter, orientation="vertical")
plt.ylabel("Elevation [m]", labelpad=8.0)
# -----------------------------------------------------------------------------
fig.savefig("/Users/csteger/Desktop/COSMO_nested_domains.pdf", dpi=300,
            bbox_inches="tight")
plt.close(fig)
