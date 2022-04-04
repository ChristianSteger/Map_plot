# Description: Determine required MERIT tiles for EXTPAR run
#
# Author: Christian Steger, April 2022

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

mpl.style.use("classic")

###############################################################################
# Settings
###############################################################################

# EXTPAR domain (example: extended EAS-CORDEX 12 km; + 30 grid cells at edges)
dom = {
    "pollon": -63.70,
    "pollat": 61.00,
    "startlon_tot": -58.05,
    "startlat_tot": -27.69,
    "dlon": 0.11,
    "dlat": 0.11,
    "ie_tot": 1118,
    "je_tot": 670,
}

# Further settings
num_gc_ext = 1  # number of grid cells added ad domain edges

# Fixed settings for EXTPAR
out_rect_dom = True  # output rectangular domain
lon_all = True
# consider all tiles along longitude if domain crosses -180.0/+180.0 lon. line

###############################################################################
# Process data and plot overview
###############################################################################

# Check consistency of settings
if num_gc_ext < 1:
    raise ValueError("'num_gc_ext' must be larger than 0")
if lon_all and not out_rect_dom:
    raise ValueError("'lon_all = True' is only allowed with "
                     + "'out_rect_dom = True'")

# Coordinate systems
rot_pole_crs = ccrs.RotatedPole(pole_latitude=dom["pollat"],
                                pole_longitude=dom["pollon"])
geo_crs = ccrs.PlateCarree()

# Create rotated grid coordinates
rlon_grid_l = dom["startlon_tot"] - dom["dlon"] / 2.0
rlon_grid = np.linspace(rlon_grid_l,
                        rlon_grid_l + dom["dlon"] * dom["ie_tot"],
                        dom["ie_tot"] + 1)
rlat_grid_l = dom["startlat_tot"] - dom["dlat"] / 2.0
rlat_grid = np.linspace(rlat_grid_l,
                        rlat_grid_l + dom["dlat"] * dom["je_tot"],
                        dom["je_tot"] + 1)

# Create polygon for domain extent
poly_rlon = np.concatenate([rlon_grid[:-1],
                            np.repeat(rlon_grid[-1], dom["je_tot"]),
                            rlon_grid[::-1][:-1],
                            np.repeat(rlon_grid[0], dom["je_tot"])])
poly_rlat = np.concatenate([np.repeat(rlat_grid[0], dom["ie_tot"]),
                            rlat_grid[:-1],
                            np.repeat(rlat_grid[-1], dom["ie_tot"]),
                            rlat_grid[::-1][:-1]])

# Extend domain by certain number of grid cells
rlon_grid_ext = np.concatenate(
    [np.linspace(rlon_grid[0] - num_gc_ext * dom["dlon"],
                 rlon_grid[0] - dom["dlon"], num_gc_ext),
     rlon_grid,
     np.linspace(rlon_grid[-1] + dom["dlon"],
                 rlon_grid[-1] + num_gc_ext * dom["dlon"], num_gc_ext)]
    )
rlat_grid_ext = np.concatenate(
    [np.linspace(rlat_grid[0] - num_gc_ext * dom["dlat"],
                 rlat_grid[0] - dom["dlat"], num_gc_ext),
     rlat_grid,
     np.linspace(rlat_grid[-1] + dom["dlat"],
                 rlat_grid[-1] + num_gc_ext * dom["dlat"], num_gc_ext)]
    )
rlon_grid_ext, rlat_grid_ext = np.meshgrid(rlon_grid_ext, rlat_grid_ext)

# Coordinate transformation
coord = geo_crs.transform_points(rot_pole_crs, rlon_grid_ext, rlat_grid_ext)
lon, lat = coord[:, :, 0], coord[:, :, 1]

# Find required MERIT tiles
tiles_merit = np.zeros((6, 12), dtype=bool)
for i in range(lon.shape[0]):
    for j in range(lon.shape[1]):
        ind_lon = int((lon[i, j] + 180.0) // 30)
        ind_lat = int((lat[i, j] + 90.0) // 30)
        tiles_merit[ind_lat, ind_lon] = True
if out_rect_dom:
    ind = np.where(tiles_merit.sum(axis=0) > 0)[0]
    mask = (tiles_merit.sum(axis=1) > 0)
    for i in ind:
        tiles_merit[:, i] = mask
    if lon_all:
        if np.any(np.all(tiles_merit[:, [0, 11]], axis=1)):
            tiles_merit = np.repeat(mask[:, np.newaxis], 12, axis=1)
print("Number of required tiles: " + str(tiles_merit.sum()) + "/"
      + str(tiles_merit.size))

# Overview plot
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=geo_crs)
poly = plt.Polygon(list(zip(poly_rlon, poly_rlat)), facecolor="blue",
                   edgecolor="none", transform=rot_pole_crs, alpha=0.5)
ax.add_patch(poly)
mask = np.ones(rlon_grid_ext.shape, dtype=bool)
ind = num_gc_ext + 2
mask[ind:-ind, ind:-ind] = False
plt.scatter(rlon_grid_ext[mask], rlat_grid_ext[mask], transform=rot_pole_crs)
for i in range(6):
    for j in range(12):
        if tiles_merit[i, j]:
            xy = [j * 30.0 - 180.0, i * 30.0 - 90.0]
            ax.add_patch(mpatches.Rectangle(xy=xy, width=30.0, height=30.0,
                                            facecolor="red", alpha=0.3,
                                            transform=ccrs.PlateCarree()))
plt.axis([-180.0, 180.0, -90.0, 90.0])
ax.coastlines()
ax.set_aspect("auto")
gl = ax.gridlines(draw_labels=True, linestyle="-", lw=1.0, color="black")
gl.xlocator = mticker.FixedLocator(range(-180, 210, 30))
gl.ylocator = mticker.FixedLocator(range(-90, 120, 30))
gl.top_labels = False
gl.right_labels = False
plt.title("Required MERIT tiles", fontsize=12, fontweight="bold", y=1.01)

###############################################################################
# Print required tile information for EXTPAR
###############################################################################


# Function to output coordinate with naming according to hemisphere
def hemi_name(coord, axis):
    hs_names = {"lon": ["W", "E", "E"],
                "lat": ["S", "N", "N"]}
    dig_num = {"lon": 3, "lat": 2}
    coord = hs_names[axis][np.sign(coord) + 1] \
        + str(abs(coord)).zfill(dig_num[axis])
    return coord


# Get coordinate part of MERIT tile names
tile_nam_coord = []
for i in range(5, -1, -1):
    for j in range(12):
        if tiles_merit[i, j]:
            name_p = hemi_name(i * 30 - 90 + 30, "lat") + "-" \
                + hemi_name(i * 30 - 90, "lat") + "_" \
                + hemi_name(j * 30 - 180, "lon") + "-" \
                + hemi_name(j * 30 - 180 + 30, "lon")
            tile_nam_coord.append(name_p)

# Output MERIT tile names
for i in tile_nam_coord:
    print("MERIT_" + i + ".nc")

# -----------------------------------------------------------------------------
# Print required lines for EXTPAR run script
# -----------------------------------------------------------------------------

# MERIT digital elevation model
print("-" * 79)
for i in tile_nam_coord:
    print("raw_data_merit_" + i.replace("-", "_") + "='MERIT_" + i + ".nc'")
print("-" * 79)

# Names of generated SGSL files
for i in tile_nam_coord:
    print("raw_data_sgsl_" + i.replace("-", "_") + "='S_ORO_" + i + ".nc'")
print("-" * 79)

# &orography_raw_data
print(" ntiles_column = " + str(tiles_merit.sum(axis=1).max()) + ",")
print(" ntiles_row = " + str(tiles_merit.sum(axis=0).max()) + ",")
print(" topo_files = \\")
for i in tile_nam_coord:
    print("'${raw_data_merit_" + i.replace("-", "_") + "}'")
print("-" * 79)

# &sgsl_io_extpar
for i in tile_nam_coord:
    print("'${raw_data_sgsl_" + i.replace("-", "_") + "}'")
print("-" * 79)
