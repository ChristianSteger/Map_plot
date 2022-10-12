# Description: Plot multiple COSMO domains with high-resolution background
#              data from Natural Earth (https://www.naturalearthdata.com)
#
# Author: Christian R. Steger, October 2022

# Load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import time
from tqdm import tqdm
import requests
import zipfile
import rasterio

mpl.style.use("classic")

# Paths to folders
path_data = "/Users/csteger/Desktop/Data/"  # working directory
path_plot = "/Users/csteger/Desktop/"       # path for plot

###############################################################################
# Download high-resolution background data from Natural Earth
###############################################################################

# Select background file
# file_url = "https://www.naturalearthdata.com/http//www.naturalearthdata" \
#            + ".com/download/10m/raster/HYP_LR_SR_W_DR.zip"
file_url = "https://www.naturalearthdata.com/http//www.naturalearthdata" \
           + ".com/download/10m/raster/HYP_LR_SR_OB_DR.zip"
# further available data can be found here:
# https://www.naturalearthdata.com/features/ -> Raster Data Themes

# Download
if not os.path.isdir(path_data):
    raise ValueError("Path for data does not exist")
response = requests.get(file_url, stream=True, headers={"User-Agent": "XY"})
if response.ok:
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 * 10
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB",
                        unit_scale=True)
    with open(path_data + os.path.split(file_url)[-1], "wb") as infile:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            infile.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise ValueError("Inconsistency in file size")
else:
    raise ValueError("URL does not exist")

# Unzip
file_name = file_url.split("/")[-1].split(".")[0]
with zipfile.ZipFile(path_data + file_url.split("/")[-1], "r") as zip_ref:
    zip_ref.extractall(path_data + file_name)
os.remove(path_data + file_url.split("/")[-1])

# Path to GeoTIFF with background
file_bg = path_data + file_name + "/" + file_name + ".tif"

###############################################################################
# Load and check Natural Earth background data
###############################################################################

# Load GeoTIFF
src = rasterio.open(file_bg)
image = np.empty(src.read(1).shape + (3,), dtype=src.read(1).dtype)
for i in range(0, 3):
    image[:, :, i] = src.read(i + 1)
extent = [src.bounds.left, src.bounds.right,
          src.bounds.bottom, src.bounds.top]
src.close()

# Test plot (equidistant cylindrical projection)
plt.figure(figsize=(12, 6))
ax = plt.axes()
plt.imshow(image, extent=extent)
ax.set_aspect("auto")
xt = plt.xticks(range(-180, 210, 30))
yt = plt.yticks(range(-90, 120, 30))

# Coordinates for pixel edges
lon = np.linspace(extent[0], extent[1], image.shape[1] + 1, dtype=np.float64)
lat = np.linspace(extent[3], extent[2], image.shape[0] + 1, dtype=np.float64)

###############################################################################
# Domain specifications and plot settings
###############################################################################

# COSMO domains
domains = {
    # -------------------------------------------------------------------------
    "Europe": {
        "EURO":
            {"name_plot": "EURO",
             "startlat_tot": -14.86,
             "startlon_tot": -18.86,
             "pollat": 43.0,
             "pollon": -170.0,
             "dlon": 0.02,
             "dlat": 0.02,
             "ie_tot": 1542,
             "je_tot": 1542},
        "EURO-CORDEX":
            {"name_plot": "EURO-\nCORDEX",
             "startlat_tot": -24.805,
             "startlon_tot": -29.805,
             "pollat": 39.25,
             "pollon": -162.00,
             "dlon": 0.11,
             "dlat": 0.11,
             "ie_tot": 450,
             "je_tot": 438},
        "ALP":
            {"name_plot": "ALP",
             "startlat_tot": -6.6,
             "startlon_tot": -6.2,
             "pollat": 43.0,
             "pollon": -170.0,
             "dlon": 0.02,
             "dlat": 0.02,
             "ie_tot": 800,
             "je_tot": 600},
    },
    # -------------------------------------------------------------------------
    "Atlantic": {
        "T_ATL":
            {"name_plot": "T-ATL",
             "startlat_tot": -7.77,
             "startlon_tot": -65.625,
             "pollat": 90.0,
             "pollon": -180.0,
             "dlon": 0.02,
             "dlat": 0.02,
             "ie_tot": 2310,
             "je_tot": 1542},
        "L_ATL":
            {"name_plot": "L-ATL",
             "startlat_tot": -37.45,
             "startlon_tot": -54.54,
             "pollat": 90.0,
             "pollon": -180.0,
             "dlon": 0.03,
             "dlat": 0.03,
             "ie_tot": 2750,
             "je_tot": 2065},
    },
    # -------------------------------------------------------------------------
    "East_Asia": {
        "EAS-CORDEX":
            {"name_plot": "EAS-CORDEX",
             "startlat_tot": -23.53,
             "startlon_tot": -44.66,
             "pollat": 61.0,
             "pollon": -63.7,
             "dlon": 0.11,
             "dlat": 0.11,
             "ie_tot": 818,
             "je_tot": 530},
        "BECCY":
            {"name_plot": "BECCY",
             "startlat_tot": -12.67,
             "startlon_tot": -28.24,
             "pollat": 61.00,
             "pollon": -63.70,
             "dlon": 0.04,
             "dlat": 0.04,
             "ie_tot": 650,
             "je_tot": 650}
    }}

# Spatial extent of regions (lon_min, lon_max, lat_min, lat_max)
reg_spat_ext = {"Europe":    [-20.0, 39.0, 21.0, 77.0],
                "Atlantic":  [-69.0, 37.0, -39.0, 25.0],
                "East_Asia": [70.0, 163.0, 1.0, 63.0]}
# -> define according to intermediate domain plots generated below

# Domain label offset (x, y)
label_offset = {
    "Europe": {
        "EURO":        (1.5, -3.2),
        "EURO-CORDEX": (1.8, -5.8),
        "ALP":         (1.5, -3.2),
    },
    "Atlantic": {
        "T_ATL":   (36.5, -3.5),
        "L_ATL":   (71.5, -3.4),
    },
    "East_Asia": {
        "EAS-CORDEX":  (2.4, -5.4),
        "BECCY":       (1.6, -4.0),
    }}
# -> define according to intermediate domain plots generated below

# Spatial extent of Natural Earth data (lon_min, lon_max, lat_min, lat_max)
ne_spat_ext = {"Europe":    [-65.0, 85.0, 5.0, 80.0],
               "Atlantic":  [-100.0, 60.0, -50.0, 30.0],
               "East_Asia": [-180.0, 180.0, -5.0, 70.0]}
# -> define according to final domain plot generated below

###############################################################################
# Plot
###############################################################################


# Function to generate domain polygon
def domain_poly(domain, poly_res=0.01):

    crs_rot = ccrs.RotatedPole(pole_latitude=domain["pollat"],
                               pole_longitude=domain["pollon"])
    rlon = np.linspace(domain["startlon_tot"] - domain["dlon"] / 2.0,
                       domain["startlon_tot"] - domain["dlon"] / 2.0
                       + domain["ie_tot"] * domain["dlon"],
                       int(domain["ie_tot"] * domain["dlon"] / poly_res) + 1)
    rlat = np.linspace(domain["startlat_tot"] - domain["dlat"] / 2.0,
                       domain["startlat_tot"] - domain["dlat"] / 2.0
                       + domain["je_tot"] * domain["dlat"],
                       int(domain["je_tot"] * domain["dlat"] / poly_res) + 1)
    poly_rlon = np.hstack((rlon,
                           np.repeat(rlon[-1], len(rlat))[1:],
                           rlon[::-1][1:],
                           np.repeat(rlon[0], len(rlat))[1:]))
    poly_rlat = np.hstack((np.repeat(rlat[0], len(rlon)),
                           rlat[1:],
                           np.repeat(rlat[-1], len(rlon))[1:],
                           rlat[::-1][1:]))

    return crs_rot, rlon, rlat, poly_rlon, poly_rlat


# -----------------------------------------------------------------------------

# Intermediate domain plots (-> determine width ratios of figure panels and
# check that map extent is appropriately chosen)
map_proj, width_ratios = {}, {}
for i in list(domains.keys()):
    lon_cen = (reg_spat_ext[i][0] + reg_spat_ext[i][1]) / 2.0
    lat_cen = (reg_spat_ext[i][2] + reg_spat_ext[i][3]) / 2.0
    if i == "East_Asia":
        lat_cen -= 5.0
    if i == "Atlantic":
        lat_cen += 10.0
    laea_crs = ccrs.LambertAzimuthalEqualArea(
        central_longitude=lon_cen, central_latitude=lat_cen)
    fig = plt.figure()
    ax = plt.axes(projection=laea_crs)
    ax.coastlines("110m")

    # Plot COSMO domains
    for j in list(domains[i].keys()):
        crs_rot, rlon, rlat, poly_rlon, poly_rlat = domain_poly(domains[i][j])
        poly = plt.Polygon(list(zip(poly_rlon, poly_rlat)), facecolor="none",
                           edgecolor="black", alpha=1.0,
                           linewidth=1.5, zorder=4, transform=crs_rot)
        ax.add_patch(poly)
        x_txt = rlon[0] + label_offset[i][j][0]
        y_txt = rlat[-1] + label_offset[i][j][1]
        plt.text(x_txt, y_txt, domains[i][j]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_rot)

    ax.set_extent(reg_spat_ext[i], crs=ccrs.PlateCarree())
    map_proj[i] = laea_crs
    width_ratios[i] = 1.0 / ax.get_data_ratio()
    # plt.close(fig)

# Final domain plot
fig = plt.figure(figsize=(16.0, 4.23))
gs = gridspec.GridSpec(1, 3, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.025,
                       width_ratios=[width_ratios[i] for i in domains.keys()])
n = 0
for i in list(domains.keys()):

    ax = plt.subplot(gs[n], projection=map_proj[i])

    # Plot raster background
    sd = (slice(np.argmin(np.abs(ne_spat_ext[i][3] - lat)),
                np.argmin(np.abs(ne_spat_ext[i][2] - lat))),
          slice(np.argmin(np.abs(ne_spat_ext[i][0] - lon)),
                np.argmin(np.abs(ne_spat_ext[i][1] - lon))))
    extent_sd = [lon[sd[1].start], lon[sd[1].stop],
                 lat[sd[0].stop], lat[sd[0].start]]
    regrid_shape = 5000  # increase this value if background is too pixelated
    t_beg = time.time()
    plt.imshow(image[sd], extent=extent_sd, transform=ccrs.PlateCarree(),
               interpolation="spline36", origin="upper",
               rasterized=True, regrid_shape=regrid_shape)
    print("Time for plotting raster: %.2f" % (time.time() - t_beg) + " s")
    ax.coastlines("50m", linewidth=0.5)
    ax.set_extent(reg_spat_ext[i], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")

    # Plot COSMO domains
    for j in list(domains[i].keys()):
        crs_rot, rlon, rlat, poly_rlon, poly_rlat = domain_poly(domains[i][j])
        poly = plt.Polygon(list(zip(poly_rlon, poly_rlat)), facecolor="none",
                           edgecolor="black", alpha=1.0,
                           linewidth=1.5, zorder=4, transform=crs_rot)
        ax.add_patch(poly)
        x_txt = rlon[0] + label_offset[i][j][0]
        y_txt = rlat[-1] + label_offset[i][j][1]
        plt.text(x_txt, y_txt, domains[i][j]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_rot)

    print("Region " + i + " plotted")

    n += 1
fig.savefig(path_plot + "COSMO_domains.png", dpi=300, bbox_inches="tight")
plt.close(fig)
