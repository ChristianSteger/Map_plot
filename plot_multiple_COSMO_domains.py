# Description: Plot multiple COSMO domains with high-resolution background
#              data from Natural Earth (https://www.naturalearthdata.com)
#
# Author: Christian R. Steger, October 2023

# Load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from tqdm import tqdm
import requests
import zipfile
from shapely.geometry import MultiPolygon
from shapely.ops import transform
import pyinterp
from PIL import Image
import PIL
from utilities.plot import polygon2patch
from utilities.grid import polygon_rectangular
from pyproj import CRS, Transformer

mpl.style.use("classic")

# Paths to folders
path_data = "/Users/csteger/Dropbox/IAC/Temp/Map_plot_data/"  # working dir.
path_plot = "/Users/csteger/Desktop/"

###############################################################################
# Download high-resolution background data from Natural Earth
###############################################################################

# Select background file
# file_url = "https://www.naturalearthdata.com/http//www.naturalearthdata" \
#            + ".com/download/10m/raster/HYP_LR_SR_W_DR.zip"
file_url = "https://www.naturalearthdata.com/http//www.naturalearthdata" \
           + ".com/download/10m/raster/HYP_LR_SR_OB_DR.zip"
# further available data can be found here:
# https://www.naturalearthdata.com/downloads/10m-raster-data

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

# Path to TIFF with background
file_bg = path_data + file_name + "/" + file_name + ".tif"

###############################################################################
# Load and check Natural Earth background data
###############################################################################

# Load image and create geographic coordinates
PIL.Image.MAX_IMAGE_PIXELS = 233280000
image = np.flipud(plt.imread(file_bg))  # (8100, 16200, 3)
# (M, N, 3): an image with RGB values (0-1 float or 0-255 int)
# longitude: -180.0 -> +180.0 degree, latitude: -90.0 -> +90.0 degree
extent = (-180.0, 180.0, -90.0, 90.0)
dlon_h = (extent[1] - extent[0]) / (float(image.shape[1]) * 2.0)
lon = np.linspace(extent[0] + dlon_h, extent[1] - dlon_h, image.shape[1])
dlat_h = (extent[3] - extent[2]) / (float(image.shape[0]) * 2.0)
lat = np.linspace(extent[2] + dlat_h, extent[3] - dlat_h, image.shape[0])
crs_image = ccrs.PlateCarree()

# # Test plot in geographic coordinate system
# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes()
# plt.imshow(np.flipud(image), extent=extent)
# ax.set_aspect("auto")
# xt = plt.xticks(range(-180, 210, 30))
# yt = plt.yticks(range(-90, 120, 30))

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
             "polgam": 0.0,
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
             "polgam": 0.0,
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
             "polgam": 0.0,
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
             "polgam": 0.0,
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
             "polgam": 0.0,
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
             "polgam": 0.0,
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
             "polgam": 0.0,
             "dlon": 0.04,
             "dlat": 0.04,
             "ie_tot": 650,
             "je_tot": 650}
    }
    # -------------------------------------------------------------------------
}

###############################################################################
# Plot
###############################################################################

# Interpolation axis
x_axis = pyinterp.Axis(lon, is_circle=True)
y_axis = pyinterp.Axis(lat)

# Intermediate plot with domains (-> compute domain settings)
fig = plt.figure(figsize=(18.0, 5.0))  # width, height
gs = gridspec.GridSpec(1, 3, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.025,
                       width_ratios=[1.0, 1.35, 1.4375])
for ind_i, i in enumerate(list(domains.keys())):

    # Compute 'average coordinate origin' for domain
    cen_lon, cen_lat = [], []
    for j in list(domains[i].keys()):
        crs_rot = ccrs.RotatedPole(pole_latitude=domains[i][j]["pollat"],
                                   pole_longitude=domains[i][j]["pollon"],
                                   central_rotated_longitude
                                   =domains[i][j]["polgam"])
        x, y = ccrs.PlateCarree().transform_point(0.0, 0.0, crs_rot)
        cen_lon.append(x)
        cen_lat.append(y)
    cen_lon, cen_lat = np.mean(cen_lon), np.mean(cen_lat)

    # Define map projection
    crs_laea = ccrs.LambertAzimuthalEqualArea(
        central_longitude=cen_lon, central_latitude=cen_lat)
    crs_proj_laea = CRS.from_user_input(crs_laea)

    # Compute bounding box in map projection
    domains_map_proj = []
    for j in list(domains[i].keys()):

        # Rectangular domain in rotated coordinates
        crs_rot = ccrs.RotatedPole(pole_latitude=domains[i][j]["pollat"],
                                   pole_longitude=domains[i][j]["pollon"],
                                   central_rotated_longitude
                                   =domains[i][j]["polgam"])
        rlon_llc = domains[i][j]["startlon_tot"] \
            - (domains[i][j]["dlon"] / 2.0)
        rlat_llc = domains[i][j]["startlat_tot"] \
            - (domains[i][j]["dlat"] / 2.0)
        box = (rlon_llc,
               rlat_llc,
               rlon_llc + domains[i][j]["ie_tot"] * domains[i][j]["dlon"],
               rlat_llc + domains[i][j]["je_tot"] * domains[i][j]["dlat"])
        poly = polygon_rectangular(box, spacing=0.01)

        # Rectangular domain in map projection
        crs_proj_rot = CRS.from_user_input(crs_rot)
        project = Transformer.from_crs(crs_proj_rot, crs_proj_laea,
                                       always_xy=True).transform
        domains_map_proj.append(transform(project, poly))
    bounds = MultiPolygon(domains_map_proj).bounds
    # x_min, y_min, x_max, y_max
    bounds = (bounds[0] - 250_000.0,
              bounds[1] - 250_000.0,
              bounds[2] + 250_000.0,
              bounds[3] + 250_000.0)

    # Interpolate background image
    x_ip = np.linspace(bounds[0], bounds[2], 3000)
    y_ip = np.linspace(bounds[1], bounds[3], 3000)
    crs_proj_image = CRS.from_user_input(crs_image)
    transformer = Transformer.from_crs(crs_proj_laea, crs_proj_image,
                                       always_xy=True)
    lon_ip, lat_ip = transformer.transform(*np.meshgrid(x_ip, y_ip))

    temp = []
    for j in range(3):
        grid = pyinterp.Grid2D(x_axis, y_axis, image[:, :, j].transpose())
        data_ip = pyinterp.bivariate(
            grid, lon_ip.ravel(), lat_ip.ravel(),
            interpolator="bilinear", bounds_error=True, num_threads=0)
        temp.append(data_ip.reshape(lon_ip.shape)[:, :, np.newaxis])
    image_ip = np.concatenate(temp, axis=2).astype(np.uint8)

    # Domain labels (-> have to be set manually...)
    label_pos = {
        # ---------------------------------------------------------------------
        "Europe": {
            "EURO":        (-2_000_000, 1_400_000),
            "EURO-CORDEX": (-2_800_000, 2_000_000),
            "ALP":         (-800_000, 180_000)},
        # ---------------------------------------------------------------------
        "Atlantic": {
            "T_ATL":       (-2_900_000, 2_200_000),
            "L_ATL":       (+2_000_000, 2_300_000)},
        # ---------------------------------------------------------------------
        "East_Asia": {
            "EAS-CORDEX":  (-4_100_000, 3_500_000),
            "BECCY":       (-2_900_000, 1_100_000)
                      }
        # ---------------------------------------------------------------------
    }

    # Plot
    bounds_ro = (bounds[0], bounds[2], bounds[1], bounds[3])
    ax = plt.subplot(gs[ind_i], projection=crs_laea)
    ax.imshow(np.flipud(image_ip), extent=bounds_ro, transform=crs_laea)
    ax.set_aspect("auto")
    ax.coastlines("50m", linewidth=0.5)
    for ind_j, j in enumerate(domains_map_proj):
        poly = polygon2patch(j, facecolor="none", edgecolor="black",
                             alpha=1.0, linewidth=2.0, transform=crs_laea)
        ax.add_collection(poly)
        # ---------------------------------------------------------------------
        dom_name = list(domains[i].keys())[ind_j]
        plt.text(*label_pos[i][dom_name], domains[i][dom_name]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_laea)
        # ---------------------------------------------------------------------
    ax.set_extent(bounds_ro, crs=crs_laea)
    print("Region " + i + " plotted")

fig.savefig(path_plot + "COSMO_domains.png", dpi=300, bbox_inches="tight")
plt.close(fig)
