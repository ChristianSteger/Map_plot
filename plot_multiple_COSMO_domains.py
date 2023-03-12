# Description: Plot multiple COSMO domains with high-resolution background
#              data from Natural Earth (https://www.naturalearthdata.com)
#
# Required conda environment:
# conda create -n plot_env numpy matplotlib cartopy tqdm requests rasterio
# shapely descartes pykdtree -c conda-forge
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
from shapely.geometry import Polygon
from descartes import PolygonPatch

mpl.style.use("classic")

# Paths to folders
path_data = os.getenv("HOME") + "/Desktop/Data/"  # working directory
path_plot = os.getenv("HOME") + "/Desktop/"       # path for plot

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
fig = plt.figure(figsize=(12, 6))
ax = plt.axes()
plt.imshow(image, extent=extent)
ax.set_aspect("auto")
xt = plt.xticks(range(-180, 210, 30))
yt = plt.yticks(range(-90, 120, 30))
# plt.close(fig)

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

# -----------------------------------------------------------------------------
# Optional settings (-> enhance plot layout)
# -----------------------------------------------------------------------------

# Domain label offsets (in rotated coordinates [rlon, rlat])
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
# -> define according to intermediate plot generated below

# Miscellaneous settings
dist_edge = {"Europe": 200000.0, "Atlantic": 300000.0, "East_Asia": 300000.0}
dist_edge_default = 300000.0
# minimal distance from domains to plot axis in map projection units
regrid_shape = 5000
# remapping resolution of plt.imshow(). Increase this value if background is
# too pixelated. Plotting is substantially faster with smaller value.

###############################################################################
# Functions
###############################################################################


# Compute 1-dimensional coordinates for domain
def rlon_rlat_1d(domain, poly_res=0.01):

    rlon = np.linspace(domain["startlon_tot"] - domain["dlon"] / 2.0,
                       domain["startlon_tot"] - domain["dlon"] / 2.0
                       + domain["ie_tot"] * domain["dlon"],
                       int(domain["ie_tot"] * domain["dlon"] / poly_res) + 1)
    rlat = np.linspace(domain["startlat_tot"] - domain["dlat"] / 2.0,
                       domain["startlat_tot"] - domain["dlat"] / 2.0
                       + domain["je_tot"] * domain["dlat"],
                       int(domain["je_tot"] * domain["dlat"] / poly_res) + 1)

    return rlon, rlat


# Compute polygon from 1-dimensional coordinates
def coord2poly(x, y):

    poly_x = np.hstack((x,
                        np.repeat(x[-1], len(y))[1:],
                        x[::-1][1:],
                        np.repeat(x[0], len(y))[1:]))
    poly_y = np.hstack((np.repeat(y[0], len(x)),
                        y[1:],
                        np.repeat(y[-1], len(x))[1:],
                        y[::-1][1:]))

    return poly_x, poly_y


###############################################################################
# Plot
###############################################################################

# Intermediate plot with domains (-> compute domain settings)
domain_set = {}
for i in list(domains.keys()):

    # Define map projection
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
    crs_laea = ccrs.LambertAzimuthalEqualArea(
        central_longitude=cen_lon, central_latitude=cen_lat)

    fig = plt.figure()
    ax = plt.axes(projection=crs_laea)
    plt.scatter(cen_lon, cen_lat, s=80, marker="*", color="black")
    ax.coastlines("110m")

    # Plot COSMO domains
    domain_ext = [0.0, 0.0, 0.0, 0.0]
    for j in list(domains[i].keys()):
        crs_rot = ccrs.RotatedPole(pole_latitude=domains[i][j]["pollat"],
                                   pole_longitude=domains[i][j]["pollon"],
                                   central_rotated_longitude
                                   =domains[i][j]["polgam"])
        rlon, rlat = rlon_rlat_1d(domains[i][j], poly_res=0.02)
        poly_rlon, poly_rlat = coord2poly(rlon, rlat)
        coords = crs_laea.transform_points(crs_rot, poly_rlon, poly_rlat)
        poly = plt.Polygon(list(zip(coords[:, 0], coords[:, 1])),
                           facecolor="none", edgecolor="black", alpha=1.0,
                           linewidth=1.5, zorder=4, transform=crs_laea)
        ax.add_patch(poly)
        poly_shp = Polygon(zip(coords[:, 0], coords[:, 1])) \
            .buffer(dist_edge.get(i, dist_edge_default))
        poly = PolygonPatch(poly_shp, facecolor="none", edgecolor="gray",
                            alpha=1.0, linewidth=1.5, zorder=4,
                            transform=crs_laea)
        ax.add_patch(poly)
        x, y = poly_shp.exterior.coords.xy
        domain_ext = [np.minimum(domain_ext[0], np.min(x)),
                      np.maximum(domain_ext[1], np.max(x)),
                      np.minimum(domain_ext[2], np.min(y)),
                      np.maximum(domain_ext[3], np.max(y))]
        x_txt, y_txt = rlon[0], rlat[-1]
        if (i in label_offset.keys()) and (j in label_offset[i].keys()):
            x_txt += label_offset[i][j][0]
            y_txt += label_offset[i][j][1]
        plt.text(x_txt, y_txt, domains[i][j]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_rot)

    ax.set_extent(domain_ext, crs=crs_laea)

    # Define boundaries for Natural Earth data that is considered
    dx = dy = 2000.0  # resolution of polygon
    x = np.linspace(domain_ext[0], domain_ext[1],
                    int((domain_ext[1] - domain_ext[0]) / dx))
    y = np.linspace(domain_ext[2], domain_ext[3],
                    int((domain_ext[3] - domain_ext[2]) / dy))
    poly_x, poly_y = coord2poly(x, y)
    coords = ccrs.PlateCarree().transform_points(crs_laea, poly_x, poly_y)
    add_ext_na = 2.0
    extent_na = [np.maximum(coords[:, 0].min() - add_ext_na, -180.0),
                 np.minimum(coords[:, 0].max() + add_ext_na, 180.0),
                 np.maximum(coords[:, 1].min() - add_ext_na, -90.0),
                 np.minimum(coords[:, 1].max() + add_ext_na, 90.0)]
    domain_set[i] = {}
    domain_set[i]["map_projection"] = crs_laea
    domain_set[i]["domain_extent"] = domain_ext
    domain_set[i]["width"] = 1.0 / ax.get_data_ratio()
    domain_set[i]["extent_natural_earth"] = extent_na
    # plt.close(fig)

# Final plot with domains
width_ratios = [domain_set[i]["width"] for i in domains.keys()]
fig_width = float(len(domains) * (16 / 3))
fig = plt.figure(figsize=(fig_width, fig_width / sum(width_ratios)))
gs = gridspec.GridSpec(1, len(domains), left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.0, wspace=0.025,
                       width_ratios=width_ratios)
n = 0
for i in list(domains.keys()):

    ax = plt.subplot(gs[n], projection=domain_set[i]["map_projection"])

    # Plot raster background
    extent_na = domain_set[i]["extent_natural_earth"]
    sd = (slice(np.argmin(np.abs(extent_na[3] - lat)),
                np.argmin(np.abs(extent_na[2] - lat))),
          slice(np.argmin(np.abs(extent_na[0] - lon)),
                np.argmin(np.abs(extent_na[1] - lon))))
    extent_sd = [lon[sd[1].start], lon[sd[1].stop],
                 lat[sd[0].stop], lat[sd[0].start]]
    t_beg = time.time()
    plt.imshow(image[sd], extent=extent_sd, transform=ccrs.PlateCarree(),
               interpolation="spline36", origin="upper",
               rasterized=True, regrid_shape=regrid_shape)
    print("Time for plotting raster: %.2f" % (time.time() - t_beg) + " s")
    ax.coastlines("50m", linewidth=0.5)
    ax.set_extent(domain_set[i]["domain_extent"],
                  crs=domain_set[i]["map_projection"])
    ax.set_aspect("auto")

    # Plot COSMO domains
    for j in list(domains[i].keys()):
        crs_rot = ccrs.RotatedPole(pole_latitude=domains[i][j]["pollat"],
                                   pole_longitude=domains[i][j]["pollon"],
                                   central_rotated_longitude
                                   =domains[i][j]["polgam"])
        rlon, rlat = rlon_rlat_1d(domains[i][j])
        poly_rlon, poly_rlat = coord2poly(rlon, rlat)
        poly = plt.Polygon(list(zip(poly_rlon, poly_rlat)), facecolor="none",
                           edgecolor="black", alpha=1.0,
                           linewidth=1.5, zorder=4, transform=crs_rot)
        ax.add_patch(poly)
        x_txt, y_txt = rlon[0], rlat[-1]
        if (i in label_offset.keys()) and (j in label_offset[i].keys()):
            x_txt += label_offset[i][j][0]
            y_txt += label_offset[i][j][1]
        plt.text(x_txt, y_txt, domains[i][j]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_rot)

    print("Region " + i + " plotted")

    n += 1
fig.savefig(path_plot + "COSMO_domains.png", dpi=300, bbox_inches="tight")
plt.close(fig)
