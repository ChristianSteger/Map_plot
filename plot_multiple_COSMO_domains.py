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
from shapely.ops import transform
from shapely.ops import unary_union
from pyproj import CRS, Transformer
from utilities.plot import polygon2patch
from utilities.grid import polygon_rectangular
from utilities.plot import naturalearth_background

mpl.style.use("classic")

# Paths to folders
path_plot = "/Users/csteger/Desktop/"

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

# Plot maps
fig = plt.figure(figsize=(18.0, 5.0))  # width, height
gs = gridspec.GridSpec(1, 3, left=0.02, bottom=0.02, right=0.98,
                       top=0.98, hspace=0.1, wspace=-0.2,
                       width_ratios=[1.0, 1.0, 1.0])
for ind_i, i in enumerate(domains.keys()):

    # Compute domain boundaries and map centre
    domains_rot = []
    for ind_j, j in enumerate(domains[i].keys()):
        crs_rot = ccrs.RotatedPole(
            pole_latitude=domains[i][j]["pollat"],
            pole_longitude=domains[i][j]["pollon"],
            central_rotated_longitude=domains[i][j]["polgam"])
        rlon_llc = domains[i][j]["startlon_tot"] \
            - (domains[i][j]["dlon"] / 2.0)
        rlat_llc = domains[i][j]["startlat_tot"] \
            - (domains[i][j]["dlat"] / 2.0)
        box = (rlon_llc,
               rlat_llc,
               rlon_llc + domains[i][j]["ie_tot"] * domains[i][j]["dlon"],
               rlat_llc + domains[i][j]["je_tot"] * domains[i][j]["dlat"])
        poly = polygon_rectangular(box, spacing=0.01)
        if ind_j == 0:
            domains_rot.append(poly)
            crs_rot_ref = crs_rot
        else:
            project = Transformer.from_crs(
                CRS.from_user_input(crs_rot),
                CRS.from_user_input(crs_rot_ref),
                always_xy=True).transform
            domains_rot.append(transform(project, poly))
    domains_union = unary_union(domains_rot)
    cen_lon, cen_lat = ccrs.PlateCarree().transform_point(
        *domains_union.centroid.xy, crs_rot)
    crs_map = ccrs.Orthographic(central_longitude=cen_lon,
                                central_latitude=cen_lat)

    # Domain labels (-> have to be set manually)
    label_pos = {
        # ---------------------------------------------------------------------
        "Europe": {
            "EURO":        (-1_900_000, -2_400_000),
            "EURO-CORDEX": (1_200_000, -3_400_000),
            "ALP":         (200_000, -100_000)},
        # ---------------------------------------------------------------------
        "Atlantic": {
            "T_ATL":       (-1_500_000, -750_000),
            "L_ATL":       (2_800_000, 2_550_000)},
        # ---------------------------------------------------------------------
        "East_Asia": {
            "EAS-CORDEX":  (-3_400_000, 3_400_000),
            "BECCY":       (-2_800_000, 1_100_000)
                      }
        # ---------------------------------------------------------------------
    }

    # Plot
    ax = plt.subplot(gs[ind_i], projection=crs_map)
    ax.set_global()
    image_name = "cross_blended_hypso_with_relief_water_drains_" \
                 + "and_ocean_bottom"
    naturalearth_background(ax, image_name=image_name, image_res="medium",
                            interp_res=(3000, 3000))
    ax.coastlines("50m", linewidth=0.5)
    for ind_j, j in enumerate(domains_rot):
        poly = polygon2patch(domains_rot[ind_j], facecolor="none",
                             edgecolor="black", alpha=1.0, linewidth=2.0,
                             transform=crs_rot_ref)
        ax.add_collection(poly)
        dom_name = list(domains[i].keys())[ind_j]
        plt.text(*label_pos[i][dom_name], domains[i][dom_name]["name_plot"],
                 fontsize=10, fontweight="bold", transform=crs_map)
    print("Region " + i + " plotted")

fig.savefig(path_plot + "COSMO_domains.png", dpi=300, bbox_inches="tight")
plt.close(fig)
