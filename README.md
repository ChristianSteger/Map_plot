# Map_plot
Python scripts for producing map plots related to climate modelling.

# Installation

First, create a Conda environment with all the required Python packages:

```bash
conda create -n map_plot -c conda-forge numpy tqdm requests xarray matplotlib cartopy cmcrameri shapely pyinterp
```

Subsequently, install the Python packages [Utilities](https://github.com/ChristianSteger/Utilities).

# Visualisation scripts

- **plot_nested_COSMO_domains.py**: Plot nested COSMO domains.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/COSMO_nested_domains.png?raw=true "Output example")

- **plot_multiple_COSMO_domains.py**: Plot multiple COSMO domains with high-resolution background data from [Natural Earth](https://www.naturalearthdata.com).
![Alt text](https://github.com/ChristianSteger/Media/blob/master/COSMO_domains.png?raw=true "Output example")

- **plot_DEM.py**: Plot Digital Elevation Model (DEM) data.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/DEM_map_plot.png?raw=true "Output example")

- **plot_PRUDENCE_regions.py**: Plot PRUDENCE regions on a map with terrain from COSMO simulation.
![Alt text](https://github.com/ChristianSteger/Media/blob/master/PRUDENCE_regions_map.png?raw=true "Output example")