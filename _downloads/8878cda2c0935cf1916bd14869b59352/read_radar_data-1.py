import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import pyart
from pyart.testing import get_test_data

warnings.filterwarnings("ignore")

# read in the data from both XSAPR radars
xsapr_sw_file = get_test_data("swx_20120520_0641.nc")
xsapr_se_file = get_test_data("sex_20120520_0641.nc")
radar_sw = pyart.io.read_cfradial(xsapr_sw_file)
radar_se = pyart.io.read_cfradial(xsapr_se_file)

fig = plt.figure(figsize=(16, 6))
ax = plt.subplot(121, projection=ccrs.PlateCarree())

# Plot the southwestern radar
disp1 = pyart.graph.RadarMapDisplay(radar_sw)
disp1.plot_ppi_map(
    "reflectivity_horizontal",
    sweep=1,
    ax=ax,
    vmin=-20,
    vmax=70,
    min_lat=36,
    max_lat=37,
    min_lon=-98,
    max_lon=-97,
    lat_lines=np.arange(36, 37.25, 0.25),
    lon_lines=np.arange(-98, -96.75, 0.25),
)
# Plot the southeastern radar
ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
disp2 = pyart.graph.RadarMapDisplay(radar_se)
disp2.plot_ppi_map(
    "reflectivity_horizontal",
    sweep=1,
    ax=ax2,
    vmin=-20,
    vmax=70,
    min_lat=36,
    max_lat=37,
    min_lon=-98,
    max_lon=-97,
    lat_lines=np.arange(36, 37.25, 0.25),
    lon_lines=np.arange(-98, -96.75, 0.25),
)