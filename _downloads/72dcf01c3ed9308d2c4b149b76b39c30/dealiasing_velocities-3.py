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

# Calculate the Velocity Texture and apply the PyART GateFilter Utilityx
vel_tex_sw = pyart.retrieve.calculate_velocity_texture(radar_sw,
                                                       vel_field='mean_doppler_velocity',
                                                       nyq=19
                                                       )
vel_tex_se = pyart.retrieve.calculate_velocity_texture(radar_se,
                                                       vel_field='mean_doppler_velocity',
                                                       nyq=19
                                                       )

## Add velocity texture to the radar objects
radar_sw.add_field('velocity_texture', vel_tex_sw, replace_existing=True)
radar_se.add_field('velocity_texture', vel_tex_se, replace_existing=True)

# Apply a GateFilter
gatefilter_sw = pyart.filters.GateFilter(radar_sw)
gatefilter_sw.exclude_above('velocity_texture', 3)
gatefilter_se = pyart.filters.GateFilter(radar_se)
gatefilter_se.exclude_above('velocity_texture', 3)

# Apply Region Based DeAlising Utiltiy
vel_dealias_sw = pyart.correct.dealias_region_based(radar_sw,
                                                    vel_field='mean_doppler_velocity',
                                                    nyquist_vel=19,
                                                    centered=True,
                                                    gatefilter=gatefilter_sw
                                                    )

# Apply Region Based DeAlising Utiltiy
vel_dealias_se = pyart.correct.dealias_region_based(radar_se,
                                                    vel_field='mean_doppler_velocity',
                                                    nyquist_vel=19,
                                                    centered=True,
                                                    gatefilter=gatefilter_se
                                                    )

# Add our data dictionary to the radar object
radar_se.add_field('corrected_velocity', vel_dealias_se, replace_existing=True)
radar_sw.add_field('corrected_velocity', vel_dealias_sw, replace_existing=True)

fig = plt.figure(figsize=(16, 6))

# Plot the southwestern radar
ax = plt.subplot(121, projection=ccrs.PlateCarree())
disp1 = pyart.graph.RadarMapDisplay(radar_sw)
disp1.plot_ppi_map("corrected_velocity",
                   sweep=1,
                   ax=ax,
                   vmin=-35,
                   vmax=35,
                   min_lat=36,
                   max_lat=37,
                   min_lon=-98,
                   max_lon=-97,
                   lat_lines=np.arange(36, 37.25, 0.25),
                   lon_lines=np.arange(-98, -96.75, 0.25),
                   cmap=plt.get_cmap('twilight_shifted')
)

# Plot the southeastern radar
ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
disp2 = pyart.graph.RadarMapDisplay(radar_se)
disp2.plot_ppi_map("corrected_velocity",
                   sweep=1,
                   ax=ax2,
                   vmin=-35,
                   vmax=35,
                   min_lat=36,
                   max_lat=37,
                   min_lon=-98,
                   max_lon=-97,
                   lat_lines=np.arange(36, 37.25, 0.25),
                   lon_lines=np.arange(-98, -96.75, 0.25),
                   cmap=plt.get_cmap('twilight_shifted')
)