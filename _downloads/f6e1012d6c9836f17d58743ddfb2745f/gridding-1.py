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

grid_limits = ((0., 15000.), (-50000., 50000.), (-50000., 50000.))
grid_shape = (31, 201, 201)

grid_sw = pyart.map.grid_from_radars([radar_sw], grid_limits=grid_limits,
                                 grid_shape=grid_shape, gatefilter=gatefilter_sw)
grid_se = pyart.map.grid_from_radars([radar_se], grid_limits=grid_limits,
                                 grid_shape=grid_shape, gatefilter=gatefilter_se)

fig = plt.figure(figsize=(8, 12))
ax1 = plt.subplot(211)
display1 = pyart.graph.GridMapDisplay(grid_sw)
display1.plot_latitude_slice('corrected_velocity', lat=36.5, ax=ax1, fig=fig, vmin=-30, vmax=30)
ax2 = plt.subplot(212)
display2 = pyart.graph.GridMapDisplay(grid_se)
display2.plot_latitude_slice('corrected_velocity', lat=36.5, ax=ax2, fig=fig, vmin=-30, vmax=30)