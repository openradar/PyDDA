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

fig = plt.figure(figsize=(8, 6))
display = pyart.graph.RadarDisplay(radar_sw)
display.plot_ppi('velocity_texture',
                     sweep=0,
                     vmin=0,
                     vmax=10,
                     cmap=plt.get_cmap('twilight_shifted')
                     )

# Plot a histogram of the velocity textures
fig = plt.figure(figsize=[8, 8])
hist, bins = np.histogram(radar_sw.fields['velocity_texture']['data'],
                          bins=150)
bins = (bins[1:]+bins[:-1])/2.0
plt.plot(bins,
         hist,
         label='Velocity Texture Frequency'
         )
plt.axvline(3,
            color='r',
            label='Proposed Velocity Texture Threshold'
            )
plt.xlabel('Velocity texture')
plt.ylabel('Count')
plt.legend()