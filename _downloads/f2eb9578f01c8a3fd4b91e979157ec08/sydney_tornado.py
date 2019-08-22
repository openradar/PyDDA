"""

Example of a wind retrieval in a tornado over Sydney
----------------------------------------------------

This shows an example of how to retrieve winds from 4 radars over Sydney.

Grided netCDF files are downloadable from:
https://drive.google.com/drive/u/1/folders/1pcQxWRJV78xuJePTZnlXPPpMe1qut0ie

.. image:: ../../sydney_tornado.png

"""

import pyart
import pydda
import matplotlib.pyplot as plt
import numpy as np

grid1 = pyart.io.read_grid('grid1_sydney.nc')
grid2 = pyart.io.read_grid('grid2_sydney.nc')
grid3 = pyart.io.read_grid('grid3_sydney.nc')
grid4 = pyart.io.read_grid('grid4_sydney.nc')

# Set initialization and do retrieval
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(grid1, vel_field='VRADH_corr')
new_grids = pydda.retrieval.get_dd_wind_field([grid1, grid2, grid3, grid4],
                                              u_init, v_init, w_init,
                                              vel_name='VRADH_corr', refl_field='DBZH',
                                              mask_outside_opt=True)
# Make a neat plot
fig = plt.figure(figsize=(10,7))
ax = pydda.vis.plot_horiz_xsection_quiver_map(new_grids, background_field='DBZH', level=3,
                                              show_lobes=False, bg_grid_no=3, vmin=0, vmax=60,
                                              quiverkey_len=40.0,
                                              quiver_spacing_x_km=2.0, quiver_spacing_y_km=2.0,
                                              quiverkey_loc='top', colorbar_contour_flag=True,
                                              cmap='pyart_HomeyerRainbow')
ax.set_xticks(np.arange(150.5, 153, 0.1))
ax.set_yticks(np.arange(-36, -32.0, 0.1))
ax.set_xlim([151.0, 151.35])
ax.set_ylim([-34.15, -33.9])
plt.show(ax)
