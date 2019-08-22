"""

Example of a wind retrieval from KVNX and 2 XSAPRs in Oklahoma
--------------------------------------------------------------

This shows an example of how to combine data from 3 radars and HRRR in
the vicinity of the SGP site in North Central Oklahoma.

Grided netCDF files are downloadable from:
https://drive.google.com/drive/u/1/folders/1pcQxWRJV78xuJePTZnlXPPpMe1qut0ie

.. image:: ../../arm_figure.png

"""

import pyart
import pydda
import matplotlib.pyplot as plt
import numpy as np
import urllib

hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20181004/' +
            'hrrr.t10z.wrfprsf00.grib2')
grid0 = pyart.io.read_grid('grid0.20171004.095021.nc')
grid1 = pyart.io.read_grid('grid1.20171004.095021.nc')
grid2 = pyart.io.read_grid('grid2.20171004.095021.nc')

urllib.request.urlretrieve(hrrr_url, 'test.grib2')
grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid0,
                                                         'test.grib2')

# Set initialization and do retrieval
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(grid1)
new_grids = pydda.retrieval.get_dd_wind_field([grid0, grid1, grid2],
                                              u_init, v_init, w_init, Co=0.1, Cm=100.0,
                                              model_fields=["hrrr"],
                                              mask_outside_opt=True)
# Make a neat plot
fig = plt.figure(figsize=(10,10))
ax = pydda.vis.plot_horiz_xsection_quiver(new_grids, background_field='reflectivity', level=3,
                                          show_lobes=False, bg_grid_no=0, vmin=0, vmax=60,
                                          quiverkey_len=10.0,
                                          quiver_spacing_x_km=2.0, quiver_spacing_y_km=2.0,
                                          quiverkey_loc='top', colorbar_contour_flag=True,
                                          cmap='pyart_HomeyerRainbow')
ax.set_xlim([-20, 40])
ax.set_ylim([-20, 40])
plt.show(ax)
