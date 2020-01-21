"""
Example on incorporating ASOS observations into PyDDA
-----------------------------------------------------------------

This shows how to download ASOS observations automatically from
the Iowa Mesonet Archive and integrate them into PyDDA.

Author: Robert C. Jackson

"""

import pydda
import pyart
import matplotlib.pyplot as plt
berr_grid = pyart.io.read_grid('grid1.20171004.095021.nc')

station_obs = pydda.constraints.get_iem_obs(berr_grid)
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(berr_grid, (0., 0., 0.))

new_grid = pydda.retrieval.get_dd_wind_field(
    [berr_grid], u_init, v_init, w_init, Co=0.0, Cpoint=10.0, points=station_obs, roi=5000.)

pydda.vis.plot_horiz_xsection_streamlines(new_grid, level=0)
plt.show()