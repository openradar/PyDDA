"""
Example on retrieving and plotting winds
----------------------------------------

This is a simple example for how to retrieve and plot winds from 2 radars
using PyDDA.

Author: Robert C. Jackson

"""

import pyart
import pydda
import numpy as np
from matplotlib import pyplot as plt


berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
        cpol_grid, (0.0, 0.0, 0.0), vel_field='corrected_velocity')

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init,
                                          v_init, w_init, Co=1.0, Cm=100.0,
                                          Jveltol=1e-3, cvtol=1e-4,
                                          Cx=0.0, Cy=0., Cz=0.0, Cb=0., 
                                          frz=5000.0, filter_window=5,
                                          mask_outside_opt=True, upper_bc=1,
                                          wind_tol=0.5, engine="auglag")
# Plot a horizontal cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_horiz_xsection_barbs(Grids, background_field='reflectivity',
                                    level=6,
                                    w_vel_contours=[5, 10, 15],
                                    barb_spacing_x_km=5.0,
                                    barb_spacing_y_km=15.0)
plt.show()
plt.savefig("Darwin_horiz.png")

# Plot a vertical X-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_xz_xsection_barbs(Grids, background_field='reflectivity',
                                 level=40,
                                 w_vel_contours=[5, 10, 15],
                                 barb_spacing_x_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.show()

# Plot a vertical Y-Z cross section
plt.figure(figsize=(9, 9))
pydda.vis.plot_yz_xsection_barbs(Grids, background_field='reflectivity', 
                                 level=40,
                                 w_vel_contours=[5, 10, 15],
                                 barb_spacing_y_km=10.0,
                                 barb_spacing_z_km=2.0)
plt.savefig('Darwin.png')
