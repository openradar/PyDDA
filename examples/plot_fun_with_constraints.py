"""
Example on geographic plotting and constraint variation
-------------------------------------------------------

In this example we show how to plot wind fields on a map and change
the default constraint coefficients using PyDDA.

This shows how important it is to have the proper intitial state and
constraints when you derive your wind fields. In the first figure,
the sounding was used as the initial state, but for the latter
two examples we use a zero initial state which provides for more 
questionable winds at the edges of the Dual Doppler Lobes.

"""

import pydda
import pyart
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

# Load our radar data
sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
    berr_grid, (0.0, 0.0, 0.0))

# Let's make a plot on a map
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

pydda.vis.plot_horiz_xsection_streamlines_map(
    [cpol_grid, berr_grid], ax=ax, bg_grid_no=-1, level=7, w_vel_contours=[3, 5, 8])
plt.show()

# Let's see what happens when we use a zero initialization
new_grids = pydda.retrieval.get_dd_wind_field([cpol_grid, berr_grid],
                                    u_init, v_init, w_init,
                                    Co=1.0, Cm=1500.0, frz=5000.0,
                                    mask_outside_opt=False)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

pydda.vis.plot_horiz_xsection_streamlines_map(
    new_grids, ax=ax, bg_grid_no=-1, level=7, w_vel_contours=[3, 5, 8])
plt.show()

# Or, let's make the radar data more important!
new_grids = pydda.retrieval.get_dd_wind_field([cpol_grid, berr_grid],
                                    u_init, v_init, w_init,
                                    Co=1.0, Cm=1500.0, frz=5000.0,
                                    mask_outside_opt=False)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

pydda.vis.plot_horiz_xsection_streamlines_map(
    new_grids, ax=ax, bg_grid_no=-1, level=7, w_vel_contours=[3, 5, 8])
plt.show()
