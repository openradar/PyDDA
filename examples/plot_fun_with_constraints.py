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

This shows that your initial state and background are key to 
providing a physically realistic retrieval. Assuming a zero
background will likely result in false regions of convergence
and divergence that will generate artificial updrafts and downdrafts
at the edges of data coverage.

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
fig = plt.figure(figsize=(7, 3))

pydda.vis.plot_xz_xsection_streamlines(
    [cpol_grid, berr_grid], bg_grid_no=-1, level=50, w_vel_contours=[1, 3, 5, 8])
plt.show()

# Let's provide an initial state from the sounding
u_back = sounding[1].u_wind
v_back = sounding[1].v_wind
z_back = sounding[1].height
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(cpol_grid, sounding[1])

new_grids = pydda.retrieval.get_dd_wind_field([cpol_grid, berr_grid],
                                    u_init, v_init, w_init,
                                    u_back=u_back, v_back=v_back, z_back=z_back,
                                    Co=10.0, Cm=4096.0, frz=5000.0, Cb=1e-6,
                                    mask_outside_opt=False, wind_tol=0.2,
                                    engine="tensorflow")
fig = plt.figure(figsize=(7, 7))

pydda.vis.plot_xz_xsection_streamlines(
    new_grids,  bg_grid_no=-1, level=50, w_vel_contours=[1, 3, 5, 8])
plt.show()
# Let's see what happens when we use a zero initialization
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
    berr_grid, (0.0, 0.0, 0.0))    
new_grids = pydda.retrieval.get_dd_wind_field([cpol_grid, berr_grid],
                                    u_init, v_init, w_init,
                                    u_back=u_back, v_back=v_back, z_back=z_back,
                                    Co=1.0, Cm=128.0, frz=5000.0, Cb=1e-6,
                                    mask_outside_opt=False, wind_tol=0.2,
                                    engine="tensorflow")

fig = plt.figure(figsize=(7, 7))

pydda.vis.plot_xz_xsection_streamlines(
    new_grids, bg_grid_no=-1, level=50, w_vel_contours=[1, 3, 5, 8])
plt.show()

# Or, let's make the radar data more important!
new_grids = pydda.retrieval.get_dd_wind_field([cpol_grid, berr_grid],
                                    u_init, v_init, w_init,
                                    Co=100.0, Cm=128.0, frz=5000.0,
                                    mask_outside_opt=False, wind_tol=0.2,
                                    engine="tensorflow")
fig = plt.figure(figsize=(7, 7))

pydda.vis.plot_xz_xsection_streamlines(
    new_grids, bg_grid_no=-1, level=50, w_vel_contours=[1, 3, 5, 8])
plt.show()
