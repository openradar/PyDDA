"""

Example of a wind retrieval in a tornado over Sydney
----------------------------------------------------

This shows an example of how to retrieve winds from 4 radars over Sydney.

We use smoothing to decrease the magnitude of the updraft in the region of
the mesocyclone. The reduction of noise also helps the solution converge
much faster since the cost function is smoother and therefore less susecptible
to find a local minimum that is in noise.

The observational constraint is reduced to 0.01 from the usual 1 because we are factoring in
many more data points as we are using 4 radars instead of the two in the Darwin example.

This example uses pooch to download the data files.

"""

import pydda
import matplotlib.pyplot as plt
import numpy as np


grid1_path = pydda.tests.get_sample_file("grid1_sydney.nc")
grid2_path = pydda.tests.get_sample_file("grid2_sydney.nc")
grid3_path = pydda.tests.get_sample_file("grid3_sydney.nc")
grid4_path = pydda.tests.get_sample_file("grid4_sydney.nc")
grid1 = pydda.io.read_grid(grid1_path)
grid2 = pydda.io.read_grid(grid2_path)
grid3 = pydda.io.read_grid(grid3_path)
grid4 = pydda.io.read_grid(grid4_path)

# Set initialization and do retrieval
grid1 = pydda.initialization.make_constant_wind_field(grid1, vel_field="VRADH_corr")
new_grids, _ = pydda.retrieval.get_dd_wind_field(
    [grid1, grid2, grid3, grid4],
    Co=1e-2,
    Cm=256.0,
    Cx=10,
    Cy=10,
    Cz=10,
    vel_name="VRADH_corr",
    refl_field="DBZH",
    mask_outside_opt=True,
    wind_tol=0.5,
    max_iterations=200,
    engine="scipy",
)
# Make a neat plot
fig = plt.figure(figsize=(10, 7))
ax = pydda.vis.plot_horiz_xsection_quiver_map(
    new_grids,
    background_field="DBZH",
    level=3,
    show_lobes=False,
    bg_grid_no=3,
    vmin=0,
    vmax=60,
    quiverkey_len=20.0,
    w_vel_contours=[1.0, 3.0, 5.0, 10.0, 20.0],
    quiver_spacing_x_km=2.0,
    quiver_spacing_y_km=2.0,
    quiverkey_loc="top",
    colorbar_contour_flag=True,
    cmap="ChaseSpectral",
)
ax.set_xticks(np.arange(150.5, 153, 0.1))
ax.set_yticks(np.arange(-36, -32.0, 0.1))
ax.set_xlim([151.0, 151.35])
ax.set_ylim([-34.15, -33.9])
plt.show()
