"""
Guide and example on how to use nested grids with DataTrees
-----------------------------------------------------------

This is an example on how to use PyDDA's ability to handle nested
grids using xarray DataTrees. In this example, we load radars with
two pre-generated Cf/Radial grid. The fine grids are higher resolution
grids that are contained within the coarser grid.

The DataTree structure that PyDDA follows is:

::

    root
      |---nest_0/radar_1
      |---nest_0/radar_2
      |---nest_0/radar_n
      |---nest_1/radar_1
      |---nest_1/radar_2
      |---nest_1/radar_m

Each member of this tree is a DataTree itself. PyDDA will know if the
DataTree contains data from a radar when the name of the node begins
with **radar_**. The root node of each grid level, in this example,
**root** and **inner_nest** will contain the keyword arguments that are
inputs to :code:`pydda.retrieval.get_dd_wind_field` as attributes for the
tree. PyDDA will use the attributes at each level as the arguments for the
retrieval, allowing the user to vary the coefficients by grid level.

Using :code:`pydda.retrieval.get_dd_wind_field_nested` will allow PyDDA
to perform the retrieval on the 0th grid first. It will then
perform on the subsequent grid levels, using the previous nest as both the
horizontal boundary conditions and initialization for the retrieval in the next
nest. Finally, PyDDA will update the winds in the first grid by nearest-
neighbor interpolation of the latter grid into the overlapping portion between
the inner and outer grid level.

PyDDA will then return the retrieved wind fields as the "u", "v", and "w"
DataArrays inside each of the root nodes for each level, in this case
**root** and **inner_nest**.

"""

## Do imports
import pydda
import matplotlib.pyplot as plt
import warnings
from xarray import DataTree

warnings.filterwarnings("ignore")

"""
We will load pregenerated grids for this case.
"""
test_coarse0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse0.nc"))
test_coarse1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse1.nc"))
test_fine0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine0.nc"))
test_fine1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine1.nc"))

"""
Initalize with a zero wind field. We have HRRR data already generated for this case inside
the example data files to provide a model constraint.
"""
test_coarse0 = pydda.initialization.make_constant_wind_field(
    test_coarse0, (0.0, 0.0, 0.0)
)

"""
Specify the retrieval parameters at each level
"""
kwargs_dict = dict(
    Cm=256.0,
    Co=1e-2,
    Cx=50.0,
    Cy=50.0,
    Cz=50.0,
    Cmod=1e-5,
    model_fields=["hrrr"],
    refl_field="DBZ",
    wind_tol=0.5,
    max_iterations=150,
    engine="scipy",
)

"""
Enforce equal times for each grid. This is required for the DataTree structure since time is an
inherited dimension.
"""
test_coarse1["time"] = test_coarse0["time"]
test_fine0["time"] = test_coarse0["time"]
test_fine1["time"] = test_coarse1["time"]
"""

Provide the overlying grid structure as specified above.
"""
tree_dict = {
    "/nest_0/radar_ktlx": test_coarse0,
    "/nest_0/radar_kict": test_coarse1,
    "/nest_1/radar_ktlx": test_fine0,
    "/nest_1/radar_kict": test_fine1,
}

tree = DataTree.from_dict(tree_dict)
tree["/nest_0/"].attrs = kwargs_dict
tree["/nest_1/"].attrs = kwargs_dict

"""
Perform the retrieval
"""

grid_tree = pydda.retrieval.get_dd_wind_field_nested(tree)

"""
Plot the coarse grid output and finer grid output
"""

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
pydda.vis.plot_horiz_xsection_quiver(
    grid_tree["nest_0"],
    ax=ax[0],
    level=5,
    cmap="ChaseSpectral",
    vmin=-10,
    vmax=80,
    quiverkey_len=10.0,
    background_field="DBZ",
    bg_grid_no=1,
    w_vel_contours=[1, 2, 5, 10],
    quiver_spacing_x_km=50.0,
    quiver_spacing_y_km=50.0,
    quiverkey_loc="bottom_right",
)
pydda.vis.plot_horiz_xsection_quiver(
    grid_tree["nest_1"],
    ax=ax[1],
    level=5,
    cmap="ChaseSpectral",
    vmin=-10,
    vmax=80,
    quiverkey_len=10.0,
    background_field="DBZ",
    bg_grid_no=1,
    w_vel_contours=[1, 2, 5, 10],
    quiver_spacing_x_km=50.0,
    quiver_spacing_y_km=50.0,
    quiverkey_loc="bottom_right",
)

plt.show()
