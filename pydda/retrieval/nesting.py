import pyart
import numpy as np
import xarray as xr

from scipy.interpolate import RegularGridInterpolator
from .wind_retrieve import get_dd_wind_field
from datatree import DataTree


def get_dd_wind_field_nested(grid_tree: DataTree, **kwargs):
    """
    Does a wind retrieval over nested grids. The nested grids are created using PyART's
    :func:`pyart.map.grid_from_radars` function and then placed into a tree structure using
    :func:`dataTree`s. Each node of the tree has three parameters:
    .. list-table:: Title
        :widths: 25 100
        :header-rows: 1

        * - Dictionary key
          - Description
        * - input_grids
          - The list of PyART grids for the given level of the grid
        * - kwargs
          - The list of key word arguments for input to the :py:func:`pydda.retrieval.get_dd_wind_field` function for the set of grids.
        * - children
          - The list of trees that are the children of this node.

    The function will output the same tree, with the list of output grids of each level output to the 'output_grids'
    member of the tree structure. If *kwargs* is set to None, then the input keyword arguments will be
    used throughout the retrieval.
    """

    # Look for radars in current level
    child_list = list(grid_tree.children.keys())
    grid_list = []
    rad_names = []
    for child in child_list:
        if "radar" in child:
            grid_list.append(grid_tree[child].to_dataset())
            rad_names.append(child)

    if len(list(grid_tree.attrs.keys())) == 0 and len(grid_list) > 0:
        output_grids, output_parameters = get_dd_wind_field(grid_list, **kwargs)
    elif len(grid_list) > 0:
        my_kwargs = grid_tree.attrs
        output_grids, output_parameters = get_dd_wind_field(grid_list, **my_kwargs)
        output_parameters = output_parameters.__dict__
        grid_tree["weights"] = xr.DataArray(
            output_parameters.pop("weights"), dims=("nradar", "z", "y", "x")
        )
        grid_tree["bg_weights"] = xr.DataArray(
            output_parameters.pop("bg_weights"), dims=("z", "y", "x")
        )
        grid_tree["model_weights"] = xr.DataArray(
            output_parameters.pop("model_weights"), dims=("nmodel", "z", "y", "x")
        )
        output_parameters.pop("u_model")
        output_parameters.pop("v_model")
        output_parameters.pop("w_model")
        grid_tree["output_parameters"] = xr.DataArray([], attrs=output_parameters)

        grid_tree.__setitem__("u", output_grids[0]["u"])
        grid_tree.__setitem__("v", output_grids[0]["v"])
        grid_tree.__setitem__("w", output_grids[0]["w"])
        grid_tree["u"].attrs = output_grids[0]["u"].attrs
        grid_tree["v"].attrs = output_grids[0]["v"].attrs
        grid_tree["w"].attrs = output_grids[0]["w"].attrs

    if child_list == []:
        return grid_tree

    nests = []
    for child in child_list:
        if "radar_" not in child:
            nests.append(child)
    nests = sorted(nests)
    for child in nests:
        # Only update child initalization if we are not in parent node
        if len(grid_list) > 0:
            temp_src = grid_tree[rad_names[0]].to_dataset()
            temp_src["u"] = grid_tree.ds["u"]
            temp_src["v"] = grid_tree.ds["v"]
            temp_src["w"] = grid_tree.ds["w"]
            input_grids = make_initialization_from_other_grid(
                temp_src, grid_tree.children[child][rad_names[0]].to_dataset()
            )

            if "u" not in grid_tree.children[child][rad_names[0]].ds.variables.keys():
                grid_tree.children[child][rad_names[0]].__setitem__(
                    "u",
                    xr.zeros_like(
                        grid_tree.children[child][rad_names[0]].ds[
                            grid_tree.children[child][rad_names[0]].ds.attrs[
                                "first_grid_name"
                            ]
                        ]
                    ),
                )
                grid_tree.children[child][rad_names[0]]["u"].attrs = input_grids[
                    "u"
                ].attrs

            if "v" not in grid_tree.children[child][rad_names[0]].ds.variables.keys():
                grid_tree.children[child][rad_names[0]].__setitem__(
                    "v",
                    xr.zeros_like(
                        grid_tree.children[child][rad_names[0]].ds[
                            grid_tree.children[child][rad_names[0]].ds.attrs[
                                "first_grid_name"
                            ]
                        ]
                    ),
                )
                grid_tree.children[child][rad_names[0]]["v"].attrs = input_grids[
                    "v"
                ].attrs

            if "w" not in grid_tree.children[child][rad_names[0]].ds.variables.keys():
                grid_tree.children[child][rad_names[0]].__setitem__(
                    "w",
                    xr.zeros_like(
                        grid_tree.children[child][rad_names[0]].ds[
                            grid_tree.children[child][rad_names[0]].ds.attrs[
                                "first_grid_name"
                            ]
                        ]
                    ),
                )
                grid_tree.children[child][rad_names[0]]["w"].attrs = input_grids[
                    "w"
                ].attrs

            grid_tree.children[child][rad_names[0]].__setitem__("u", input_grids["u"])
            grid_tree.children[child][rad_names[0]].__setitem__("v", input_grids["v"])
            grid_tree.children[child][rad_names[0]].__setitem__("w", input_grids["w"])

        grid_tree.children[child].parent.attrs["const_boundary_cond"] = True

        temp_tree = get_dd_wind_field_nested(grid_tree.children[child])
        grid_tree.children[child].__setitem__("u", temp_tree["u"])
        grid_tree.children[child].__setitem__("v", temp_tree["v"])
        grid_tree.children[child].__setitem__("w", temp_tree["w"])
        grid_tree.children[child]["u"].attrs = temp_tree.ds["u"].attrs
        grid_tree.children[child]["v"].attrs = temp_tree.ds["v"].attrs
        grid_tree.children[child]["w"].attrs = temp_tree.ds["w"].attrs

    # Update parent grids from children
    if len(rad_names) > 0:
        for child in nests:
            temp_src = grid_tree.children[child][rad_names[0]].to_dataset()
            temp_src["u"] = grid_tree.children[child].ds["u"]
            temp_src["v"] = grid_tree.children[child].ds["v"]
            temp_src["w"] = grid_tree.children[child].ds["w"]
            temp_dest = grid_tree[rad_names[0]].to_dataset()
            temp_dest["u"] = grid_tree.ds["u"]
            temp_dest["v"] = grid_tree.ds["v"]
            temp_dest["w"] = grid_tree.ds["w"]
            output_grids = make_initialization_from_other_grid(
                temp_src,
                temp_dest,
            )
            grid_tree.__setitem__("u", output_grids["u"])
            grid_tree.__setitem__("v", output_grids["v"])
            grid_tree.__setitem__("w", output_grids["w"])
            grid_tree["u"].attrs = output_grids["u"].attrs
            grid_tree["v"].attrs = output_grids["v"].attrs
            grid_tree["w"].attrs = output_grids["w"].attrs
    return grid_tree


def make_initialization_from_other_grid(grid_src, grid_dest, method="linear"):
    """
    This function will create an initaliation by interpolating a wind field
    from a grid with a different specification than the analysis grid. This
    allows, for example, for interpolating a coarser grid onto a finer grid
    for further refinement of the retrieval. The source and destination grid
    must have the same origin point.

    Parameters
    ----------
    grid_src: Grid
        The grid to interpolate.
    grid_dst: Grid
        The destination analysis grid to interpolate the source grid on.
    method: str
        Interpolation method to use
    Returns
    -------
    grid: Grid
        The grid with the u, v, and w from the source grid interpolated.
    """
    if not np.all(
        grid_src["origin_latitude"].values == grid_dest["origin_latitude"].values
    ):
        raise ValueError("Source and destination grid must have same lat/lon origin!")

    if not np.all(
        grid_src["origin_longitude"].values == grid_dest["origin_longitude"].values
    ):
        raise ValueError("Source and destination grid must have same lat/lon origin!")

    if not np.all(
        grid_src["origin_altitude"].values == grid_dest["origin_altitude"].values
    ):
        correction_factor = (
            grid_dest["origin_altitude"].values - grid_src["origin_altitude"].values
        )
    else:
        correction_factor = 0

    u_src = grid_src["u"].values.squeeze()
    v_src = grid_src["v"].values.squeeze()
    w_src = grid_src["w"].values.squeeze()
    x_src = grid_src["x"].values
    y_src = grid_src["y"].values
    z_src = grid_src["z"].values

    x_dst = grid_dest["point_x"].values
    y_dst = grid_dest["point_y"].values
    z_dst = grid_dest["point_z"].values - correction_factor

    x_dst_p = grid_dest["x"].values
    y_dst_p = grid_dest["y"].values
    z_dst_p = grid_dest["z"].values - correction_factor

    # Subset destination grid coordinates
    x_src_min = x_src.min()
    x_src_max = x_src.max()
    y_src_min = y_src.min()
    y_src_max = y_src.max()
    z_src_min = z_src.min()
    z_src_max = z_src.max()
    subset_z = np.argwhere(
        np.logical_and(z_dst_p >= z_src_min, z_dst_p <= z_src_max)
    ).astype(int)
    subset_y = np.argwhere(
        np.logical_and(y_dst_p >= y_src_min, y_dst_p <= y_src_max)
    ).astype(int)
    subset_x = np.argwhere(
        np.logical_and(x_dst_p >= x_src_min, x_dst_p <= x_src_max)
    ).astype(int)

    u_interp = RegularGridInterpolator((z_src, y_src, x_src), u_src, method=method)
    v_interp = RegularGridInterpolator((z_src, y_src, x_src), v_src, method=method)
    w_interp = RegularGridInterpolator((z_src, y_src, x_src), w_src, method=method)
    u_dest = u_interp(
        (
            z_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            y_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            x_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
        )
    )
    v_dest = v_interp(
        (
            z_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            y_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            x_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
        )
    )
    w_dest = w_interp(
        (
            z_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            y_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
            x_dst[
                int(subset_z[0]) : int(subset_z[-1] + 1),
                int(subset_y[0]) : int(subset_y[-1] + 1),
                int(subset_x[0]) : int(subset_x[-1] + 1),
            ],
        )
    )

    if "u" not in grid_dest.variables.keys():
        grid_dest["u"] = xr.DataArray(
            np.expand_dims(u_dest, 0),
            dims=("time", "z", "y", "x"),
            attrs=grid_src["u"].attrs,
        )
    else:
        grid_dest["u"][
            :,
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = np.expand_dims(u_dest, 0)

    grid_dest["u"].attrs = grid_src["u"].attrs
    if "v" not in grid_dest.variables.keys():
        grid_dest["v"] = xr.DataArray(
            np.expand_dims(v_dest, 0),
            dims=("time", "z", "y", "x"),
            attrs=grid_src["v"].attrs,
        )
    else:
        grid_dest["v"][
            :,
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = np.expand_dims(v_dest, 0)
    grid_dest["v"].attrs = grid_src["v"].attrs
    if "w" not in grid_dest.variables.keys():
        grid_dest["w"] = xr.DataArray(
            np.expand_dims(w_dest, 0),
            dims=("time", "z", "y", "x"),
            attrs=grid_src["w"].attrs,
        )
    else:
        grid_dest["w"][
            :,
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = w_dest
    grid_dest["w"].attrs = grid_src["w"].attrs
    return grid_dest
