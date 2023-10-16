import pyart
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from .wind_retrieve import get_dd_wind_field


def get_dd_wind_field_nested(grid_tree: dict, **kwargs):
    """
    Does a wind retrieval over nested grids. The nested grids are created using PyART's
    :func:`pyart.map.grid_from_radars` function and then placed into a tree structure using
    dictionaries. Each node of the tree has three parameters:
    'input_grids': The list of PyART grids for the given level of the grid
    'kwargs': The list of key word arguments for input to the get_dd_wind_field function for the set of grids.
    If this is None, then the default keyword arguments are carried from the keyword arguments of this function.
    'children': The list of trees that are the children of this node.

    The function will output the same tree, with the list of output grids of each level output to the 'output_grids'
    member of the tree structure.
    """

    grid_list = grid_tree["input_grids"]
    if grid_tree["kwargs"] is None:
        grid_tree["output_grids"], grid_tree["output_parameters"] = get_dd_wind_field(
            grid_list, **kwargs
        )
    else:
        for parameters in grid_tree["kwargs"].keys():
            kwargs[parameters] = grid_tree["kwargs"][parameters]
        grid_tree["output_grids"], grid_tree["output_parameters"] = get_dd_wind_field(
            grid_list, **kwargs
        )

    if "children" not in grid_tree.keys():
        return grid_tree

    for child in grid_tree["children"].keys():
        grid_tree["children"][child]["input_grids"][
            0
        ] = make_initialization_from_other_grid(
            grid_tree["output_grids"][0], grid_tree["children"][child]["input_grids"][0]
        )
        grid_tree["children"][child]["kwargs"]["const_boundary_cond"] = True
        grid_tree["children"][child] = get_dd_wind_field_nested(
            grid_tree["children"][child]
        )

    # Update parent grids from children
    for child in grid_tree["children"].keys():
        grid_tree["output_grids"][0] = make_initialization_from_other_grid(
            grid_tree["children"][child]["output_grids"][0],
            grid_tree["output_grids"][0],
        )

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
    if not grid_src.origin_latitude["data"] == grid_dest.origin_latitude["data"]:
        raise ValueError("Source and destination grid must have same lat/lon origin!")

    if not grid_src.origin_longitude["data"] == grid_dest.origin_longitude["data"]:
        raise ValueError("Source and destination grid must have same lat/lon origin!")

    if not grid_src.origin_altitude["data"] == grid_dest.origin_altitude["data"]:
        correction_factor = (
            grid_dest.origin_altitude["data"] - grid_src.origin_altitude["data"]
        )
    else:
        correction_factor = 0

    u_src = grid_src.fields["u"]["data"]
    v_src = grid_src.fields["v"]["data"]
    w_src = grid_src.fields["w"]["data"]
    x_src = grid_src.x["data"]
    y_src = grid_src.y["data"]
    z_src = grid_src.z["data"]

    x_dst = grid_dest.point_x["data"]
    y_dst = grid_dest.point_y["data"]
    z_dst = grid_dest.point_z["data"] - correction_factor

    x_dst_p = grid_dest.x["data"]
    y_dst_p = grid_dest.y["data"]
    z_dst_p = grid_dest.z["data"] - correction_factor

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

    if "u" not in grid_dest.fields.keys():
        u_field = {}
        u_field["data"] = u_dest
        u_field["data"] = u_dest
        u_field["standard_name"] = "u_wind"
        u_field["long_name"] = "meridional component of wind velocity"
        grid_dest.add_field("u", u_field, replace_existing=True)
    else:
        v_field = grid_dest.fields["v"]["data"][
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = u_dest

    if "v" not in grid_dest.fields.keys():
        v_field = {}
        v_field["data"] = v_dest
        v_field["standard_name"] = "v_wind"
        v_field["long_name"] = "zonal component of wind velocity"
        grid_dest.add_field("v", v_field, replace_existing=True)
    else:
        v_field = grid_dest.fields["v"]["data"][
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = v_dest

    if "w" not in grid_dest.fields.keys():
        w_field = {}
        w_field["data"] = w_dest
        w_field["data"] = w_dest
        w_field["standard_name"] = "w_wind"
        w_field["long_name"] = "vertical component of wind velocity"
        grid_dest.add_field("w", w_field, replace_existing=True)
    else:
        grid_dest.fields["w"]["data"][
            int(subset_z[0]) : int(subset_z[-1] + 1),
            int(subset_y[0]) : int(subset_y[-1] + 1),
            int(subset_x[0]) : int(subset_x[-1] + 1),
        ] = w_dest

    return grid_dest
