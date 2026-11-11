"""
Denver hail case: flat versus DEM immersed-boundary retrieval
--------------------------------------------------------------

This example shows the complete terrain-boundary workflow used for the
18 June 2018 Denver hail case.  The selected volume scans are close to
23:30 UTC:

* KFTG: ``KFTG20180618_233053_V06``
* KCYS: ``KCYS20180618_233041_V06``

The Level-II files are available from the public Unidata NEXRAD archive.
The DEM is an SRTM v3 (approximately EGM96 orthometric) mosaic.  The large
input files are intentionally not shipped with PyDDA.  Run the case-data
preparation workflow first, or set ``CASE_DIR`` to a directory containing the
two Py-ART grids and the DEM NetCDF file.

The example compares the default flat lower boundary with the DEM-driven
immersed-boundary condition.  On sloping terrain, the physical condition is
``u dot n = 0``; ``w = 0`` is the special case for a horizontal surface.
"""

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydda
import xarray as xr


# The case-data directory is created by the reproducible workflow scripts in
# scripts/run_denver_hail_20180618_2330.py.  Change this path when running the
# example from another checkout.
CASE_DIR = Path("data/denver_hail_20180618_2330")
GRID_DIR = CASE_DIR / "grid"
TERRAIN_DIR = CASE_DIR / "retrieval_terrain"
DEM_PATH = Path("data/denver_hail_20180619/dem/denver_srtm_3arcsec.nc")
RUN_RETRIEVAL = False


def retrieve(grids, terrain=None):
    """Run a default PyDDA retrieval, optionally with terrain IBM enabled."""
    initialized = [
        pydda.initialization.make_constant_wind_field(
            grid, (0.0, 0.0, 0.0), vel_field="velocity"
        )
        for grid in grids
    ]
    kwargs = {"vel_name": "velocity", "engine": "scipy"}
    if terrain is not None:
        kwargs.update(Cterrain=1.0, terrain=terrain)
    return pydda.retrieval.get_dd_wind_field(initialized, **kwargs)[0]


def plot_3km_difference(flat, terrain, terrain_geometry):
    """Plot terrain-aware minus flat winds at 3 km."""
    flat_u = flat["u"].values.squeeze()
    flat_v = flat["v"].values.squeeze()
    flat_w = flat["w"].values.squeeze()
    terrain_u = terrain["u"].values.squeeze()
    terrain_v = terrain["v"].values.squeeze()
    terrain_w = terrain["w"].values.squeeze()

    delta_w = terrain_w[6] - flat_w[6]
    delta_speed = np.hypot(terrain_u[6], terrain_v[6]) - np.hypot(flat_u[6], flat_v[6])
    x = flat.x.values / 1000.0
    y = flat.y.values / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, field, title, label in (
        (axes[0], delta_w, r"$\Delta w$ [m/s]", "Updraft difference"),
        (
            axes[1],
            delta_speed,
            r"$\Delta |V|$ [m/s]",
            "Horizontal wind-speed difference",
        ),
    ):
        limit = np.nanpercentile(np.abs(field), 98)
        mesh = ax.pcolormesh(
            x, y, field, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto"
        )
        ax.contour(x, y, field, levels=[0.0], colors="k", linewidths=0.7)
        ax.set(xlim=(-75, 75), ylim=(-75, 75), xlabel="X [km]", ylabel="Y [km]")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label=label)
    fig.suptitle("Denver hail case: terrain-aware minus flat retrieval at 3 km")
    return fig


if GRID_DIR.joinpath("KFTG_grid.nc").exists() and DEM_PATH.exists():
    radar_grids = [
        pydda.io.read_grid(GRID_DIR / "KFTG_grid.nc"),
        pydda.io.read_grid(GRID_DIR / "KCYS_grid.nc"),
    ]
    dem = xr.open_dataarray(DEM_PATH)
    terrain_geometry = pydda.terrain.terrain_ghost_cells(radar_grids[0], dem)

    if RUN_RETRIEVAL:
        flat_grids = retrieve(radar_grids)
        terrain_grids = retrieve(radar_grids, terrain=terrain_geometry)
    elif TERRAIN_DIR.joinpath("KFTG_denver_hail_20180618_2330_retrieval.nc").exists():
        # Use precomputed fields for a fast gallery build. Set RUN_RETRIEVAL=True
        # above to execute both solves from the gridded radar data.
        flat_grids = pydda.io.read_grid(
            CASE_DIR / "retrieval/KFTG_denver_hail_20180618_2330_retrieval.nc"
        )
        terrain_grids = pydda.io.read_grid(
            TERRAIN_DIR / "KFTG_denver_hail_20180618_2330_retrieval.nc"
        )
    else:
        flat_grids = terrain_grids = None

    if flat_grids is not None:
        plot_3km_difference(flat_grids, terrain_grids, terrain_geometry)
else:
    print(
        "Denver case data not found; prepare the case data first, then rerun "
        "this example."
    )
