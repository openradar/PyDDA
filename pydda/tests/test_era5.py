import pydda
import pyart
import numpy as np

from netCDF4 import Dataset
from scipy.interpolate import interp1d
from datetime import datetime


def test_add_era_5_field():
    Grid0 = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid0 = pydda.constraints.make_constraint_from_era5(
        Grid0, pydda.tests.sample_files.ERA_PATH, vel_field="corrected_velocity"
    )
    grid_time = datetime.strptime(
        Grid0.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
    )
    era_dataset = Dataset(pydda.tests.sample_files.ERA_PATH)

    z = era_dataset.variables["z"][:]
    u = era_dataset.variables["u"][:]
    lat = era_dataset.variables["latitude"][:]
    lon = era_dataset.variables["longitude"][:]
    base_time = datetime.strptime(
        era_dataset.variables["time"].units, "hours since %Y-%m-%d %H:%M:%S.%f"
    )
    time_step = np.argmin(np.abs(base_time - grid_time))
    lat_inds = np.where(
        np.logical_and(
            lat >= Grid0.point_latitude["data"].min(),
            lat <= Grid0.point_latitude["data"].max(),
        )
    )
    lon_inds = np.where(
        np.logical_and(
            lon >= Grid0.point_longitude["data"].min(),
            lon <= Grid0.point_longitude["data"].max(),
        )
    )

    z = z[time_step, :, lat_inds[0], lon_inds[0]]
    u = u[time_step, :, lat_inds[0], lon_inds[0]]

    nonans = np.logical_and(z < 25000.0, np.isfinite(u))

    z = z[nonans].flatten()
    u = u[nonans].flatten()

    # Interpolate era data onto u as a function of z
    u_interp = interp1d(z, u, kind="nearest")

    u_new_gridded = u_interp(
        np.asarray(Grid0.point_z["data"] + Grid0.radar_altitude["data"])
    )
    u_vertical = np.mean(u_new_gridded, axis=1).mean(axis=1)
    u_grid = np.mean(Grid0.fields["U_era5"]["data"], axis=1).mean(axis=1)

    np.testing.assert_allclose(u_grid, u_vertical, atol=4)


def test_era_initialization():
    Grid0 = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid0 = pydda.constraints.make_constraint_from_era5(
        Grid0, pydda.tests.sample_files.ERA_PATH, vel_field="corrected_velocity"
    )
    igrid = pydda.initialization.make_initialization_from_era5(
        Grid0, pydda.tests.sample_files.ERA_PATH, vel_field="corrected_velocity"
    )
    u = igrid.fields["u"]["data"]
    v = igrid.fields["v"]["data"]
    w = igrid.fields["w"]["data"]
    np.testing.assert_allclose(u, Grid0.fields["U_era5"]["data"], atol=1e-2)
    np.testing.assert_allclose(v, Grid0.fields["V_era5"]["data"], atol=1e-2)
    np.testing.assert_allclose(w, Grid0.fields["W_era5"]["data"], atol=1e-2)
