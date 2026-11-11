"""Unit tests for the terrain immersed-boundary constraint."""

import numpy as np
import xarray as xr

import pydda
from pydda.cost_functions import _cost_functions_numpy


def _make_grid():
    """Return a small grid whose lower level intersects a flat DEM."""
    return xr.Dataset(
        {
            "x": ("x", np.array([-1000.0, 0.0, 1000.0])),
            "y": ("y", np.array([-1000.0, 0.0, 1000.0])),
            "z": ("z", np.array([0.0, 500.0, 1000.0])),
            "origin_latitude": ("nradar", np.array([40.0])),
            "origin_longitude": ("nradar", np.array([-105.0])),
            "origin_altitude": ("nradar", np.array([0.0])),
        }
    )


def _make_flat_dem():
    """Return a constant orthometric DEM covering the test grid."""
    return xr.DataArray(
        np.full((5, 5), 250.0),
        coords={
            "lat": np.linspace(39.98, 40.02, 5),
            "lon": np.linspace(-105.02, -104.98, 5),
        },
        dims=("lat", "lon"),
    )


def test_terrain_ghost_cells_identifies_fluid_and_ghost_cells():
    terrain = pydda.terrain.terrain_ghost_cells(
        _make_grid(), _make_flat_dem(), smooth_cells=0.0, verbose=False
    )

    assert terrain["h"].shape == (3, 3)
    np.testing.assert_allclose(terrain["h"], 250.0)
    assert not np.any(terrain["fluid"][0])
    assert np.all(terrain["fluid"][1:])
    assert terrain["n_dropped"] == 0
    assert terrain["ghost_flat"].size > 0
    np.testing.assert_allclose(terrain["stencil_w"].sum(axis=1), 1.0)

    # A flat wall has a vertical normal, so the IBM operator leaves the
    # horizontal components unchanged and reverses the normal component.
    np.testing.assert_allclose(terrain["M"][:, 0, 0], 1.0)
    np.testing.assert_allclose(terrain["M"][:, 1, 1], 1.0)
    assert np.all(terrain["M"][:, 2, 2] < 0.0)


def test_terrain_cost_and_gradient_match_finite_difference():
    shape = (2, 2, 2)
    terrain = {
        "ghost_flat": np.array([0]),
        "stencil_flat": np.array([[7] * 8]),
        "stencil_w": np.full((1, 8), 1.0 / 8.0),
        "M": np.array([np.diag([1.0, 1.0, -1.0])]),
        "solid_free_flat": np.array([1, 2]),
    }
    rng = np.random.default_rng(7)
    fields = rng.normal(size=(3,) + shape)
    coeff = 2.5

    cost = _cost_functions_numpy.calculate_terrain_cost(*fields, terrain, coeff)
    gradient = _cost_functions_numpy.calculate_terrain_gradient(*fields, terrain, coeff)
    winds = fields.ravel()
    numerical = np.empty_like(winds)
    epsilon = 1.0e-6
    for index in range(winds.size):
        plus = winds.copy()
        minus = winds.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_fields = plus.reshape((3,) + shape)
        minus_fields = minus.reshape((3,) + shape)
        numerical[index] = (
            _cost_functions_numpy.calculate_terrain_cost(*plus_fields, terrain, coeff)
            - _cost_functions_numpy.calculate_terrain_cost(
                *minus_fields, terrain, coeff
            )
        ) / (2.0 * epsilon)

    assert cost > 0.0
    np.testing.assert_allclose(gradient, numerical, rtol=1.0e-5, atol=1.0e-7)


def test_terrain_cost_is_zero_for_a_reflected_ghost_value():
    terrain = {
        "ghost_flat": np.array([0]),
        "stencil_flat": np.array([[7] * 8]),
        "stencil_w": np.full((1, 8), 1.0 / 8.0),
        "M": np.array([np.diag([1.0, 1.0, -1.0])]),
        "solid_free_flat": np.array([], dtype=int),
    }
    u = np.zeros((2, 2, 2))
    v = np.zeros((2, 2, 2))
    w = np.zeros((2, 2, 2))
    u.ravel()[0], v.ravel()[0], w.ravel()[0] = 1.0, 2.0, -3.0
    u.ravel()[7], v.ravel()[7], w.ravel()[7] = 1.0, 2.0, 3.0

    assert _cost_functions_numpy.calculate_terrain_cost(u, v, w, terrain) == 0.0
    np.testing.assert_allclose(
        _cost_functions_numpy.calculate_terrain_gradient(u, v, w, terrain), 0.0
    )
