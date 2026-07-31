#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:50:58 2018

@author: rjackson
"""

import pydda
import pyart
import numpy as np
from datetime import datetime


def test_make_const_wind_field():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)

    Grid = pydda.initialization.make_constant_wind_field(
        Grid, wind=(2.0, 3.0, 4.0), vel_field="zero_field"
    )

    assert np.all(Grid["u"].values == 2.0)
    assert np.all(Grid["v"].values == 3.0)
    assert np.all(Grid["w"].values == 4.0)


def test_make_wind_field_from_profile():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    height = np.arange(0, 10000, 100)
    u_sound = np.ones(height.shape)
    v_sound = np.ones(height.shape)

    profile = pyart.core.HorizontalWindProfile.from_u_and_v(height, u_sound, v_sound)

    Grid = pydda.initialization.make_wind_field_from_profile(
        Grid, profile, vel_field="zero_field"
    )

    assert np.all(np.round(Grid["u"].values) == 1)
    assert np.all(np.round(Grid["v"].values) == 1)
    assert np.all(Grid["w"].values == 0.0)


def test_get_iem_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    station_obs = pydda.constraints.get_iem_obs(Grid)
    names = [x["site_id"] for x in station_obs]
    assert "P28" in names
    assert "WLD" in names
    assert "WDG" in names
    assert "SWO" in names
    assert "END" in names


def test_hrrr_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    Grid = pydda.constraints.add_hrrr_constraint_to_grid(
        Grid, pydda.tests.get_sample_file("ruc2anl_130_20110520_0800_001.grb2")
    )

    assert Grid["U_hrrr"].max() > 15
    assert Grid["V_hrrr"].max() > 15
    assert Grid["W_hrrr"].max() > 0


def test_hrrr_uv_rotated_to_true_north():
    # HRRR's u and v are relative to its Lambert Conformal Conic grid, not
    # true north. add_hrrr_constraint_to_grid must rotate them before they
    # are interpolated onto the analysis grid.
    grid_shape = (4, 4, 4)
    grid_limits = ((0, 5000.0), (-50000.0, 50000.0), (-50000.0, 50000.0))
    file_path = pydda.tests.get_sample_file("ruc2anl_130_20110520_0800_001.grb2")

    def make_grid(origin_lat, origin_lon):
        Grid = pyart.testing.make_empty_grid(grid_shape, grid_limits)
        for field in ("origin_latitude", "radar_latitude"):
            getattr(Grid, field)["data"] = np.array([origin_lat])
        for field in ("origin_longitude", "radar_longitude"):
            getattr(Grid, field)["data"] = np.array([origin_lon])
        Grid.init_point_longitude_latitude()
        fdata3 = np.zeros(grid_shape)
        Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
        return pydda.io.read_from_pyart_grid(Grid)

    # Place the analysis domain well away from HRRR's -97.5 degree central
    # meridian, where the grid-rotation correction has a large, easily
    # measurable effect.
    Grid = make_grid(38.5, -85.0)
    Grid = pydda.constraints.add_hrrr_constraint_to_grid(Grid, file_path)
    u_true_north = Grid["U_hrrr"].values
    v_true_north = Grid["V_hrrr"].values

    # Setting both true latitudes to the equator collapses the Lambert
    # Conformal cone factor to zero, which makes the rotation a no-op.
    # This gives an otherwise identical baseline of the raw, grid-relative
    # winds to compare against.
    Grid_grid_relative = make_grid(38.5, -85.0)
    Grid_grid_relative = pydda.constraints.add_hrrr_constraint_to_grid(
        Grid_grid_relative, file_path, truelat1=0.0, truelat2=0.0
    )
    u_grid_relative = Grid_grid_relative["U_hrrr"].values
    v_grid_relative = Grid_grid_relative["V_hrrr"].values

    # The rotation should meaningfully change the wind components this far
    # from the central meridian...
    assert np.abs(u_true_north - u_grid_relative).max() > 0.5
    assert np.abs(v_true_north - v_grid_relative).max() > 0.5

    # ...while preserving wind speed, since a coordinate rotation cannot
    # change the magnitude of the wind vector.
    speed_true_north = np.sqrt(u_true_north**2 + v_true_north**2)
    speed_grid_relative = np.sqrt(u_grid_relative**2 + v_grid_relative**2)
    np.testing.assert_allclose(speed_true_north, speed_grid_relative, rtol=1e-4)
