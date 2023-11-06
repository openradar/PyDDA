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
    assert names == ["P28", "WLD", "WDG", "SWO", "END"]


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
