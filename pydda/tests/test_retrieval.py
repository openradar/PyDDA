#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:43 2018

@author: rjackson
"""

import pydda
import pyart
import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from datatree import DataTree

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from distributed import Client, LocalCluster
from copy import deepcopy


def test_make_updraft_from_convergence_field():
    """Do we have an updraft in a region of convergence and divergence?"""

    Grid = pyart.testing.make_empty_grid(
        (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000))
    )

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field("one_field", {"data": odata3, "_FillValue": -9999.0})

    wind_vel = 10.0
    z_ground = 500.0
    z_top = 5000.0
    radius = 3000.0
    back_u = 10.0
    back_v = 10.0
    x_center = 0.0
    y_center = 0.0
    Grid = pydda.io.read_from_pyart_grid(Grid)
    u, v, w = pydda.tests.make_test_divergence_field(
        Grid, wind_vel, z_ground, z_top, radius, back_u, back_v, x_center, y_center
    )

    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid],
        u,
        v,
        w,
        Co=0.0,
        Cz=0,
        Cm=500.0,
        Cmod=0.0,
        mask_outside_opt=False,
        vel_name="one_field",
        refl_field="one_field",
    )
    new_w = new_grids[0]["w"].values

    # We should have a pretty strong updraft in the retrieval!
    assert np.ma.max(new_w > 3)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_twpice_case_jax():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )

    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        wind_tol=0.1,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        refl_field="reflectivity",
        frz=5000.0,
        engine="jax",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_twpice_case_tensorflow():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )
    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        wind_tol=0.1,
        refl_field="reflectivity",
        frz=5000.0,
        engine="tensorflow",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


def test_twpice_case():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )
    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        wind_tol=0.1,
        refl_field="reflectivity",
        frz=5000.0,
        engine="scipy",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


def test_smoothing():
    """A field of random numbers from 0 to 1
    should smooth out to near 0.5"""
    Grid = pyart.testing.make_empty_grid(
        (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000))
    )

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field("one_field", {"data": odata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    u = np.random.random((20, 40, 40))
    v = np.random.random((20, 40, 40))
    w = np.zeros((20, 40, 40))
    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid],
        u_init=u,
        v_init=v,
        w_init=w,
        Co=0.0,
        Cx=1e-4,
        Cy=1e-4,
        Cm=0.0,
        Cmod=0.0,
        mask_outside_opt=False,
        vel_name="one_field",
        refl_field="one_field",
    )
    new_u = new_grids[0]["u"].values
    new_v = new_grids[0]["v"].values
    assert new_u.std() < u.std()
    assert new_v.std() < v.std()


def test_model_constraint():
    """A retrieval with just the model constraint should converge
    to the model constraint."""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)

    """ Make fake model grid of just U = 1 m/s everywhere"""
    Grid0["U_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])
    Grid0["V_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])
    Grid0["W_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])

    u_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()
    v_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()
    w_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()

    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0],
        u_init,
        v_init,
        w_init,
        Co=0.0,
        Cx=0.0,
        Cy=0.0,
        Cm=0.0,
        Cmod=1.0,
        mask_outside_opt=False,
        vel_name="corrected_velocity",
        refl_field="reflectivity",
        model_fields=["fakemodel"],
    )

    np.testing.assert_allclose(
        new_grids[0]["u"].values, Grid0["U_fakemodel"].values, atol=1e-2
    )
    np.testing.assert_allclose(
        new_grids[0]["v"].values, Grid0["V_fakemodel"].values, atol=1e-2
    )


@pytest.mark.mpl_image_compare(tolerance=50)
def test_nested_retrieval():
    test_coarse0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse0.nc"))
    test_coarse1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse1.nc"))
    test_fine0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine0.nc"))
    test_fine1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine1.nc"))

    test_coarse0 = pydda.initialization.make_constant_wind_field(
        test_coarse0, (0.0, 0.0, 0.0)
    )

    kwargs_dict = dict(
        Cm=256.0,
        Co=1e-2,
        Cx=150.0,
        Cy=150.0,
        Cz=150.0,
        Cmod=1e-5,
        model_fields=["hrrr"],
        refl_field="DBZ",
        wind_tol=0.5,
        max_iterations=50,
        filter_order=3,
        engine="scipy",
    )

    tree_dict = {
        "/coarse/radar_ktlx": test_coarse0,
        "/coarse/radar_kict": test_coarse1,
        "/coarse/fine/radar_ktlx": test_fine0,
        "/coarse/fine/radar_kict": test_fine1,
    }

    tree = DataTree.from_dict(tree_dict)
    tree["/coarse/"].attrs = kwargs_dict
    tree["/coarse/fine"].attrs = kwargs_dict

    grid_tree = pydda.retrieval.get_dd_wind_field_nested(tree)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    pydda.vis.plot_horiz_xsection_quiver(
        grid_tree["coarse"],
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
        grid_tree["coarse/fine"],
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
    return fig
