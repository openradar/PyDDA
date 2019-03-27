#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:43 2018

@author: rjackson
"""

import pydda
import pyart
import numpy as np

from distributed import Client, LocalCluster
from copy import deepcopy


def test_make_updraft_from_convergence_field():
    """ Do we have an updraft in a region of convergence and divergence? """

    Grid = pyart.testing.make_empty_grid(
            (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000)))

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field('zero_field', {'data': fdata3, '_FillValue': -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field('one_field', {'data': odata3, '_FillValue': -9999.0})

    wind_vel = 10.0
    z_ground = 500.0
    z_top = 5000.0
    radius = 3000.0
    back_u = 10.0
    back_v = 10.0
    x_center = 0.0
    y_center = 0.0
    u, v, w = pydda.tests.make_test_divergence_field(
        Grid, wind_vel, z_ground, z_top, radius, back_u, back_v,
        x_center, y_center)

    new_grids = pydda.retrieval.get_dd_wind_field([Grid], u, v, w, Co=0.0,
                                                  Cz=0, Cm=500.0, Cmod=0.0,
                                                  mask_outside_opt=False,
                                                  vel_name='one_field',
                                                  refl_field='one_field')
    new_w = new_grids[0].fields['w']['data']

    # We should have a pretty strong updraft in the retrieval!
    assert np.ma.max(new_w > 3)


def test_twpice_case():
    """ Use a test case from TWP-ICE """
    Grid0 = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field='corrected_velocity')

    Grids = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1], u_init, v_init, w_init, Co=100, Cm=1500.0,
        Cz=0, Cmod=0.0, vel_name='corrected_velocity',
        refl_field='reflectivity', frz=5000.0,
        filt_iterations=0, mask_outside_opt=True, upper_bc=1)

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0].fields['u']['data'])
    v_mean = np.nanmean(Grids[0].fields['v']['data'])
    w_max = np.max(Grids[0].fields['v']['data'])

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 10

    # Now we will test the nesting. Do the same retrieval, and make sure
    # that we get the same result within a prescribed tolerance
    cluster = LocalCluster(n_workers=2, processes=True)
    client = Client(cluster)
    Grids2 = pydda.retrieval.get_dd_wind_field_nested(
        [Grid0, Grid1], u_init, v_init, w_init, client, Co=100, Cm=1500.0,
        Cz=0, Cmod=0.0, vel_name='corrected_velocity',
        refl_field='reflectivity', frz=5000.0,
        filt_iterations=0, mask_outside_opt=True, upper_bc=1)

    # Make sure features are correlated between both versions. No reason
    # to expect the same answer, but features should be correlated
    # Nesting tends to make the updrafts a bit better resolved, so expect
    # less of an outright correlation (but still strong)
    assert np.corrcoef(Grids2[0].fields["u"]["data"].flatten(),
                       Grids[0].fields["u"]["data"].flatten())[0, 1] > 0.9
    assert np.corrcoef(Grids2[0].fields["v"]["data"].flatten(),
                       Grids[0].fields["v"]["data"].flatten())[0, 1] > 0.9
    assert np.corrcoef(Grids2[0].fields["w"]["data"].flatten(),
                       Grids[0].fields["w"]["data"].flatten())[0, 1] > 0.5
    cluster.close()
    client.close()


def test_smoothing():
    """ A field of random numbers from 0 to 1
        should smooth out to near 0.5 """
    Grid = pyart.testing.make_empty_grid(
            (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000)))

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field('zero_field', {'data': fdata3, '_FillValue': -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field('one_field', {'data': odata3, '_FillValue': -9999.0})

    u = np.random.random((20, 40, 40))
    v = np.random.random((20, 40, 40))
    w = np.zeros((20, 40, 40))
    new_grids = pydda.retrieval.get_dd_wind_field(
        [Grid], u, v, w, Co=0.0, Cx=1e-4, Cy=1e-4, Cm=0.0, Cmod=0.0,
        mask_outside_opt=False, filt_iterations=0, vel_name='one_field',
        refl_field='one_field')
    new_u = new_grids[0].fields['u']['data']
    new_v = new_grids[0].fields['v']['data']
    assert new_u.std() < u.std()
    assert new_v.std() < v.std()

    new_grids = pydda.retrieval.get_dd_wind_field([Grid], u, v, w, Co=0.0,
                                                  Cx=1e-2, Cy=1e-2, Cm=0.0,
                                                  Cmod=0.0,
                                                  mask_outside_opt=False,
                                                  filt_iterations=0,
                                                  vel_name='one_field',
                                                  refl_field='one_field')
    new_u2 = new_grids[0].fields['u']['data']
    new_v2 = new_grids[0].fields['v']['data']
    assert new_u2.std() < new_u.std()
    assert new_v2.std() < new_v.std()


def test_model_constraint():
    """ A retrieval with just the model constraint should converge
        to the model constraint. """
    Grid0 = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)

    """ Make fake model grid of just U = 1 m/s everywhere"""
    Grid0.fields["U_fakemodel"] = deepcopy(Grid0.fields["corrected_velocity"])
    Grid0.fields["V_fakemodel"] = deepcopy(Grid0.fields["corrected_velocity"])
    Grid0.fields["W_fakemodel"] = deepcopy(Grid0.fields["corrected_velocity"])

    Grid0.fields["U_fakemodel"]["data"] = np.ones(
        Grid0.fields["U_fakemodel"]["data"].shape)
    Grid0.fields["V_fakemodel"]["data"] = np.zeros(
        Grid0.fields["V_fakemodel"]["data"].shape)
    Grid0.fields["W_fakemodel"]["data"] = np.zeros(
        Grid0.fields["W_fakemodel"]["data"].shape)

    u_init = np.zeros(Grid0.fields["U_fakemodel"]["data"].shape)
    v_init = np.zeros(Grid0.fields["V_fakemodel"]["data"].shape)
    w_init = np.zeros(Grid0.fields["W_fakemodel"]["data"].shape)

    new_grids = pydda.retrieval.get_dd_wind_field(
        [Grid0], u_init, v_init, w_init, Co=0.0, Cx=0.0, Cy=0.0, Cm=0.0,
        Cmod=1.0, mask_outside_opt=False, filt_iterations=0,
        vel_name='corrected_velocity', refl_field='reflectivity',
        model_fields=['fakemodel'])

    np.testing.assert_allclose(new_grids[0].fields["u"]["data"],
                               Grid0.fields["U_fakemodel"]["data"],
                               atol=1e-2)
    np.testing.assert_allclose(new_grids[0].fields["v"]["data"],
                               Grid0.fields["V_fakemodel"]["data"],
                               atol=1e-2)
    np.testing.assert_allclose(new_grids[0].fields["w"]["data"],
                               Grid0.fields["W_fakemodel"]["data"],
                               atol=1e-2)
