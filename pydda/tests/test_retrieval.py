#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:43 2018

@author: rjackson
"""

import pydda
import pyart
import numpy as np


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
    u, v, w = pydda.initialization.make_test_divergence_field(
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
        Grid0, sounding[1], vel_field='VT')

    Grids = pydda.retrieval.get_dd_wind_field([Grid0, Grid1], u_init, v_init,
                                              w_init, Co=100, Cm=1500.0,
                                              Cz=0, Cmod=0.0, vel_name='VT',
                                              refl_field='DT', frz=5000.0,
                                              filt_iterations=0,
                                              mask_outside_opt=True,
                                              upper_bc=1)

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0].fields['u']['data'])
    v_mean = np.nanmean(Grids[0].fields['v']['data'])
    w_max = np.max(Grids[0].fields['v']['data'])

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 10


def test_smoothing():
    """ A field of random numbers from 0 to 1 should smooth out to near 0.5 """
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
    new_grids = pydda.retrieval.get_dd_wind_field([Grid], u, v, w, Co=0.0,
                                                  Cx=1e-4, Cy=1e-4, Cm=0.0,
                                                  Cmod=0.0,
                                                  mask_outside_opt=False,
                                                  filt_iterations=0,
                                                  vel_name='one_field',
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
