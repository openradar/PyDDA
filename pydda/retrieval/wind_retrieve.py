#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:17:40 2017

@author: rjackson
"""

import pyart
import numpy as np
import time
from .. import cost_functions
from mystic.solvers import diffev
from mystic.monitors import VerboseLoggingMonitor

from .angles import add_azimuth_as_field, add_elevation_as_field

def get_dd_wind_field(Grid1, Grid2, u_init, v_init, w_init, vel_name=None,
                      refl_field=None, frz=4500.0, C1=10.0, dudt=0.0,
                      dvdt=0.0):
    
    def J_function(winds, vr_1, vr_2, az1, az2, el1, el2,
                   wt1, wt2, C1, dudt, dvdt, grid_shape,
                   vel_name):
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))

        return cost_functions.calculate_radial_vel_cost_function(vr_1, vr_2,
                                                                 az1,
                                                                 az2, el1,
                                                                 el2,
                                                                 winds[0],
                                                                 winds[1],
                                                                 winds[2],
                                                                 wt1, wt2,
                                                                 coeff=C1,
                                                                 dudt=dudt,
                                                                 dvdt=dvdt,
                                                                 vel_name=vel_name)
    
        
    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name('reflectivity')

    # Parse names of velocity field
    if vel_name is None:
        vel_name = pyart.config.get_field_name('corrected_velocity')    
    winds = np.stack([u_init, v_init, w_init])
    wt1 = cost_functions.calculate_fall_speed(Grid1, refl_field=refl_field)
    wt2 = cost_functions.calculate_fall_speed(Grid2, refl_field=refl_field)
    add_azimuth_as_field(Grid1)
    add_azimuth_as_field(Grid2)
    add_elevation_as_field(Grid1)
    add_elevation_as_field(Grid2)
  
    grid_shape = u_init.shape
    # Parse names of velocity field

    winds = winds.flatten()
    #ones = np.ones(winds.shape)
    stepmon = VerboseLoggingMonitor(1,1)
    ndims = len(winds)
    print(len(winds.shape))
    print(("Starting solver for " + str(ndims) + " dimensional problem..."))
    bt = time.time()
    vr_1 = Grid1.fields[vel_name]['data']
    vr_2 = Grid2.fields[vel_name]['data']
    az1 = Grid1.fields['AZ']['data']
    az2 = Grid2.fields['AZ']['data']
    el1 = Grid1.fields['EL']['data']
    el2 = Grid2.fields['EL']['data']
    the_time = time.time()
    initial_cost = J_function(winds,vr_1, vr_2, az1, az2, el1, el2,
                              wt1, wt2, C1, dudt, dvdt, grid_shape, vel_name)
    print(('One execution of cost function takes ' + str(time.time()-the_time) + ' s'))
    the_winds = diffev(J_function, winds, args=(vr_1, vr_2, az1, az2,
                                                el1, el2,
                                                wt1, wt2, C1, dudt,
                                                dvdt, grid_shape,
                                                vel_name),
                       maxiter=1, ftol=1e-6, itermon=stepmon, 
                       )
    print("Done! Time = " + str(time.time() - bt))
    the_winds = np.reshape(the_winds, (3, grid_shape[0], grid_shape[1],
                                          grid_shape[2]))
    u = the_winds[0]
    v = the_winds[1]
    w = the_winds[2]
    
    return u, v, w

""" Makes a initialization wind field that is a constant everywhere"""
def make_constant_wind_field(Grid, wind=(0.0,0.0,0.0), vel_field=None):
    """
    
    :param Grid: Py-ART Grid object
         This is the Py-ART Grid containing the coordinates for the analysis 
         grid.
    :param wind: 3-tuple of floats
         The 3-tuple specifying the (u,v,w) of the wind field.
    :param vel_field: String
         The name of the velocity field
    :return: 3 float arrays 
         The u, v, and w inital conditions.
    """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name('corrected_velocity')
    analysis_grid_shape = Grid.fields[vel_field]['data'].shape
    u = wind[0]*np.ones(analysis_grid_shape)
    v = wind[1]*np.ones(analysis_grid_shape)
    w = wind[2]*np.ones(analysis_grid_shape)
    return u, v, w