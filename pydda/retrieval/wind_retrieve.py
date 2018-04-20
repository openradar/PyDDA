#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:17:40 2017

@author: rjackson
"""

import pyart
import numpy as np
import time
import cartopy.crs as ccrs
import math

from .. import cost_functions
from ..cost_functions import J_function, grad_J
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from copy import deepcopy

from .angles import add_azimuth_as_field, add_elevation_as_field

num_evaluations = 0


def get_dd_wind_field(Grids, u_init, v_init, w_init, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, Co=1.0, Cm=1500.0, Cx=0.0,
                      Cy=0.0, Cz=0.0, Cb=0.0, filt_iterations=2, 
                      mask_outside_opt=False, max_iterations=200, 
                      mask_w_outside_opt=True, filter_window=9, 
                      filter_order=4, min_bca=30.0, max_bca=150.0,
                      upper_bc=True):
    
    # Returns a field dictionary
    num_evaluations = 0

    # Disable background constraint if none provided
    if(u_back == None or v_back == None):
        u_back2 = np.zeros(u_init.shape[0])
        v_back2 = np.zeros(v_init.shape[0])
        C8 = 0.0
    else:
        # Interpolate sounding to radar grid
        print('Interpolating sounding to radar grid')
        u_interp = interp1d(z_back, u_back, bounds_error=False)
        v_interp = interp1d(z_back, v_back, bounds_error=False)
        u_back2 = u_interp(Grids[0].z['data'])
        v_back2 = v_interp(Grids[0].z['data'])
        print('Interpolated U field:')
        print(u_back2)
        print('Interpolated V field:')
        print(v_back2)
        print('Grid levels:')
        print(Grids[0].z['data'])

    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name('reflectivity')

    # Parse names of velocity field
    if vel_name is None:
        vel_name = pyart.config.get_field_name('corrected_velocity')    
    winds = np.stack([u_init, v_init, w_init])
    wts = []
    vrs = []
    azs = []
    els = []
    
    # Set up wind fields and weights from each radar
    weights = np.zeros(
        (len(Grids), u_init.shape[0], u_init.shape[1], u_init.shape[2]))
    bg_weights = np.zeros(v_init.shape)
    bca = np.zeros(
        (len(Grids), len(Grids), u_init.shape[1], u_init.shape[2]))
    M = np.zeros(len(Grids))
    sum_Vr = np.zeros(len(Grids))

    for i in range(len(Grids)):
        wts.append(cost_functions.calculate_fall_speed(Grids[i], 
                                                       refl_field=refl_field))
        add_azimuth_as_field(Grids[i])
        add_elevation_as_field(Grids[i])
        vrs.append(Grids[i].fields[vel_name]['data'])
        azs.append(Grids[i].fields['AZ']['data']*np.pi/180)
        els.append(Grids[i].fields['EL']['data']*np.pi/180)
        
    for i in range(len(Grids)):    
        for j in range(i+1, len(Grids)):
            print(("Calculating weights for radars " + str(i) +
                   " and " + str(j)))
            bca[i,j] = get_bca(Grids[i].radar_longitude['data'],
                               Grids[i].radar_latitude['data'],
                               Grids[j].radar_longitude['data'],
                               Grids[j].radar_latitude['data'],
                               Grids[i].point_x['data'][0],
                               Grids[i].point_y['data'][0],
                               Grids[i].get_projparams())

            for k in range(vrs[i].shape[0]):
                cur_array = weights[i,k]
                cur_array[np.logical_and(
                    vrs[i][k].mask == False,
                    np.logical_and(
                        bca[i,j] >= math.radians(min_bca), 
                        bca[i,j] <= math.radians(max_bca)))] += 1
                weights[i,k] = cur_array
                cur_array = weights[j,k]
                cur_array[np.logical_and(
                    vrs[j][k].mask == False,
                    np.logical_and(
                        bca[i,j] >= math.radians(min_bca), 
                        bca[i,j] <= math.radians(max_bca)))] += 1
                weights[j,k] = cur_array
                cur_array = bg_weights[k]
                cur_array[np.logical_or(
                    bca[i,j] >= math.radians(min_bca),
                    bca[i,j] <= math.radians(max_bca))] = 1
                cur_array[vrs[i][k].mask == True] = 0
                bg_weights[i] = cur_array
    
    weights[weights > 0] = 1            
    sum_Vr = np.sum(np.square(vrs*weights))

    rmsVr = np.sum(sum_Vr)/np.sum(weights)
    
    del bca
    grid_shape = u_init.shape
    # Parse names of velocity field

    winds = winds.flatten()
    #ones = np.ones(winds.shape)

    ndims = len(winds)

    print(("Starting solver "))
    dx = np.diff(Grids[0].x['data'], axis=0)[0]
    dy = np.diff(Grids[0].y['data'], axis=0)[0]
    dz = np.diff(Grids[0].z['data'], axis=0)[0]
    print('rmsVR = ' + str(rmsVr))
    print('Total points:' +str(weights.sum()))
    z = Grids[0].point_z['data']

    the_time = time.time()
    bt = time.time()
    
    # First pass - no filter
    wcurr = w_init
    wprev = 100*np.ones(w_init.shape)
    wprevmax = 99
    wcurrmax = w_init.max()
    iterations = 0
    warnflag = 99999
    coeff_max = np.max([Co, Cb, Cm, Cx, Cy, Cz, Cb])
    bounds = [(-x,x) for x in 100*np.ones(winds.shape)]
    while(iterations < max_iterations and 
          (abs(wprevmax-wcurrmax) > 0.02)):
        wprevmax = wcurrmax
        winds = fmin_l_bfgs_b(J_function, winds, args=(vrs, azs, els, 
                                                       wts, u_back, v_back,
                                                       Co, Cm, Cx, Cy, Cz, Cb, 
                                                       grid_shape,  
                                                       dx, dy, dz, z, rmsVr, 
                                                       weights, bg_weights,
                                                       upper_bc),
                                maxiter=10, pgtol=1e-3, bounds=bounds, 
                                fprime=grad_J, disp=1)
        

        # Print out cost function values after 10 iterations
        J = J_function(winds[0], vrs, azs, els, wts, u_back, v_back,
                       Co, Cm, Cx, Cy, Cz, Cb, grid_shape,  
                       dx, dy, dz, z, rmsVr, weights, bg_weights, upper_bc=True,
                       print_out=True)
        J = grad_J(winds[0], vrs, azs, els, wts, u_back, v_back,
                   Co, Cm, Cx, Cy, Cz, Cb, grid_shape,  
                   dx, dy, dz, z, rmsVr, weights, bg_weights, upper_bc=True,
                   print_out=True)
        
        warnflag = winds[2]['warnflag']
        
        winds = np.reshape(winds[0], (3, grid_shape[0], grid_shape[1],
                                           grid_shape[2]))
        iterations = iterations+10
        print('Iterations before filter: ' + str(iterations))
        
        wcurrmax = winds[2].max()
        
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()

        
    if(filt_iterations > 0):
        print('Applying low pass filter to wind field...')
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                           grid_shape[2]))
        winds[0] = savgol_filter(winds[0], 9, 3, axis=0)
        winds[0] = savgol_filter(winds[0], 9, 3, axis=1)
        winds[0] = savgol_filter(winds[0], 9, 3, axis=2)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=0)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=1)
        winds[1] = savgol_filter(winds[1], 9, 3, axis=2)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=0)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=1)
        winds[2] = savgol_filter(winds[2], 9, 3, axis=2)  
        
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()
        iterations = 0
        while(iterations < filt_iterations):
            winds = fmin_l_bfgs_b(J_function, winds, args=(vrs, azs, els, 
                                  wts, u_back, v_back,
                                  Co, Cm, Cx, Cy, Cz, Cb, 
                                  grid_shape, dx, dy, dz, z, rmsVr, 
                                  weights, bg_weights, upper_bc),
                                  maxiter=1, pgtol=1e-3, bounds=bounds, 
                                  fprime=grad_J, disp=1)

            warnflag = winds[2]['warnflag']
        
            winds = np.reshape(winds[0], (3, grid_shape[0], grid_shape[1],
                                             grid_shape[2]))
            iterations = iterations+1
            print('Iterations after filter: ' + str(iterations))
                
            winds = np.stack([winds[0], winds[1], winds[2]])
            winds = winds.flatten()
            
    print("Done! Time = " + "{:2.1f}".format(time.time() - bt))

    # First pass - no filter

    the_winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                       grid_shape[2]))
    u = the_winds[0]
    v = the_winds[1]
    w = the_winds[2]

    where_mask = np.sum(weights, axis=0)
    if(mask_outside_opt==True):
        u = np.ma.masked_where(where_mask < 1, u)
        v = np.ma.masked_where(where_mask < 1, v)
        w = np.ma.masked_where(where_mask < 1, w)
    if(mask_w_outside_opt==True):
        w = np.ma.masked_where(where_mask < 1, w)

    u_field = deepcopy(Grids[0].fields[vel_name])
    u_field['data'] = u
    u_field['standard_name'] = 'u_wind'
    u_field['long_name'] = 'meridional component of wind velocity'
    u_field['min_bca'] = min_bca
    u_field['max_bca'] = max_bca
    v_field = deepcopy(Grids[0].fields[vel_name])
    v_field['data'] = v
    v_field['standard_name'] = 'v_wind'
    v_field['long_name'] = 'zonal component of wind velocity' 
    v_field['min_bca'] = min_bca
    v_field['max_bca'] = max_bca
    w_field = deepcopy(Grids[0].fields[vel_name])
    w_field['data'] = w
    w_field['standard_name'] = 'w_wind'
    w_field['long_name'] = 'vertical component of wind velocity' 
    w_field['min_bca'] = min_bca
    w_field['max_bca'] = max_bca

    
    new_grid_list = []
    
    for grid in Grids:
        temp_grid = deepcopy(grid)
        temp_grid.add_field('u', u_field, replace_existing=True)
        temp_grid.add_field('v', v_field, replace_existing=True)
        temp_grid.add_field('w', w_field, replace_existing=True)
        
        new_grid_list.append(temp_grid)
        
    return new_grid_list


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
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    return u, v, w


def make_wind_field_from_profile(Grid, profile, vel_field=None):
    """

        :param Grid: Py-ART Grid object
             This is the Py-ART Grid containing the coordinates for the analysis 
             grid.
        :param profile: Py-ART HorizontalWindProfile
             This is the horizontal wind profile from the sounding
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
    u = np.ones(analysis_grid_shape)
    v = np.ones(analysis_grid_shape)
    w = np.zeros(analysis_grid_shape)
    u_back = profile[1].u_wind
    v_back = profile[1].v_wind
    z_back = profile[1].height
    u_interp = interp1d(
        z_back, u_back, bounds_error=False, fill_value='extrapolate')
    v_interp = interp1d(
        z_back, v_back, bounds_error=False, fill_value='extrapolate')
    u_back2 = u_interp(Grid.z['data'])
    v_back2 = v_interp(Grid.z['data'])
    for i in range(analysis_grid_shape[0]):
        u[i] = u_back2[i]
        v[i] = v_back2[i]
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    return u, v, w


""" Makes a test wind field that converges at center near ground and
    Diverges aloft at center """
def make_test_divergence_field(Grid, wind_vel, z_ground, z_top, radius,
                               back_u, back_v, x_center, y_center,):
    """
    Parameters
    ----------
    Grid: Py-ART Grid object
             This is the Py-ART Grid containing the coordinates for the analysis 
             grid.
    wind_vel: float
        The maximum wind velocity.
    z_ground: float 
        The bottom height where the maximum convergence occurs
    z_top: float
        The height where the maximum divergence occurrs
        
    Returns
    -------
    u_init, v_init, w_init: ndarrays of floats
         Initial U, V, W field
    """
    
    x = Grid.point_x['data']
    y = Grid.point_y['data']
    z = Grid.point_z['data']
    
    
    theta = np.arctan2(x-x_center, y-y_center)
    phi = np.pi*((z-z_ground)/(z_top-z_ground))
    r = np.sqrt(np.square(x-x_center) + np.square(y-y_center))
    
    u = wind_vel*(r/radius)**2*np.cos(phi)*np.sin(theta)*np.ones(x.shape)
    v = wind_vel*(r/radius)**2*np.cos(phi)*np.cos(theta)*np.ones(x.shape)
    w = np.zeros(x.shape)
    u[r > radius] = back_u
    v[r > radius] = back_v
    return u,v,w


# Gets beam crossing angle over 2D grid centered over Radar 1.
# grid_x, grid_y are cartesian coordinates from pyproj.Proj (or basemap)
def get_bca(rad1_lon, rad1_lat,
            rad2_lon, rad2_lat,
            x, y, projparams):

    rad1 = pyart.core.geographic_to_cartesian(rad1_lon, rad1_lat, projparams)
    rad2 = pyart.core.geographic_to_cartesian(rad2_lon, rad2_lat, projparams)
    # Create grid with Radar 1 in center

    x = x-rad1[0]
    y = y-rad1[1]
    rad2 = np.array(rad2) - np.array(rad1)
    a = np.sqrt(np.multiply(x,x)+np.multiply(y,y))
    b = np.sqrt(pow(x-rad2[0],2)+pow(y-rad2[1],2))
    c = np.sqrt(rad2[0]*rad2[0]+rad2[1]*rad2[1])
    theta_1 = np.arccos(x/a)
    theta_2 = np.arccos((x-rad2[1])/b)
    return np.arccos((a*a+b*b-c*c)/(2*a*b))



