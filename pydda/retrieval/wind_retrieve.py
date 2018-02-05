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

from .. import cost_functions
from scipy.optimize import fmin_cg
from scipy.interpolate import interp1d
from scipy.signal import convolve2d, gaussian
from matplotlib import pyplot as plt

from .angles import add_azimuth_as_field, add_elevation_as_field

num_evaluations = 0


def J_function(winds, vr_1, vr_2, az1, az2, el1, el2,
                   wt1, wt2, u_back, v_back,
                   C1, C2, C4, C5, C6, C7, C8,
                   dudt, dvdt, grid_shape,
                   vel_name, dx, dy, dz, z, rmsVr, weights, bg_weights):
        global num_evaluations
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))

        Jvel = (cost_functions.calculate_radial_vel_cost_function(vr_1, vr_2,
                                                                 az1,
                                                                 az2, el1,
                                                                 el2,
                                                                 winds[0],
                                                                 winds[1],
                                                                 winds[2],
                                                                 wt1, wt2,
                                                                 rmsVr=rmsVr,
                                                                 weights=weights,
                                                                 coeff=C1,
                                                                 dudt=dudt,
                                                                 dvdt=dvdt,
                                                                 vel_name=vel_name))
        Jmass = (cost_functions.calculate_mass_continuity(winds[0], winds[1],
                                                        winds[2], z, el1, 
                                                        dx, dy, dz, coeff=C2))
        if(C4 > 0 or C5 > 0 or C6 > 0 or C7 > 0):
            Jsmooth = (cost_functions.calculate_smoothness_cost(winds[0], winds[1], winds[2], z, el1,
                                                        dx, dy, dz, cutoff=1000.0,
                                                        C4=C4, C5=C5, C6=C6, C7=C7))
        else:
            Jsmooth = 0
        if(C8 > 0):
            Jbackground = cost_functions.calculate_background_cost(winds[0], winds[1], winds[2],
                                                        bg_weights, u_back, v_back, C8=C8)
        else:
            Jbackground = 0

        print('Jvel = ' + str(Jvel))
        print('Jmass = ' + str(Jmass))
        print('Jsmooth = ' + str(Jsmooth))
        print('Jbackground = ' + str(Jbackground))
        print('Maximum w: ' + str(np.abs(winds[2]).max()))

        return Jvel + Jmass + Jsmooth + Jbackground

    
def grad_J(winds, vr_1, vr_2, az1, az2, el1, el2,
               wt1, wt2, u_back, v_back,
               C1, C2, C4, C5, C6, C7, C8,
               dudt, dvdt, grid_shape,
               vel_name, dx, dy, dz, z, rmsVr, weights, bg_weights):

    winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))

    grad = cost_functions.calculate_grad_radial_vel(vr_1, vr_2,el1, az1,
                                                     el2, az2, winds[0], winds[1], winds[2],
                                                     wt1, wt2, weights, rmsVr, coeff=C1)
    grad +=  cost_functions.calculate_mass_continuity_gradient(winds[0],
                                                              winds[1],
                                                              winds[2],
                                                              z, el1,
                                                              dx, dy, dz,
                                                              coeff=C2
                                                              )
    if(C4 > 0 or C5 > 0 or C6 > 0 or C7 > 0):
        grad += cost_functions.calculate_smoothness_gradient(winds[0], winds[1], winds[2], z, el1,
                                                         dx, dy, dz, cutoff=1000.0,
                                                         C4=C4, C5=C5, C6=C6, C7=C7)
    if(C8 > 0):
        grad += cost_functions.calculate_background_gradient(winds[0], winds[1], winds[2],
                                                          bg_weights, u_back, v_back, C8=C8)
    print(grad[200])
    return grad


def get_dd_wind_field(Grid1, Grid2, u_init, v_init, w_init, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, C1=1.0, C2=1500.0, C4=50.0, C5=0.0,
                      C6=0.0, C7=5.0, C8=0.0, dudt=0.0, dvdt=0.0,
                      filt_iterations=50):

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
        u_back2 = u_interp(Grid1.z['data'])
        v_back2 = v_interp(Grid1.z['data'])
        print('Interpolated U field:')
        print(u_back2)
        print('Interpolated V field:')
        print(v_back2)
        print('Grid levels:')
        print(Grid1.z['data'])
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

    ndims = len(winds)
    print(len(winds.shape))
    print(("Starting solver for " + str(ndims) + " dimensional problem..."))
    vr_1 = Grid1.fields[vel_name]['data']
    vr_2 = Grid2.fields[vel_name]['data']
    z_1 = Grid1.fields[refl_field]['data']
    z_2 = Grid1.fields[refl_field]['data']
    az1 = Grid1.fields['AZ']['data']*np.pi/180
    az2 = Grid2.fields['AZ']['data']*np.pi/180
    el1 = Grid1.fields['EL']['data']*np.pi/180
    el2 = Grid2.fields['EL']['data']*np.pi/180

    dx = np.diff(Grid1.x['data'], axis=0)[0]
    dy = np.diff(Grid1.y['data'], axis=0)[0]
    dz = np.diff(Grid1.z['data'], axis=0)[0]
    bca = get_bca(Grid1.radar_longitude['data'],
                  Grid1.radar_latitude['data'],
                  Grid2.radar_longitude['data'],
                  Grid2.radar_latitude['data'],
                  Grid1.point_x['data'][0],
                  Grid1.point_y['data'][0],
                  Grid1.get_projparams())

    M1 = len(np.where(vr_1.mask == False)[1])
    M2 = len(np.where(vr_2.mask == False)[1])
    weights = np.ones(vr_1.shape)
    bg_weights = np.ones(vr_1.shape)
    weights = np.ma.where(np.logical_or(vr_1.mask == True, vr_2.mask == True), 0, 1)
    bg_weights = np.zeros(vr_1.shape)

    for i in range(vr_1.shape[0]):
        cur_array = weights[i]
        cur_array[np.logical_or(bca < np.pi/6, bca > 5*np.pi/6)] = 0
        weights[i] = cur_array
        cur_array = bg_weights[i]
        cur_array[np.logical_or(bca < np.pi / 6, bca > 5 * np.pi / 6)] = 1
        cur_array[vr_1[i].mask == True] = 1
        cur_array[vr_2[i].mask == True] = 1
        bg_weights[i] = cur_array

    print(len(np.where(weights == 0)))
    print(M1)
    print(M2)
    rmsVr = ((np.sum(np.square(vr_1)) + np.sum(np.square(vr_2))) / (M1 + M2))
    print('rmsVR = ' + str(rmsVr))
    print('Total points:' +str(M1+M2))
    z = Grid1.point_z['data']

    print(dx)
    print(dy)
    print(dz)
    the_time = time.time()
    bt = time.time()

    filter_winds = lambda winds: filter_wind(winds, grid_shape)
    # First pass - no filter
    wcurr = w_init
    wprev = 100*np.ones(w_init.shape)
    maxwprev = 99
    iterations = 0
    while(np.max(np.abs(wprev-wcurr)) > 0.2 and iterations < 100):
        wprev = wcurr
        winds = fmin_cg(J_function, winds, args=(vr_1, vr_2, az1, az2,
                                                     el1, el2,
                                                     wt1, wt2, u_back2, v_back2,
                                                     C1, C2, C4, C5, C6, C7, C8, dudt,
                                                     dvdt, grid_shape,
                                                     vel_name, dx, dy, dz, z, rmsVr, weights,
                                                 bg_weights),
                                maxiter=10, gtol=1e-3, fprime=grad_J, disp=True,
                              )
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                           grid_shape[2]))
        iterations = iterations+10
        if(np.abs(wprev-winds[2]).max() > 0.2 and
                   iterations % filt_iterations == 0 and
                   filt_iterations != 1):
            print('Applying filter...')
            for i in range(vr_1.shape[0]):
                winds[0,i] = mean_filter(winds[0,i], 4)
                winds[1,i] = mean_filter(winds[1,i], 4)
                winds[2,i] = mean_filter(winds[2,i], 4)

        wcurr = winds[2]
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()
    print("Done! Time = " + str(time.time() - bt))


    print('Filtering wind field...')
    bt = time.time()

    #the_winds = np.stack([u, v, w])
    print("Done! Time = " + str(time.time() - bt))
    #print('Doing second minimization post-filter...')
    # First pass - no filter

    print("Done! Time = " + str(time.time() - bt))
    the_winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                       grid_shape[2]))
    u = the_winds[0]
    v = the_winds[1]
    w = the_winds[2]
    #u = np.ma.masked_where(np.logical_or(vr_1.mask == True, vr_2.mask == True), u)
    #v = np.ma.masked_where(np.logical_or(vr_1.mask == True, vr_2.mask == True), u)
    #w = np.ma.masked_where(np.logical_or(vr_1.mask == True, vr_2.mask == True), u)
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
    u_interp = interp1d(z_back, u_back, bounds_error=False)
    v_interp = interp1d(z_back, v_back, bounds_error=False)
    u_back2 = u_interp(Grid.z['data'])
    v_back2 = v_interp(Grid.z['data'])
    print(u_back2)
    for i in range(analysis_grid_shape[0]):
        u[i] = u_back2[i]
        v[i] = v_back2[i]
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    return u, v, w


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


def mean_filter(array, N):
    # Calculate convolution
    kernel = np.outer(gaussian(4,1), gaussian(4,1))
    xs = convolve2d(array, kernel, mode="same", boundary="symm")
    ns = N ** 2
    return xs/ns



