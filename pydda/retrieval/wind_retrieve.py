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
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.signal import convolve2d, gaussian
from matplotlib import pyplot as plt
from copy import deepcopy

from .angles import add_azimuth_as_field, add_elevation_as_field

num_evaluations = 0


def J_function(winds, vrs, azs, els, wts, u_back, v_back,
                   C1, C2, C4, Cx, Cy, Cz, C8, 
                   dudt, dvdt, grid_shape,
                   vel_name, dx, dy, dz, z, rmsVr, weights, bg_weights):
        global num_evaluations
        winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))


        
        Jvel = (cost_functions.calculate_radial_vel_cost_function(vrs, azs,
                                                                  els,
                                                                  winds[0],
                                                                  winds[1],
                                                                  winds[2],
                                                                  wts,
                                                                  rmsVr=rmsVr,
                                                                  weights=weights,
                                                                  coeff=C1,
                                                                  dudt=dudt,
                                                                  dvdt=dvdt))
        if(C2 > 0):
            Jmass = (cost_functions.calculate_mass_continuity(winds[0], 
                                                              winds[1],
                                                              winds[2], z, 
                                                              dx, dy, dz,
                                                              coeff=C2))
        else:
            Jmass = 0
            
        if(Cx > 0 or Cy > 0 or Cz > 0):
            Jsmooth = cost_functions.calculate_smoothness_cost(
                winds[0], winds[1], winds[2], Cx=Cx, Cy=Cy, Cz=Cz)
        else:
            Jsmooth = 0
        if(C8 > 0):
            Jbackground = cost_functions.calculate_background_cost(winds[0], 
                                                                   winds[1], 
                                                                   winds[2],
                                                                   bg_weights,
                                                                   u_back,
                                                                   v_back,
                                                                   C8=C8)
        else:
            Jbackground = 0

        print('Jvel = ' + str(Jvel))
        print('Jmass = ' + str(Jmass))
        print('Jsmooth = ' + str(Jsmooth))
        print('Jbackground = ' + str(Jbackground))
        print('Maximum w: ' + str(np.abs(winds[2]).max()))

        return Jvel + Jmass + Jsmooth + Jbackground

    
def grad_J(winds, vrs, azs, els,
               wts, u_back, v_back,
               C1, C2, C4, Cx, Cy, Cz, C8,  
               dudt, dvdt, grid_shape,
               vel_name, dx, dy, dz, z, rmsVr, weights, bg_weights):

    winds = np.reshape(winds, (3, grid_shape[0], grid_shape[1],
                                      grid_shape[2]))
    grad = cost_functions.calculate_grad_radial_vel(vrs, els, azs, winds[0],
                                                     winds[1], winds[2],
                                                     wts, weights,
                                                     rmsVr, coeff=C1)
    if(C2 > 0):
         grad +=  cost_functions.calculate_mass_continuity_gradient(winds[0],
                                                                    winds[1],
                                                                    winds[2],
                                                                    z,
                                                                    dx, dy, dz,
                                                                    coeff=C2
                                                                    )
    if(Cx > 0 or Cy > 0 or Cz > 0):
        grad += cost_functions.calculate_smoothness_gradient(winds[0], 
                                                             winds[1], 
                                                             winds[2],
                                                             Cx=Cx, Cy=Cy,
                                                             Cz=Cz)
    if(C8 > 0):
        grad += cost_functions.calculate_background_gradient(winds[0], winds[1], winds[2],
                                                          bg_weights, u_back, v_back, C8=C8)
    print('Norm of gradient: ' + str(np.linalg.norm(grad, 2)))
    return grad


def get_dd_wind_field(Grids, u_init, v_init, w_init, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, C1=1.0, C2=1500.0, C4=50.0, Cx=0.0,
                      Cy=0.0, Cz=5.0, C8=0.001, dudt=0.0, dvdt=0.0,
                      filt_iterations=50, mask_outside_opt=False,
                      max_iterations=200, mask_w_outside_opt=True):
    
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
        wts.append(cost_functions.calculate_fall_speed(Grids[i], refl_field=refl_field))
        add_azimuth_as_field(Grids[i])
        add_elevation_as_field(Grids[i])
        vrs.append(Grids[i].fields[vel_name]['data'])
        azs.append(Grids[i].fields['AZ']['data']*np.pi/180)
        els.append(Grids[i].fields['EL']['data']*np.pi/180)
        sum_Vr[i] = np.sum(np.square(vrs[i]))
        M[i] = len(np.where(vrs[i].mask == False)[0])
        for j in range(i+1, len(Grids)):
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
                     np.logical_and(bca[i,j] > np.pi / 6, bca[i,j] < 5 * np.pi / 6))] += 1
                weights[i,k] = cur_array
                cur_array = bg_weights[k]
                cur_array[np.logical_or(bca[i,j] < np.pi / 6, bca[i,j] > 5 * np.pi / 6)] = 1
                cur_array[vrs[i][k].mask == True] = 1
                bg_weights[i] = cur_array
    print(M)
    rmsVr = np.sum(sum_Vr)/np.sum(M)
    weights[weights > 1] = 1
    del bca
    grid_shape = u_init.shape
    # Parse names of velocity field

    winds = winds.flatten()
    #ones = np.ones(winds.shape)

    ndims = len(winds)
    print(len(winds.shape))
    print(("Starting solver for " + str(ndims) + " dimensional problem..."))
    dx = np.diff(Grids[0].x['data'], axis=0)[0]
    dy = np.diff(Grids[0].y['data'], axis=0)[0]
    dz = np.diff(Grids[0].z['data'], axis=0)[0]
    print('rmsVR = ' + str(rmsVr))
    print('Total points:' +str(M.sum()))
    z = Grids[0].point_z['data']

    the_time = time.time()
    bt = time.time()

    filter_winds = lambda winds: filter_wind(winds, grid_shape)
    
    # First pass - no filter
    wcurr = w_init
    wprev = 100*np.ones(w_init.shape)
    maxwprev = 99
    iterations = 0
    warnflag = 99999
    while(iterations < max_iterations and warnflag > 0 and (abs(wprev-wcurr).max() > 0.2)):
        wprev = wcurr
        bounds = [(-x,x) for x in 100*np.ones(winds.shape)]
        winds = fmin_l_bfgs_b(J_function, winds, args=(vrs, azs,
                                                 els, wts, u_back2,
                                                 v_back2, C1, C2, C4, Cx, 
                                                 Cy, Cz, C8, dudt,
                                                 dvdt, grid_shape,
                                                 vel_name, dx, dy, dz, z, 
                                                 rmsVr, weights,
                                                 bg_weights),
                                maxiter=10, pgtol=5e-4, bounds=bounds, 
                                fprime=grad_J, disp=True)

        warnflag = winds[2]['warnflag']
        if(warnflag > 1):
            raise ArithmeticError('Cost function does not converge!')
        winds = np.reshape(winds[0], (3, grid_shape[0], grid_shape[1],
                                           grid_shape[2]))
        iterations = iterations+10
        wcurr = winds[2]
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()
    print("Done! Time = " + str(time.time() - bt))

    print('Filtering wind field...')
    bt = time.time()

    #the_winds = np.stack([u, v, w])
    print("Done! Time = " + str(time.time() - bt))

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
    v_field = deepcopy(Grids[0].fields[vel_name])
    v_field['data'] = v
    v_field['standard_name'] = 'v_wind'
    v_field['long_name'] = 'zonal component of wind velocity'  
    w_field = deepcopy(Grids[0].fields[vel_name])
    w_field['data'] = w
    w_field['standard_name'] = 'w_wind'
    w_field['long_name'] = 'vertical component of wind velocity' 
    return u_field, v_field, w_field


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
    print(u_back2)
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


def mean_filter(array, N):
    # Calculate convolution
    kernel = np.outer(gaussian(4,1), gaussian(4,1))
    xs = convolve2d(array, kernel, mode="same", boundary="symm")
    ns = N ** 2
    return xs/ns



