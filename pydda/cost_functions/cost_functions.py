"""
Created on Wed Jul 19 11:31:02 2017

@author: rjackson
"""

import numpy as np
import pyart
from numba import jit

#@jit(parallel=True)
def calculate_radial_vel_cost_function(vr_1, vr_2, az1, az2, el1, el2, u, v,
                                       w, wt1,
                                       wt2, coeff=10.0, dudt=0.0, dvdt=0.0, 
                                       vel_name=None):
    """
    Calculates the cost function due to difference of wind field from
    radar radial velocities. Radar 1 and Radar 2 must be in 
    
    Parameters
    ----------
    rad_vel1: float
        Grid object from Radar 1
    rad_vel2: float
        Grid object from Radar 2. Radar 1 and Radar 2 must be gridded to same
        locations.
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    coeff: float
        Constant for cost function
    dudt: float
        Background storm motion
    dvdt: float
        Background storm motion
    vel_name: str
        Background velocity field name
    """
    ## Need to implement time of observation 
    v_ar1 = (np.cos(el1)*np.sin(az1)*u + 
             np.cos(el1)*np.cos(az1)*v + 
             np.sin(el1)*(w - wt1))
    v_ar2 = (np.cos(el2)*np.sin(az2)*u + 
             np.cos(el2)*np.cos(az2)*v + 
             np.sin(el2)*(w - wt2))
    J_o = coeff*(np.sum(coeff*np.square(vr_1 - v_ar1) +
                 np.sum(coeff*np.square(vr_2 - v_ar2))))
    
    return J_o

#@jit(parallel=True)
def calculate_grad_radial_vel(el, az, u, v, w,
                              wt, coeff=10.0):
    v_ar1 = (np.cos(el)*np.sin(az)*u + 
            np.cos(el)*np.cos(az)*v + 
            np.sin(el)*(w - wt))
    p_x1 = coeff*(2*(v_ar1 - u)*np.cos(el)*np.sin(az)) 
    p_y1 = coeff*(2*(v_ar1 - v)*np.cos(el)*np.cos(az)) 
    p_z1 = coeff*(2*(v_ar1 - w)*np.sin(el))
    y = np.stack([p_x1, p_y1, p_z1], axis=0)
    return np.reshape(y, np.prod((y.shape,)))
           
def calculate_fall_speed(grid, refl_field=None, frz=4500.0):
    """
    Estimates fall speed based on reflectivity
    Uses methodology of Mike Biggerstaff and Dan Betten
    
    Parameters
    ----------
    Grid: Py-ART Grid
    
    Returns
    -------
    Field dict:
        Float array of terminal velocities
    
    """
    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name('reflectivity')
        
    refl = grid.fields[refl_field]['data']
    grid_z = grid.point_z['data']
    term_vel = np.zeros(refl.shape)    
    A = np.zeros(refl.shape)
    B = np.zeros(refl.shape)
    rho = grid_z/10000.0
    A[np.logical_and(grid_z < frz, refl < 55)] = -2.6
    B[np.logical_and(grid_z < frz, refl < 55)] = 0.0107
    A[np.logical_and(grid_z < frz, 
                     np.logical_and(refl >= 55, refl < 60))] = -2.5
    B[np.logical_and(grid_z < frz, 
                     np.logical_and(refl >= 55, refl < 60))] = 0.013
    A[np.logical_and(grid_z < frz, refl > 60)] = -3.95
    B[np.logical_and(grid_z < frz, refl > 60)] = 0.0148
    A[np.logical_and(grid_z > frz, refl < 33)] = -0.817
    B[np.logical_and(grid_z > frz, refl < 33)] = 0.0063
    A[np.logical_and(grid_z > frz,
                     np.logical_and(refl >= 55, refl < 49))] = -2.5
    B[np.logical_and(grid_z > frz,
                     np.logical_and(refl >= 55, refl < 49))] = 0.013
    A[np.logical_and(grid_z > frz, refl > 49)] = -3.95
    B[np.logical_and(grid_z > frz, refl > 49)] = 0.0148

    fallspeed = A*np.power(10, refl*B)*np.power(1.2/rho, 0.4)
    return fallspeed
    del A,B,rho