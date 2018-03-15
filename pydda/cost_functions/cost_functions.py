"""
Created on Wed Jul 19 11:31:02 2017

@author: rjackson
"""

import numpy as np
import pyart

from numba import jit, cuda
from numba import vectorize
import scipy.ndimage.filters 

def calculate_radial_vel_cost_function(vrs, azs, els, u, v,
                                       w, wts, rmsVr, weights, coeff=1.0,
                                       dudt=0.0, dvdt=0.0):
    """
    Calculates the cost function due to difference of wind field from
    radar radial velocities. Radar 1 and Radar 2 must be in 
    
    Parameters
    ----------
    vrs: List of float arrays
        List of radial velocities from each radar
    els: List of float arrays
        List of elevations from each radar
    azs: List of azimuths
        List of azimuths from each radar
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
    weights: n_radars x_bins x y_bins float array
        Data weights for each pair of radars
    """
    ## Need to implement time of observation
    for i in range(len(vrs)):
        vrs[i] = np.ma.filled(vrs[i], 0)
        wts[i] = np.ma.filled(wts[i], 0)
        vrs[i][np.isnan(vrs[i])] = 0
        wts[i][np.isnan(wts[i])] = 0

    J_o = 0
    lambda_o = coeff / (rmsVr * rmsVr)
    for i in range(len(vrs)-1):
        v_ar = (np.cos(els[i])*np.sin(azs[i])*u +
                np.cos(els[i])*np.cos(azs[i])*v +
                np.sin(els[i])*(w - np.abs(wts[i])))
        J_o += lambda_o*np.sum(np.square(vrs[i]-v_ar)*weights[i])

    return J_o


def calculate_grad_radial_vel(vrs, els, azs, u, v, w,
                              wts, weights, rmsVr, coeff=10.0):
    for i in range(len(vrs)):
        vrs[i] = np.ma.filled(vrs[i], 0)
        wts[i] = np.ma.filled(wts[i], 0)
        vrs[i][np.isnan(vrs[i])] = 0
        wts[i][np.isnan(wts[i])] = 0

    p_x1 = np.zeros(vrs[1].shape)
    p_y1 = np.zeros(vrs[1].shape)
    p_z1 = np.zeros(vrs[1].shape)
    lambda_o = coeff / (rmsVr * rmsVr)
    for i in range(len(vrs)-1):
        v_ar = (np.cos(els[i])*np.sin(azs[i])*u +
            np.cos(els[i])*np.cos(azs[i])*v +
            np.sin(els[i])*(w - np.abs(wts[i])))
        p_x1 += (2*(v_ar - vrs[i])*np.cos(els[i])*np.sin(azs[i])*weights[i])*lambda_o
        p_y1 += (2*(v_ar - vrs[i])*np.cos(els[i])*np.cos(azs[i])*weights[i])*lambda_o
        p_z1 += (2*(v_ar - vrs[i])*np.sin(els[i])*weights[i])*lambda_o

    # Impermeability condition
    p_z1[0, :, :] = 0
    p_z1[-1, :, :] = 0
    y = np.stack((p_x1, p_y1, p_z1), axis=0)

    return y.flatten()


def calculate_smoothness_cost(u, v, w, Cx=0.0, Cy=0.0, Cz=0.0):
    du = np.zeros(w.shape)
    dv = np.zeros(w.shape)
    dw = np.zeros(w.shape)
    scipy.ndimage.filters.laplace(u,du, mode='wrap')
    scipy.ndimage.filters.laplace(v,dv, mode='wrap')
    scipy.ndimage.filters.laplace(w,dw, mode='wrap')
    return np.sum(Cx*du**2 + Cy*dv**2 + Cz*dw**2)



def calculate_smoothness_gradient(u, v, w, Cx=0.0, Cy=0.0, Cz=0.0):
    du = np.zeros(w.shape)
    dv = np.zeros(w.shape)
    dw = np.zeros(w.shape)
    grad_u = np.zeros(w.shape)
    grad_v = np.zeros(w.shape)
    grad_w = np.zeros(w.shape)
    scipy.ndimage.filters.laplace(u,du, mode='wrap')
    scipy.ndimage.filters.laplace(v,dv, mode='wrap')
    scipy.ndimage.filters.laplace(w,dw, mode='wrap')
    scipy.ndimage.filters.laplace(du, grad_u, mode='wrap')
    scipy.ndimage.filters.laplace(dv, grad_v, mode='wrap')
    scipy.ndimage.filters.laplace(dw, grad_w, mode='wrap')
           
    # Impermeability condition
    grad_w[0, :, :] = 0
    grad_w[-1, :, :] = 0
    y = np.stack([grad_u*Cx*2, grad_v*Cy*2, grad_w*Cz*2], axis=0)
    return y.flatten()



def calculate_mass_continuity(u, v, w, z, dx, dy, dz, coeff=1500.0, anel=1):
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=1)
    dwdz = np.gradient(w, dz, axis=0)

    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.gradient(rho, dz, axis=0)
        anel_term = w/rho*drho_dz
    else:
        anel_term = np.zeros(w.shape)
    return coeff*np.sum(np.square(dudx + dvdy + dwdz + anel_term))/2.0



def calculate_mass_continuity_gradient(u, v, w, z, dx,
                                        dy, dz, coeff=1500.0, anel=1):
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=1)
    dwdz = np.gradient(w, dz, axis=0)
    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.gradient(rho, dz, axis=0)
        anel_term = w/rho*drho_dz
    else:
        anel_term = 0

    div2 = dudx + dvdy + dwdz + anel_term
    
    grad_u = -np.gradient(div2, dx, axis=2)*coeff
    grad_v = -np.gradient(div2, dy, axis=1)*coeff
    grad_w = -np.gradient(div2, dz, axis=0)*coeff
   
    
    # Impermeability condition
    grad_w[0,:,:] = 0
    grad_w[-1,:,:] = 0
    y = np.stack([grad_u, grad_v, grad_w], axis=0)
    return y.flatten()


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
    rho = np.exp(-grid_z/10000.0)
    A[np.logical_and(grid_z < frz, refl < 55)] = -2.6
    B[np.logical_and(grid_z < frz, refl < 55)] = 0.0107
    A[np.logical_and(grid_z < frz, 
                     np.logical_and(refl >= 55, refl < 60))] = -2.5
    B[np.logical_and(grid_z < frz, 
                     np.logical_and(refl >= 55, refl < 60))] = 0.013
    A[np.logical_and(grid_z < frz, refl > 60)] = -3.95
    B[np.logical_and(grid_z < frz, refl > 60)] = 0.0148
    A[np.logical_and(grid_z >= frz, refl < 33)] = -0.817
    B[np.logical_and(grid_z >= frz, refl < 33)] = 0.0063
    A[np.logical_and(grid_z >= frz,
                     np.logical_and(refl >= 33, refl < 49))] = -2.5
    B[np.logical_and(grid_z >= frz,
                     np.logical_and(refl >= 33, refl < 49))] = 0.013
    A[np.logical_and(grid_z >= frz, refl > 49)] = -3.95
    B[np.logical_and(grid_z >= frz, refl > 49)] = 0.0148

    fallspeed = A*np.power(10, refl*B)*np.power(1.2/rho, 0.4)
    del A, B, rho
    return fallspeed



def calculate_background_cost(u, v, w, weights, u_back, v_back, C8=0.01):
    the_shape = u.shape
    cost = 0
    for i in range(the_shape[0]):
        cost += C8*np.sum(np.square(u[i]-u_back[i])*(weights[i]) + np.square(v[i]-v_back[i])*(weights[i]))
    return cost



def calculate_background_gradient(u, v, w, weights, u_back, v_back, C8=0.01):
    the_shape = u.shape
    u_grad = np.zeros(the_shape)
    v_grad = np.zeros(the_shape)
    w_grad = np.zeros(the_shape)

    for i in range(the_shape[0]):
        u_grad[i] = C8*2*(u[i]-u_back[i])*(weights[i])
        v_grad[i] = C8*2*(v[i]-v_back[i])*(weights[i])

    y = np.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten()

