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
                                       wt2, coeff=1.0, dudt=0.0, dvdt=0.0, 
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
    M1 = len(np.where(vr_1.mask == False))
    M2 = len(np.where(vr_2.mask == False))
    v_ar1 = (np.cos(el1)*np.sin(az1)*u + 
             np.cos(el1)*np.cos(az1)*v + 
             np.sin(el1)*(w - wt1))
    v_ar2 = (np.cos(el2)*np.sin(az2)*u + 
             np.cos(el2)*np.cos(az2)*v + 
             np.sin(el2)*(w - wt2))
    lambda_o = coeff*(1/(M1+M2)/(np.sum(np.square(vr_1)) +
                      np.sum(np.square(vr_2)))) 
    J_o = lambda_o*(np.sum(coeff*np.square(vr_1 - v_ar1) +
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

#def calculate_smoothness_cost():
    
def calculate_mass_continuity(u, v, w, z, el, dx, dy, dz, coeff=1500.0, anel=1):
    dudx = np.empty_like(u)
    dudx[:,:,:-1] = np.diff(u, axis=2)/dx
    dudx[:,:,-1] = -u[:,:,-1]/dx
    dvdy = np.empty_like(v)
    dvdy[:,:-1,:] = np.diff(v, axis=1)/dy
    dvdy[:,-1,:] = -v[:,-1,:]/dy
    dwdz = np.empty_like(w)
    dwdz[:-1,:,:] = np.diff(w, axis=0)/dz
    dwdz[-1] = -w[-1]/dz
    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.empty_like(rho)
        drho_dz[:-1] = np.diff(rho, axis=0)/dz
        drho_dz[-1] = -rho[-1]/dz
        anel_term = w/rho*drho_dz
    else:
        anel_term = np.zeros(w.shape)
    return coeff*np.sum(np.square((dudx[:-1,1:,1:] + dudx[:-1,1:,:-1])/2.0 + 
                                  (dvdy[:-1,1:,1:] + dvdy[:-1,1:,:-1])/2.0 +
                                   dwdz[:-1,1:,1:] + anel_term[:-1,1:,1:]))

def calculate_mass_continuity_gradient(u, v, w, z, el, dx, 
                                        dy, dz, coeff=1500.0, anel=1):
    dudx = np.empty_like(u)
    dudx[:,:,:-1] = np.diff(u, axis=2)/dx
    dudx[:,:,-1] = -u[:,:,-1]/dx
    dvdy = np.empty_like(v)
    dvdy[:,:-1,:] = np.diff(v, axis=1)/dy
    dvdy[:,-1,:] = -v[:,-1,:]/dy
    dwdz = np.empty_like(w)
    dwdz[:-1,:,:] = np.diff(w, axis=0)/dz
    dwdz[-1] = -w[-1]/dz
    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.empty_like(rho)
        drho_dz[:-1] = np.diff(rho, axis=0)/dz
        drho_dz[-1] = -rho[-1]/dz
        anel_term = w/rho*drho_dz
        anel_term_adj = np.empty_like(anel_term)
        anel_term_adj[-1] = rho[-1]/dz
        anel_term_adj[:-1,:,:] = 2/(rho[:-1,:,:]+rho[1:,:,:])*(rho[1:,:,:]-rho[:-1,:,:])/dz
    else:
        anel_term = np.zeros(w.shape)
        anel_term_adj = np.zeros(w.shape)
        
    div2 = np.empty_like(w)
    div2[:-1,1:-1,1:-1] = coeff*((dudx[:-1,1:-1,1:-1] + dudx[:-1,1:-1,1:-1])/2.0 +
        (dvdy[:-1,1:-1,1:-1] + dvdy[:-1,1:-1,1:-1])/2.0 +
        (dwdz[:-1,1:-1,1:-1] + anel_term[:-1,1:-1,1:-1]))
    grad_u = np.zeros(w.shape)
    grad_v = np.zeros(w.shape)
    grad_w = np.zeros(w.shape)
    grad_u[:-1,1:-1,:-2] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    grad_u[:-1,1:-1,2:] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    grad_u[1:,1:-1,:-2] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    grad_u[1:,1:-1,2:] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    
    grad_v[:-1,2:,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    grad_v[:-1:,:-2,1:-1] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    grad_v[1:,2:,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    grad_v[1:,:-2,1:-1] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    
    grad_w[1:,1:-1,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]//dz
    grad_w[:-1,1:-1,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]*(anel_term_adj[:-1,1:-1,1:-1]-1/dz)
    y = np.stack([grad_u, grad_v, grad_w], axis=0)
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
    rho = np.exp(grid_z/10000.0)
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
