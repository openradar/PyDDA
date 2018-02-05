"""
Created on Wed Jul 19 11:31:02 2017

@author: rjackson
"""

import numpy as np
import pyart

from numba import jit


def calculate_radial_vel_cost_function(vr_1, vr_2, az1, az2, el1, el2, u, v,
                                       w, wt1,
                                       wt2, rmsVr, weights, coeff=1.0, dudt=0.0, dvdt=0.0,
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
    vr_1 = np.ma.filled(vr_1, 0)
    vr_2 = np.ma.filled(vr_2, 0)
    wt1 = np.ma.filled(wt1, 0)
    wt2 = np.ma.filled(wt2, 0)

    v_ar1 = (np.cos(el1)*np.sin(az1)*u + 
             np.cos(el1)*np.cos(az1)*v + 
             np.sin(el1)*(w - np.abs(wt1)))
    v_ar2 = (np.cos(el2)*np.sin(az2)*u + 
             np.cos(el2)*np.cos(az2)*v + 
             np.sin(el2)*(w - np.abs(wt2)))

    lambda_o = coeff/(rmsVr*rmsVr)

    ## Do not consider points where vr_1 or vr_2 is masked

    J_o = lambda_o*(np.sum(np.square(vr_1 - v_ar1)*weights) +
                    np.sum(np.square(vr_2 - v_ar2)*weights))
    
    return J_o



def calculate_grad_radial_vel(vr_1, vr_2, el1, az1, el2, az2, u, v, w,
                              wt1, wt2, weights, rmsVr, coeff=10.0):
    vr_1 = np.ma.filled(vr_1,0)
    vr_2 = np.ma.filled(vr_2,0)
    wt1 = np.ma.filled(wt1, 0)
    wt2 = np.ma.filled(wt2, 0)
    v_ar1 = (np.cos(el1)*np.sin(az1)*u +
            np.cos(el1)*np.cos(az1)*v +
            np.sin(el1)*(w - np.abs(wt1)))
    v_ar2 = (np.cos(el2) * np.sin(az2) * u +
             np.cos(el2) * np.cos(az2) * v +
             np.sin(el2) * (w - np.abs(wt2)))

    lambda_o = coeff / (rmsVr * rmsVr)
    p_x1 = (2*(v_ar1 - vr_1)*np.cos(el1)*np.sin(az1)*weights)*lambda_o
    p_y1 = (2*(v_ar1 - vr_1)*np.cos(el1)*np.cos(az1)*weights)*lambda_o
    p_z1 = (2*(v_ar1 - vr_1)*np.sin(el1)*weights)*lambda_o

    p_x1 += (2*(v_ar2 - vr_2)*np.cos(el2)*np.sin(az2)*weights)*lambda_o
    p_y1 += (2*(v_ar2 - vr_2)*np.cos(el2)*np.cos(az2)*weights)*lambda_o
    p_z1 += (2*(v_ar2 - vr_2)*np.sin(el2)*weights)*lambda_o

    # Impermeability condition
    p_z1[0, :, :] = 0
    p_z1[-1, :, :] = 0
    y = np.stack([p_x1, p_y1, p_z1], axis=0)

    return y.flatten()


def calculate_smoothness_cost(u, v, w, z, el, dx, dy, dz, cutoff=1000.0,
                              C4=50.0, C5=0.0, C6=0.0, C7=50.0, laplace=0):
    if(laplace == 1):
        dudx = np.gradient(u, dx, axis=2)
        dudx2 = np.gradient(dudx, dx, axis=2)
        dvdx = np.gradient(v, dx, axis=2)
        dvdx2 = np.gradient(dvdx, dx, axis=2)
        dwdx = np.gradient(w, dx, axis=2)
        dwdx2 = np.gradient(dwdx, dx, axis=2)

        dudy = np.gradient(u, dy, axis=1)
        dudy2 = np.gradient(dudy, dy, axis=1)
        dvdy = np.gradient(v, dy, axis=1)
        dvdy2 = np.gradient(dvdy, dy, axis=1)
        dwdy = np.gradient(w, dy, axis=1)
        dwdy2 = np.gradient(dwdy, dy, axis=1)

        dudz = np.gradient(u, dz, axis=0)
        dudz2 = np.gradient(dudz, dz, axis=0)
        dvdz = np.gradient(v, dz, axis=0)
        dvdz2 = np.gradient(dvdz, dz, axis=0)
        dwdz = np.gradient(w, dz, axis=0)
        dwdz2 = np.gradient(dwdz, dz, axis=0)

        return np.sum(C5*(np.square(dudz2) + np.square(dvdz2)) +
                      C6*(np.square(dwdz2)) +
                      C7*(np.square(dwdx2) + np.square(dwdy2)) +
                      C4*(np.square(dudx2) + np.square(dudy2) +
                          np.square(dvdy2) + np.square(dvdx2)))
    else:
        dudx = np.gradient(u, dx, axis=2)
        dudy = np.gradient(u, dy, axis=1)
        dudz = np.gradient(u, dz, axis=0)
        dvdx = np.gradient(v, dx, axis=2)
        dvdy = np.gradient(v, dy, axis=1)
        dvdz = np.gradient(v, dz, axis=0)
        dwdx = np.gradient(w, dx, axis=2)
        dwdy = np.gradient(w, dy, axis=1)
        dwdz = np.gradient(w, dz, axis=0)
        return np.sum(C5 * (np.square(dudz) + np.square(dvdz)) +
                      C6 * (np.square(dwdz)) +
                      C7 * (np.square(dwdx) + np.square(dwdy)) +
                      C4 * (np.square(dudx) + np.square(dudy) +
                            np.square(dvdy) + np.square(dvdx)))



def calculate_smoothness_gradient(u, v, w, z, el, dx, dy, dz, cutoff=1000.0,
                                  C4=50.0, C5=0.0, C6=0.0, C7=50.0, laplace=0):
    grad_u = np.zeros(w.shape)
    grad_v = np.zeros(w.shape)
    grad_w = np.zeros(w.shape)
    if(laplace == 1):
        dudx = np.gradient(u, dx, axis=2)
        dudx2 = np.gradient(dudx, dx, axis=2)
        dvdx = np.gradient(v, dx, axis=2)
        dvdx2 = np.gradient(dvdx, dx, axis=2)
        dwdx = np.gradient(w, dx, axis=2)
        dwdx2 = np.gradient(dwdx, dx, axis=2)

        dudy = np.gradient(u, dy, axis=1)
        dudy2 = np.gradient(dudy, dy, axis=1)
        dvdy = np.gradient(v, dy, axis=1)
        dvdy2 = np.gradient(dvdy, dy, axis=1)
        dwdy = np.gradient(w, dy, axis=1)
        dwdy2 = np.gradient(dwdy, dy, axis=1)

        dudz = np.gradient(u, dz, axis=0)
        dudz2 = np.gradient(dudz, dz, axis=0)
        dvdz = np.gradient(v, dz, axis=0)
        dvdz2 = np.gradient(dvdz, dz, axis=0)
        dwdz = np.gradient(w, dz, axis=0)
        dwdz2 = np.gradient(dwdz, dz, axis=0)

        grad_u = 2*C4*(dudx2 + dudy2) + 2*C5*(dudz2)
        grad_v = 2*C4*(dvdx2 + dvdy2) + 2*C5*(dvdz2)
        grad_w = 2*C6*(dwdx2 + dwdy2) + 2*C7*(dwdz2)
    else:
        dudx = np.gradient(u, dx, axis=2)
        dvdx = np.gradient(v, dx, axis=2)
        dwdx = np.gradient(w, dx, axis=2)

        dudy = np.gradient(u, dy, axis=1)
        dvdy = np.gradient(v, dy, axis=1)
        dwdy = np.gradient(w, dy, axis=1)

        dudz = np.gradient(u, dz, axis=0)
        dvdz = np.gradient(v, dz, axis=0)
        dwdz = np.gradient(w, dz, axis=0)

        grad_u = 2 * C4 * (dudx + dudy) + 2 * C5 * (dudz)
        grad_v = 2 * C4 * (dvdx + dvdy) + 2 * C5 * (dvdz)
        grad_w = 2 * C6 * (dwdx + dwdy) + 2 * C7 * (dwdz)

    #grad_u[1:,1:,1:-1] += (8*u[1:,1:,1:-1] - 4 * (u[1:,1:,1:-1] + u[1:,1:,:-2]))/np.power(dx, 4)*C4
    #grad_u[1:,1:,:-2] += (2*u[1:,1:,:-2]-4*u[1:,1:,1:-1]+ 2*u[1:,1:,2:])/np.power(dx,4)*C4
    #grad_u[1:,1:,2:] += (2*u[1:,1:,:-2]-4*u[1:,1:,1:-1] + 2*u[1:,1:,2:])/np.power(dx,4)*C4

    #grad_v[1:, 1:, 1:-1] += (8 * v[1:,1:,1:-1] - 4 * (v[1:, 1:, 1:-1] + v[1:, 1:, :-2])) / np.power(dx, 4) * C4
    #grad_v[1:, 1:, :-2] += (2 * v[1:, 1:, :-2] - 4 * v[1:, 1:, 1:-1] + 2 * v[1:, 1:, 2:]) / np.power(dx, 4) * C4
    #grad_v[1:, 1:, 2:] += (2 * v[1:, 1:, :-2] - 4 * v[1:, 1:, 1:-1] + 2 * v[1:, 1:, 2:]) / np.power(dx, 4) * C4

    #grad_w[1:, 1:, 1:-1] += (8 * w[1:,1:,1:-1] - 4 * (w[1:, 1:, 1:-1] + w[1:, 1:, :-2])) / np.power(dx, 4) * C7
    #grad_w[1:, 1:, :-2] += (2 * w[1:, 1:, :-2] - 4 * w[1:, 1:, 1:-1] + 2 * w[1:, 1:, 2:]) / np.power(dx, 4) * C7
    #grad_w[1:, 1:, 2:] += (2 * w[1:, 1:, :-2] - 4 * w[1:, 1:, 1:-1] + 2 * w[1:, 1:, 2:]) / np.power(dx, 4) * C7

    #grad_u[1:, 1:-1, 1:] += (8 * u[1:, 1:-1, 1:] - 4 * (u[1:, 1:-1, 1:] + u[1:, :-2, 1:])) / np.power(dy, 4) * C4
    #grad_u[1:, :-2, 1:] += (2 * u[1:, :-2, 1:] - 4 * u[1:, 1:-1, 1:] + 2 * u[1:, 2:, 1:]) / np.power(dy, 4) * C4
    #grad_u[1:, 2:, 1:] += (2 * u[1:, :-2, 1:] - 4 * u[1:, 1:-1, 1:] + 2 * u[1:, 2:, 1:]) / np.power(dy, 4) * C4

    #grad_v[1:, 1:-1, 1:] += (8 * v[1:,1:-1,1:] - 4 * (v[1:, 1:-1, 1:] + v[1:, :-2, 1:])) / np.power(dy, 4) * C4
    #grad_v[1:, :-2, 1:] += (2 * v[1:, :-2, 1:] - 4 * v[1:, 1:-1, 1:] + 2 * v[1:, 2:, 1:]) / np.power(dy, 4) * C4
    #grad_v[1:, 2:, 1:] += (2 * v[1:, :-2, 1:] - 4 * v[1:, 1:-1, 1:] + 2 * v[1:, 2:, 1:]) / np.power(dy, 4) * C4

    #grad_w[1:, 1:-1, 1:] += (8 * w[1:,1:-1,1:] - 4 * (w[1:, 1:-1, 1:] + w[1:, :-2, 1:])) / np.power(dy, 4) * C7
    #grad_w[1:, :-2, 1:] += (2 * w[1:, :-2, 1:] - 4 * w[1:, 1:-1, 1:] + 2 * w[1:, 2:, 1:]) / np.power(dy, 4) * C7
    #grad_w[1:, 2:, 1:] += (2 * w[1:, :-2, 1:] - 4 * w[1:, 1:-1, 1:] + 2 * w[1:, 2:, 1:]) / np.power(dy, 4) * C7

    #grad_u[1:-1, 1:, 1:] += (8 * u[1:-1, 1:, 1:] - 4 * (u[1:-1, 1:, 1:] + u[:-2, 1:, 1:])) / np.power(dz, 4) * C5
    #grad_u[:-2, 1:, 1:] += (2 * u[:-2, 1:, 1:] - 4 * u[1:-1, 1:, 1:] + 2 * u[2:, 1:, 1:]) / np.power(dz, 4) * C5
    #grad_u[2:, 1:, 1:] += (2 * u[2:, 1:, :1] - 4 * u[1:-1, 1:, 1:] + 2 * u[2:, 1:, 1:]) / np.power(dz, 4) * C5

    #grad_v[1:-1, 1:, 1:] += (8 * v[1:-1, 1:, 1:] - 4 * (v[1:-1, 1:, 1:] + v[:-2, 1:, 1:])) / np.power(dz, 4) * C5
    #grad_v[:-2, 1:, 1:] += (2 * v[:-2, 1:, 1:] - 4 * v[1:-1, 1:, 1:] + 2 * v[2:, 1:, 1:]) / np.power(dz, 4) * C5
    #grad_v[2:, 1:, 1:] += (2 * v[2:, 1:, 1:] - 4 * v[1:-1, 1:, 1:] + 2 * v[2:, 1:, 1:]) / np.power(dz, 4) * C5

    #grad_w[1:-1, 1:, 1:] += (8 * w[1:-1, 1:, 1:] - 4 * (w[1:-1, 1:, 1:] + w[:-2, 1:, 1:])) / np.power(dz, 4) * C6
    #grad_w[:-2, 1:, 1:] += (2 * w[:-2, 1:, 1:] - 4 * w[1:-1, 1:, 1:] + 2 * w[2:, 1:, 1:]) / np.power(dz, 4) * C6
    #grad_w[2:, 1:, 1:] += (2 * w[:-2, 1:, 1:] - 4 * w[1:-1, 1:, 1:] + 2 * w[2:, 1:, 1:]) / np.power(dz, 4) * C6

    # Impermeability condition
    grad_w[0, :, :] = 0
    grad_w[-1, :, :] = 0
    y = np.stack([grad_u, grad_v, grad_w], axis=0)
    return y.flatten()


def calculate_mass_continuity(u, v, w, z, el, dx, dy, dz, coeff=1500.0, anel=1):
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=1)
    dwdz = np.gradient(w, dz, axis=0)

    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.gradient(rho, dz, axis=0)
        anel_term = w/rho*drho_dz
    else:
        anel_term = np.zeros(w.shape)
    return coeff*np.sum(np.square(dudx + dvdy + dwdz + anel_term))


def calculate_mass_continuity_gradient(u, v, w, z, el, dx,
                                        dy, dz, coeff=1500.0, anel=1):
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=1)
    dwdz = np.gradient(w, dz, axis=0)
    if(anel == 1):
        rho = np.exp(-z/10000.0)
        drho_dz = np.gradient(rho, dz, axis=0)
        anel_term = w/rho*drho_dz
        anel_term_adj = 1 + drho_dz/rho
    else:
        anel_term = np.zeros(w.shape)
        anel_term_adj = np.ones(w.shape)

    div2 = dudx + dvdy + dwdz + anel_term
    grad_u = 2*div2*coeff
    grad_v = 2*div2*coeff
    grad_w = 2*div2*anel_term_adj*coeff
    #grad_u[:-1,1:-1,:-2] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    #grad_u[:-1,1:-1,2:] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    #grad_u[1:,1:-1,:-2] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    #grad_u[1:,1:-1,2:] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dx)
    
    #grad_v[:-1,2:,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    #grad_v[:-1,:-2,1:-1] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    #grad_v[1:,2:,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    #grad_v[1:,:-2,1:-1] -= 2*coeff*div2[:-1,1:-1,1:-1]/(4*dy)
    
    #grad_w[1:,1:-1,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]/dz
    #grad_w[:-1,1:-1,1:-1] += 2*coeff*div2[:-1,1:-1,1:-1]*(anel_term_adj[:-1,1:-1,1:-1]-1/dz)

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
        cost += C8*np.sum(np.square(u[i]-u_back[i])*(weights) + np.square(v[i]-v_back[i])*(weights))
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

