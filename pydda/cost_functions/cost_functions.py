import numpy as np
import pyart
import scipy.ndimage.filters


def J_function(winds, parameters):
    """
    Calculates the total cost function. This typically does not need to be
    called directly as get_dd_wind_field is a wrapper around this function and
    :py:func:`pydda.cost_functions.grad_J`.
    In order to add more terms to the cost function, modify this
    function and :py:func:`pydda.cost_functions.grad_J`.

    Parameters
    ----------
    winds: 1-D float array
        The wind field, flattened to 1-D for f_min. The total size of the
        array will be a 1D array of 3*nx*ny*nz elements.
    parameters: DDParameters
        The parameters for the cost function evaluation as specified by the
        :py:func:`pydda.retrieval.DDParameters` class.

    Returns
    -------
    J: float
        The value of the cost function
    """
    winds = np.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jvel = calculate_radial_vel_cost_function(
         parameters.vrs, parameters.azs, parameters.els,
         winds[0], winds[1], winds[2], parameters.wts, rmsVr=parameters.rmsVr,
         weights=parameters.weights, coeff=parameters.Co)

    if(parameters.Cm > 0):
        Jmass = calculate_mass_continuity(
            winds[0], winds[1], winds[2], parameters.z,
            parameters.dx, parameters.dy, parameters.dz,
            coeff=parameters.Cm)
    else:
        Jmass = 0

    if(parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0):
        Jsmooth = calculate_smoothness_cost(
            winds[0], winds[1], winds[2], Cx=parameters.Cx,
            Cy=parameters.Cy, Cz=parameters.Cz)
    else:
        Jsmooth = 0

    if(parameters.Cb > 0):
        Jbackground = calculate_background_cost(
            winds[0], winds[1], winds[2], parameters.bg_weights,
            parameters.u_back, parameters.v_back, parameters.Cb)
    else:
        Jbackground = 0

    if(parameters.Cv > 0):
        Jvorticity = calculate_vertical_vorticity_cost(
            winds[0], winds[1], winds[2], parameters.dx,
            parameters.dy, parameters.dz, parameters.Ut,
            parameters.Vt, coeff=parameters.Cv)
    else:
        Jvorticity = 0

    if(parameters.Cmod > 0):
        Jmod = calculate_model_cost(
            winds[0], winds[1], winds[2],
            parameters.model_weights, parameters.u_model,
            parameters.v_model,
            parameters.w_model, coeff=parameters.Cmod)
    else:
        Jmod = 0

    if parameters.Cpoint > 0:
        Jpoint = calculate_point_cost(
            winds[0], winds[1], parameters.x, parameters.y, parameters.z,
            parameters.point_list, Cp=parameters.Cpoint, roi=parameters.roi)
    else:
        Jpoint = 0

    if(parameters.print_out is True):
        print(('| Jvel    | Jmass   | Jsmooth |   Jbg   | Jvort   | Jmodel  | Jpoint  |' +
               ' Max w  '))
        print(('|' + "{:9.4f}".format(Jvel) + '|' +
               "{:9.4f}".format(Jmass) + '|' +
               "{:9.4f}".format(Jsmooth) + '|' +
               "{:9.4f}".format(Jbackground) + '|' +
               "{:9.4f}".format(Jvorticity) + '|' +
               "{:9.4f}".format(Jmod) + '|' +
               "{:9.4f}".format(Jpoint)) + '|' +
               "{:9.4f}".format(np.ma.max(np.ma.abs(winds[2]))))

    return Jvel + Jmass + Jsmooth + Jbackground + Jvorticity + Jmod + Jpoint


def grad_J(winds, parameters):
    """
    Calculates the gradient of the cost function. This typically does not need
    to be called directly as get_dd_wind_field is a wrapper around this
    function and :py:func:`pydda.cost_functions.J_function`.
    In order to add more terms to the cost function,
    modify this function and :py:func:`pydda.cost_functions.grad_J`.

    Parameters
    ----------
    winds: 1-D float array
        The wind field, flattened to 1-D for f_min
    parameters: DDParameters
        The parameters for the cost function evaluation as specified by the
        :py:func:`pydda.retrieve.DDParameters` class.

    Returns
    -------
    grad: 1D float array
        Gradient vector of cost function
    """
    winds = np.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))
    grad = calculate_grad_radial_vel(
        parameters.vrs, parameters.els, parameters.azs,
        winds[0], winds[1], winds[2], parameters.wts, parameters.weights,
        parameters.rmsVr, coeff=parameters.Co, upper_bc=parameters.upper_bc)

    if(parameters.Cm > 0):
        grad += calculate_mass_continuity_gradient(
            winds[0], winds[1], winds[2], parameters.z,
            parameters.dx, parameters.dy, parameters.dz,
            coeff=parameters.Cm, upper_bc=parameters.upper_bc)

    if(parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0):
        grad += calculate_smoothness_gradient(
            winds[0], winds[1], winds[2], Cx=parameters.Cx,
            Cy=parameters.Cy, Cz=parameters.Cz, upper_bc=parameters.upper_bc)

    if(parameters.Cb > 0):
        grad += calculate_background_gradient(
            winds[0], winds[1], winds[2], parameters.bg_weights,
            parameters.u_back, parameters.v_back, parameters.Cb,
            upper_bc=parameters.upper_bc)

    if(parameters.Cv > 0):
        grad += calculate_vertical_vorticity_gradient(
            winds[0], winds[1], winds[2], parameters.dx,
            parameters.dy, parameters.dz, parameters.Ut,
            parameters.Vt, coeff=parameters.Cv)

    if(parameters.Cmod > 0):
        grad += calculate_model_gradient(
            winds[0], winds[1], winds[2],
            parameters.model_weights, parameters.u_model, parameters.v_model,
            parameters.w_model, coeff=parameters.Cmod)

    if parameters.Cpoint > 0:
        grad += calculate_point_gradient(
            winds[0], winds[1], parameters.x, parameters.y, parameters.z,
            parameters.point_list, Cp=parameters.Cpoint, roi=parameters.roi)

    if(parameters.print_out is True):
        print('Norm of gradient: ' + str(np.linalg.norm(grad, np.inf)))

    return grad


def calculate_radial_vel_cost_function(vrs, azs, els, u, v,
                                       w, wts, rmsVr, weights, coeff=1.0):
    """
    Calculates the cost function due to difference of the wind field from
    radar radial velocities. For more information on this cost function, see
    Potvin et al. (2012) and Shapiro et al. (2009).

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    vrs: List of float arrays
        List of radial velocities from each radar
    els: List of float arrays
        List of elevations from each radar
    azs: List of float arrays
        List of azimuths from each radar
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    wts: List of float arrays
        Float array containing fall speed from radar.
    rmsVr: float
        The sum of squares of velocity/num_points. Use for normalization
        of data weighting coefficient
    weights: n_radars x_bins x y_bins float array
        Data weights for each pair of radars
    coeff: float
        Constant for cost function

    Returns
    -------
    J_o: float
         Observational cost function

    References
    -----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
    """

    J_o = 0
    lambda_o = coeff / (rmsVr * rmsVr)
    for i in range(len(vrs)):
        v_ar = (np.cos(els[i])*np.sin(azs[i])*u +
                np.cos(els[i])*np.cos(azs[i])*v +
                np.sin(els[i])*(w - np.abs(wts[i])))
        the_weight = weights[i]
        the_weight[els[i].mask] = 0
        the_weight[azs[i].mask] = 0
        the_weight[vrs[i].mask] = 0
        the_weight[wts[i].mask] = 0
        J_o += lambda_o*np.sum(np.square(vrs[i] - v_ar)*the_weight)

    return J_o


def calculate_grad_radial_vel(vrs, els, azs, u, v, w,
                              wts, weights, rmsVr, coeff=1.0, upper_bc=True):
    """
    Calculates the gradient of the cost function due to difference of wind
    field from radar radial velocities.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

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
    vel_name: str
        Background velocity field name
    weights: n_radars x_bins x y_bins float array
        Data weights for each pair of radars

    Returns
    -------
    y: 1-D float array
         Gradient vector of observational cost function. 
         
    More information
    ----------------
    The gradient is calculated by taking the functional derivative of the
    cost function. For more information on functional derivatives, see the
    Euler-Lagrange Equation:

    https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation

    """

    # Use zero for all masked values since we don't want to add them into
    # the cost function

    p_x1 = np.zeros(vrs[0].shape)
    p_y1 = np.zeros(vrs[0].shape)
    p_z1 = np.zeros(vrs[0].shape)
    lambda_o = coeff / (rmsVr * rmsVr)

    for i in range(len(vrs)):
        v_ar = (np.cos(els[i])*np.sin(azs[i])*u +
                np.cos(els[i])*np.cos(azs[i])*v +
                np.sin(els[i])*(w - np.abs(wts[i])))

        x_grad = (2*(v_ar - vrs[i]) * np.cos(els[i]) *
                  np.sin(azs[i]) * weights[i]) * lambda_o
        y_grad = (2*(v_ar - vrs[i]) * np.cos(els[i]) *
                  np.cos(azs[i]) * weights[i]) * lambda_o
        z_grad = (2*(v_ar - vrs[i]) * np.sin(els[i]) * weights[i]) * lambda_o

        x_grad[els[i].mask] = 0
        y_grad[els[i].mask] = 0
        z_grad[els[i].mask] = 0
        x_grad[azs[i].mask] = 0
        y_grad[azs[i].mask] = 0
        z_grad[azs[i].mask] = 0

        x_grad[els[i].mask] = 0
        x_grad[azs[i].mask] = 0
        x_grad[vrs[i].mask] = 0
        x_grad[wts[i].mask] = 0
        y_grad[els[i].mask] = 0
        y_grad[azs[i].mask] = 0
        y_grad[vrs[i].mask] = 0
        y_grad[wts[i].mask] = 0
        z_grad[els[i].mask] = 0
        z_grad[azs[i].mask] = 0
        z_grad[vrs[i].mask] = 0
        z_grad[wts[i].mask] = 0

        p_x1 += x_grad
        p_y1 += y_grad
        p_z1 += z_grad

    # Impermeability condition
    p_z1[0, :, :] = 0
    if(upper_bc is True):
        p_z1[-1, :, :] = 0
    y = np.stack((p_x1, p_y1, p_z1), axis=0)
    return y.flatten()


def calculate_smoothness_cost(u, v, w, Cx=1e-5, Cy=1e-5, Cz=1e-5):
    """
    Calculates the smoothness cost function by taking the Laplacian of the
    wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    Cx: float
        Constant controlling smoothness in x-direction
    Cy: float
        Constant controlling smoothness in y-direction
    Cz: float
        Constant controlling smoothness in z-direction

    Returns
    -------
    Js: float
        value of smoothness cost function
    """
    du = np.zeros(w.shape)
    dv = np.zeros(w.shape)
    dw = np.zeros(w.shape)
    scipy.ndimage.filters.laplace(u, du, mode='wrap')
    scipy.ndimage.filters.laplace(v, dv, mode='wrap')
    scipy.ndimage.filters.laplace(w, dw, mode='wrap')
    return np.sum(Cx*du**2 + Cy*dv**2 + Cz*dw**2)


def calculate_smoothness_gradient(u, v, w, Cx=1e-5, Cy=1e-5, Cz=1e-5,
                                  upper_bc=True):
    """
    Calculates the gradient of the smoothness cost function
    by taking the Laplacian of the Laplacian of the wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    Cx: float
        Constant controlling smoothness in x-direction
    Cy: float
        Constant controlling smoothness in y-direction
    Cz: float
        Constant controlling smoothness in z-direction

    Returns
    -------
    y: float array
        value of gradient of smoothness cost function
    """
    du = np.zeros(w.shape)
    dv = np.zeros(w.shape)
    dw = np.zeros(w.shape)
    grad_u = np.zeros(w.shape)
    grad_v = np.zeros(w.shape)
    grad_w = np.zeros(w.shape)
    scipy.ndimage.filters.laplace(u, du, mode='wrap')
    scipy.ndimage.filters.laplace(v, dv, mode='wrap')
    scipy.ndimage.filters.laplace(w, dw, mode='wrap')
    scipy.ndimage.filters.laplace(du, grad_u, mode='wrap')
    scipy.ndimage.filters.laplace(dv, grad_v, mode='wrap')
    scipy.ndimage.filters.laplace(dw, grad_w, mode='wrap')

    # Impermeability condition
    grad_w[0, :, :] = 0
    if(upper_bc is True):
        grad_w[-1, :, :] = 0
    y = np.stack([grad_u*Cx*2, grad_v*Cy*2, grad_w*Cz*2], axis=0)
    return y.flatten()


def calculate_point_cost(u, v, x, y, z, point_list, Cp=1e-3, roi=500.0):
    """
    Calculates the cost function related to point observations. A mean square error cost
    function term is applied to points that are within the sphere of influence
    whose radius is determined by *roi*.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    x:  Float array
        X coordinates of grid centers
    y:  Float array
        Y coordinates of grid centers
    z:  Float array
        Z coordinated of grid centers
    point_list: list of dicts
        List of point constraints.
        Each member is a dict with keys of "u", "v", to correspond
        to each component of the wind field and "x", "y", "z"
        to correspond to the location of the point observation.

        In addition, "site_id" gives the METAR code (or name) to the station.
    Cp: float
        The weighting coefficient of the point cost function.
    roi: float
        Radius of influence of observations

    Returns
    -------
    J: float
        The cost function related to the difference between wind field and points.

    """
    J = 0.0
    for the_point in point_list:
        # Instead of worrying about whole domain, just find points in radius of influence
        # Since we know that the weight will be zero outside the sphere of influence anyways
        the_box = np.where(np.logical_and.reduce(
            (np.abs(x - the_point["x"]) < roi, np.abs(y - the_point["y"]) < roi,
                                            np.abs(z - the_point["z"]) < roi)))
        J += np.sum(((u[the_box] - the_point["u"])**2 + (v[the_box] - the_point["v"])**2))

    return J * Cp


def calculate_point_gradient(u, v, x, y, z, point_list, Cp=1e-3, roi=500.0):
    """
    Calculates the gradient of the cost function related to point observations.
    A mean square error cost function term is applied to points that are within the sphere of influence
    whose radius is determined by *roi*.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    x: Float array
        X coordinates of grid centers
    y: Float array
        Y coordinates of grid centers
    z: Float array
        Z coordinated of grid centers
    point_list: list of dicts
        List of point constraints. Each member is a dict with keys of "u", "v",
        to correspond to each component of the wind field and "x", "y", "z"
        to correspond to the location of the point observation.

        In addition, "site_id" gives the METAR code (or name) to the station.
    Cp: float
        The weighting coefficient of the point cost function.
    roi: float
        Radius of influence of observations

    Returns
    -------
    gradJ: float array
        The gradient of the cost function related to the difference between wind field and points.

    """

    gradJ_u = np.zeros_like(u)
    gradJ_v = np.zeros_like(v)
    gradJ_w = np.zeros_like(u)

    for the_point in point_list:
        the_box = np.where(np.logical_and.reduce(
            (np.abs(x - the_point["x"]) < roi, np.abs(y - the_point["y"]) < roi,
             np.abs(z - the_point["z"]) < roi)))
        gradJ_u[the_box] += 2 * (u[the_box] - the_point["u"])
        gradJ_v[the_box] += 2 * (v[the_box] - the_point["v"])

    gradJ = np.stack([gradJ_u, gradJ_v, gradJ_w], axis=0).flatten()
    return gradJ * Cp


def calculate_mass_continuity(u, v, w, z, dx, dy, dz, coeff=1500.0, anel=1):
    """
    Calculates the mass continuity cost function by taking the divergence
    of the wind field.

    All arrays in the given lists must have the same dimensions and represent
    the same spatial coordinates.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    dx: float
        Grid spacing in x direction.
    dy: float
        Grid spacing in y direction.
    dz: float
        Grid spacing in z direction.
    z: Float array (1D)
        1D Float array with heights of grid
    coeff: float
        Constant controlling contribution of mass continuity to cost function
    anel: int
        = 1 use anelastic approximation, 0=don't

    Returns
    -------
    J: float
        value of mass continuity cost function
    """
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
                                       dy, dz, coeff=1500.0, anel=1,
                                       upper_bc=True):
    """
    Calculates the gradient of mass continuity cost function. This is done by
    taking the negative gradient of the divergence of the wind field.

    All grids must have the same grid specification.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    z: Float array (1D)
        1D Float array with heights of grid
    dx: float
        Grid spacing in x direction.
    dy: float
        Grid spacing in y direction.
    dz: float
        Grid spacing in z direction.
    coeff: float
        Constant controlling contribution of mass continuity to cost function
    anel: int
        = 1 use anelastic approximation, 0=don't

    Returns
    -------
    y: float array
        value of gradient of mass continuity cost function
    """
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
    grad_w[0, :, :] = 0
    if(upper_bc is True):
        grad_w[-1, :, :] = 0
    y = np.stack([grad_u, grad_v, grad_w], axis=0)
    return y.flatten()


def calculate_fall_speed(grid, refl_field=None, frz=4500.0):
    """
    Estimates fall speed based on reflectivity.

    Uses methodology of Mike Biggerstaff and Dan Betten

    Parameters
    ----------
    Grid: Py-ART Grid
        Py-ART Grid containing reflectivity to calculate fall speed from
    refl_field: str
        String containing name of reflectivity field. None will automatically
        determine the name.
    frz: float
        Height of freezing level in m

    Returns
    -------
    3D float array:
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


def calculate_background_cost(u, v, w, weights, u_back, v_back, Cb=0.01):
    """
    Calculates the background cost function. The background cost function is
    simply the sum of the squared differences between the wind field and the
    background wind field multiplied by the weighting coefficient.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: Float array
        Weights for each point to consider into cost function
    u_back: 1D float array
        Zonal winds vs height from sounding
    w_back: 1D float array
        Meridional winds vs height from sounding
    Cb: float
        Weight of background constraint to total cost function

    Returns
    -------
    cost: float
        value of background cost function
    """
    the_shape = u.shape
    cost = 0
    for i in range(the_shape[0]):
        cost += (Cb*np.sum(np.square(u[i]-u_back[i])*(weights[i]) +
                           np.square(v[i]-v_back[i])*(weights[i])))
    return cost


def calculate_background_gradient(u, v, w, weights, u_back, v_back, Cb=0.01):
    """
    Calculates the gradient of the background cost function. For each u, v
    this is given as 2*coefficent*(analysis wind - background wind).

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: Float array
        Weights for each point to consider into cost function
    u_back: 1D float array
        Zonal winds vs height from sounding
    w_back: 1D float array
        Meridional winds vs height from sounding
    Cb: float
        Weight of background constraint to total cost function

    Returns
    -------
    y: float array
        value of gradient of background cost function
    """
    the_shape = u.shape
    u_grad = np.zeros(the_shape)
    v_grad = np.zeros(the_shape)
    w_grad = np.zeros(the_shape)

    for i in range(the_shape[0]):
        u_grad[i] = Cb*2*(u[i]-u_back[i])*(weights[i])
        v_grad[i] = Cb*2*(v[i]-v_back[i])*(weights[i])

    y = np.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten()


def calculate_vertical_vorticity_cost(u, v, w, dx, dy, dz, Ut, Vt,
                                      coeff=1e-5):
    """
    Calculates the cost function due to deviance from vertical vorticity
    equation. For more information of the vertical vorticity cost function,
    see Potvin et al. (2012) and Shapiro et al. (2009).

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    dx: float array
        Spacing in x grid
    dy: float array
        Spacing in y grid
    dz: float array
        Spacing in z grid
    coeff: float
        Weighting coefficient
    Ut: float
        U component of storm motion
    Vt: float
        V component of storm motion

    Returns
    -------
    Jv: float
        Value of vertical vorticity cost function.

    References
    ----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
    """
    dvdz = np.gradient(v, dz, axis=0)
    dudz = np.gradient(u, dz, axis=0)
    dwdz = np.gradient(w, dx, axis=2)
    dvdx = np.gradient(v, dx, axis=2)
    dwdy = np.gradient(w, dy, axis=1)
    dwdx = np.gradient(w, dx, axis=2)
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=2)
    dudy = np.gradient(u, dy, axis=1)
    zeta = dvdx - dudy
    dzeta_dx = np.gradient(zeta, dx, axis=2)
    dzeta_dy = np.gradient(zeta, dy, axis=1)
    dzeta_dz = np.gradient(zeta, dz, axis=0)
    jv_array = ((u - Ut) * dzeta_dx + (v - Vt) * dzeta_dy +
                w * dzeta_dz + (dvdz * dwdx - dudz * dwdy) +
                zeta * (dudx + dvdy))
    return np.sum(coeff*jv_array**2)


def calculate_vertical_vorticity_gradient(u, v, w, dx, dy, dz, Ut, Vt,
                                          coeff=1e-5):
    """
    Calculates the gradient of the cost function due to deviance from vertical
    vorticity equation. This is done by taking the functional derivative of
    the vertical vorticity cost function.

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    dx: float array
        Spacing in x grid
    dy: float array
        Spacing in y grid
    dz: float array
        Spacing in z grid
    Ut: float
        U component of storm motion
    Vt: float
        V component of storm motion
    coeff: float
        Weighting coefficient

    Returns
    -------
    Jv: 1D float array
        Value of the gradient of the vertical vorticity cost function.

    References
    ----------

    Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity
    Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and
    Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49,
    https://doi.org/10.1175/JTECH-D-11-00019.1

    Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity
    Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic
    Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
    """

    # First derivatives
    dvdz = np.gradient(v, dz, axis=0)
    dudz = np.gradient(u, dz, axis=0)
    dwdy = np.gradient(w, dy, axis=1)
    dudx = np.gradient(u, dx, axis=2)
    dvdy = np.gradient(v, dy, axis=2)
    dwdx = np.gradient(w, dx, axis=2)
    dvdx = np.gradient(v, dx, axis=2)
    dwdx = np.gradient(w, dx, axis=2)
    dudz = np.gradient(u, dz, axis=0)
    dudy = np.gradient(u, dy, axis=1)

    zeta = dvdx - dudy
    dzeta_dx = np.gradient(zeta, dx, axis=2)
    dzeta_dy = np.gradient(zeta, dy, axis=1)
    dzeta_dz = np.gradient(zeta, dz, axis=0)

    # Second deriviatives
    dwdydz = np.gradient(dwdy, dz, axis=0)
    dwdxdz = np.gradient(dwdx, dz, axis=0)
    dudzdy = np.gradient(dudz, dy, axis=1)
    dvdxdy = np.gradient(dvdx, dy, axis=1)
    dudx2 = np.gradient(dudx, dx, axis=2)
    dudxdy = np.gradient(dudx, dy, axis=1)
    dudxdz = np.gradient(dudx, dz, axis=0)
    dudy2 = np.gradient(dudx, dy, axis=1)

    dzeta_dt = ((u - Ut)*dzeta_dx + (v - Vt)*dzeta_dy + w*dzeta_dz +
                (dvdz*dwdx - dudz*dwdy) + zeta*(dudx + dvdy))

    # Now we intialize our gradient value
    u_grad = np.zeros(u.shape)
    v_grad = np.zeros(v.shape)
    w_grad = np.zeros(w.shape)

    # Vorticity Advection
    u_grad += dzeta_dx + (Ut - u)*dudxdy + (Vt - v)*dudxdy
    v_grad += dzeta_dy + (Vt - v)*dvdxdy + (Ut - u)*dvdxdy
    w_grad += dzeta_dz

    # Tilting term
    u_grad += dwdydz
    v_grad += dwdxdz
    w_grad += dudzdy - dudxdz

    # Stretching term
    u_grad += -dudxdy + dudy2 - dzeta_dx
    u_grad += -dudx2 + dudxdy - dzeta_dy

    # Multiply by 2*dzeta_dt according to chain rule
    u_grad = u_grad*2*dzeta_dt*coeff
    v_grad = v_grad*2*dzeta_dt*coeff
    w_grad = w_grad*2*dzeta_dt*coeff

    y = np.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten()


def calculate_model_cost(u, v, w, weights, u_model, v_model, w_model,
                         coeff=1.0):
    """
    Calculates the cost function for the model constraint.
    This is calculated simply as the sum of squares of the differences
    between the model wind field and the analysis wind field. Vertical
    velocities are not factored into this cost function as there is typically
    a high amount of uncertainty in model derived vertical velocities.

    Parameters
    ----------
    u: 3D array
        Float array with u component of wind field
    v: 3D array
        Float array with v component of wind field
    w: 3D array
        Float array with w component of wind field
    weights: list of 3D arrays
        Float array showing how much each point from model weighs into
        constraint.
    u_model: list of 3D arrays
        Float array with u component of wind field from model
    v_model: list of 3D arrays
        Float array with v component of wind field from model
    w_model: list of 3D arrays
        Float array with w component of wind field from model
    coeff: float
        Weighting coefficient

    Returns
    -------
    Jv: float
        Value of model cost function
    """

    cost = 0
    for i in range(len(u_model)):
        cost += (coeff*np.sum(np.square(u-u_model[i])*weights[i] +
                              np.square(v-v_model[i])*weights[i]))
    return cost


def calculate_model_gradient(u, v, w, weights, u_model,
                             v_model, w_model, coeff=1.0):
    """
    Calculates the cost function for the model constraint.
    This is calculated simply as twice the differences
    between the model wind field and the analysis wind field for each u, v.
    Vertical velocities are not factored into this cost function as there is
    typically a high amount of uncertainty in model derived vertical
    velocities. Therefore, the gradient for all of the w's will be 0.

    Parameters
    ----------
    u: Float array
        Float array with u component of wind field
    v: Float array
        Float array with v component of wind field
    w: Float array
        Float array with w component of wind field
    weights: list of 3D float arrays
        Weights for each point to consider into cost function
    u_model: list of 3D float arrays
        Zonal wind field from model
    v_model: list of 3D float arrays
        Meridional wind field from model
    w_model: list of 3D float arrays
        Vertical wind field from model
    coeff: float
        Weight of background constraint to total cost function

    Returns
    -------
    y: float array
        value of gradient of background cost function
    """
    the_shape = u.shape
    u_grad = np.zeros(the_shape)
    v_grad = np.zeros(the_shape)
    w_grad = np.zeros(the_shape)
    for i in range(len(u_model)):
        u_grad += coeff*2*(u-u_model[i])*weights[i]
        v_grad += coeff*2*(v-v_model[i])*weights[i]

    y = np.stack([u_grad, v_grad, w_grad], axis=0)
    return y.flatten()
