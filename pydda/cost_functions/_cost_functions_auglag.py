import numpy as np
import pyart
import scipy.ndimage.filters
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from scipy.signal import savgol_filter

from ._cost_functions_tensorflow import _tf_gradient

def radial_velocity_function(winds, parameters):
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
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jvel = calculate_radial_vel_cost_function(
        parameters.vrs, parameters.azs, parameters.els,
        winds[0], winds[1], winds[2], parameters.wts, rmsVr=parameters.rmsVr,
        weights=parameters.weights, coeff=parameters.Co)

    grad = grad_radial_velocity(winds, parameters)

    return Jvel, grad


def al_mass_cont_function(winds, parameters, mult, mu):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))
    u = winds[0]
    v = winds[1]
    w = winds[2]
    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        div = calculate_mass_continuity(
            u, v, w, parameters.z,
            parameters.dx, parameters.dy, parameters.dz)

        mult = tf.reshape(mult, (parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))

        #rho = tf.math.exp(-parameters.z / 10000.0)
        #drho_dz = _tf_gradient(rho, parameters.dz, axis=0)
        #anel_coeffs = drho_dz/rho

        # COMPUTING THE GRADIENT: THIS CAN BE REPLACED WITH JAX/TENSORFLOW/WHATEVER
        #vars = {'u': u, 'v': v, 'w': w}
        #grad = tape.gradient(div, vars)
        #grad_u = grad['u']
        #grad_v = grad['v']
        #grad_w = grad['w']

        al = -tf.math.reduce_sum(mult * div) + (mu / 2.0) * tf.math.reduce_sum(div ** 2)
    
    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(al, vars)
    grad_u = grad['u']
    grad_v = grad['v']
    grad_w = grad['w']

    al_grad = tf.stack([grad_u, grad_v, grad_w], axis=0)

    return al, tf.reshape(al_grad, np.prod(al_grad.shape, ))


def al_vert_vort_function(winds, parameters, mult, mu):
    with tf.GradientTape() as tape:
        tape.watch(winds[0])
        tape.watch(winds[1])
        tape.watch(winds[2])
        vort = calculate_vertical_vorticity(winds[0], winds[1], winds[2], parameters.dx, parameters.dy, parameters.dz,
                                        parameters.Ut, parameters.Vt)

    
        al = -tf.math.reduce_sum(mult * vort) + (mu / 2.0) * tf.math.reduce_sum(vort ** 2)
    vars = {'u': winds[0], 'v': winds[1], 'w': winds[2]}
    grad = tape.gradient(al, vars)
    grad_u = grad['u']
    grad_v = grad['v']
    grad_w = grad['w']
    al_grad = tf.stack([grad_u, grad_v, grad_w], axis=0)

    return al, tf.reshape(al_grad, np.prod(al_grad.shape, ))


def smooth_cost_function(winds, parameters, grad, pos):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jsmooth = calculate_smoothness_cost(
        winds[0], winds[1], winds[2], Cx=1.0, Cy=1.0, Cz=1.0)

    if grad.size > 0:
        grad = grad_smooth_cost(winds, parameters)

    if not pos:
        Jsmooth = -1.0 * Jsmooth
        grad = -1.0 * grad

    return Jsmooth - sum(parameters.Cx + parameters.Cy + parameters.Cz)


def background_cost_function(winds, parameters, grad, pos):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jbackground = calculate_background_cost(
        winds[0], winds[1], winds[2], parameters.bg_weights,
        parameters.u_back, parameters.v_back, 1.0)
    if grad.size > 0:
        grad = grad_background_cost(winds, parameters)

    if not pos:
        Jbackground = -1.0 * Jbackground
        grad = -1.0 * grad

    return Jbackground - parameters.Cb


def vertical_vorticity_cost_function(winds, parameters, grad, pos):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jvorticity = calculate_vertical_vorticity_cost(
        winds[0], winds[1], winds[2], parameters.dx,
        parameters.dy, parameters.dz, parameters.Ut,
        parameters.Vt, coeff=1.0)
    if grad.size > 0:
        grad = grad_vertical_vorticity_cost(winds, parameters)

    if not pos:
        Jvorticity = -1.0 * Jvorticity
        grad = -1.0 * grad

    return Jvorticity - parameters.Cv


def model_cost_function(winds, parameters, grad, pos):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jmod = calculate_model_cost(
        winds[0], winds[1], winds[2],
        parameters.model_weights, parameters.u_model,
        parameters.v_model,
        parameters.w_model, coeff=1.0)

    if grad.size > 0:
        grad = grad_model_cost(winds, parameters)

    if not pos:
        Jmod = -1.0 * Jmod
        grad = -1.0 * grad

    return Jmod - parameters.Cmod


def point_cost_function(winds, parameters, grad, pos):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0], parameters.grid_shape[1],
                        parameters.grid_shape[2]))

    Jpoint = calculate_point_cost(
        winds[0], winds[1], parameters.x, parameters.y, parameters.z,
        parameters.point_list, Cp=1.0, roi=parameters.roi)
    if grad.size > 0:
        grad = grad_point_cost(winds, parameters)

    if not pos:
        Jpoint = -1.0 * Jpoint
        grad = -1.0 * grad

    return Jpoint - parameters.Cpoint


def grad_radial_velocity(winds, parameters):
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
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))
    grad = calculate_grad_radial_vel(
        parameters.vrs, parameters.els, parameters.azs,
        winds[0], winds[1], winds[2], parameters.wts, parameters.weights,
        parameters.rmsVr, coeff=parameters.Co, upper_bc=parameters.upper_bc)

    return grad


def grad_mass_cont(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_mass_continuity_gradient(
        winds[0], winds[1], winds[2], parameters.z,
        parameters.dx, parameters.dy, parameters.dz, upper_bc=parameters.upper_bc, coeff=1.0)

    return grad


def grad_smooth_cost(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_smoothness_gradient(
        winds[0], winds[1], winds[2], Cx=parameters.Cx,
        Cy=parameters.Cy, Cz=parameters.Cz, upper_bc=parameters.upper_bc)

    return grad


def grad_background_cost(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_background_gradient(
        winds[0], winds[1], winds[2], parameters.bg_weights,
        parameters.u_back, parameters.v_back, parameters.Cb,
        upper_bc=parameters.upper_bc)

    return grad


def grad_vertical_vorticity_cost(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_vertical_vorticity_gradient(
        winds[0], winds[1], winds[2], parameters.dx,
        parameters.dy, parameters.dz, parameters.Ut,
        parameters.Vt, coeff=parameters.Cv)

    return grad


def grad_model_cost(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_model_gradient(
        winds[0], winds[1], winds[2],
        parameters.model_weights, parameters.u_model, parameters.v_model,
        parameters.w_model, coeff=parameters.Cmod)

    return grad


def grad_point_cost(winds, parameters):
    winds = tf.reshape(winds,
                       (3, parameters.grid_shape[0],
                        parameters.grid_shape[1], parameters.grid_shape[2]))

    grad = calculate_point_gradient(
        winds[0], winds[1], parameters.x, parameters.y, parameters.z,
        parameters.point_list, Cp=parameters.Cpoint, roi=parameters.roi)

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
        v_ar = (tf.math.cos(els[i]) * tf.math.sin(azs[i]) * u +
                tf.math.cos(els[i]) * tf.math.cos(azs[i]) * v +
                tf.math.sin(els[i]) * (w - tf.math.abs(wts[i])))
        the_weight = weights[i]
        J_o += lambda_o * tf.reduce_sum(tf.math.square(vrs[i] - v_ar) * the_weight)

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
    upper_bc: bool
        Set to true to impose w=0 at top of domain.

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

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_radial_vel_cost_function(vrs, azs, els,
                                                  u, v, w, wts, rmsVr, weights, coeff)
    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = grad['w']

    # Impermeability condition
    p_z1 = tf.concat(
        [tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float64), p_z1[1:, :, :]], axis=0)
    if (upper_bc is True):
        p_z1 = tf.concat(
            [p_z1[:-1, :, :],
             tf.zeros((1, u.shape[1], u.shape[2]))], axis=0)
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


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
    dudx = _tf_gradient(u, dx, axis=2)
    dudy = _tf_gradient(u, dy, axis=1)
    dudz = _tf_gradient(u, dz, axis=0)
    dvdx = _tf_gradient(v, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dvdz = _tf_gradient(v, dz, axis=0)
    dwdx = _tf_gradient(w, dx, axis=2)
    dwdy = _tf_gradient(w, dy, axis=1)
    dwdz = _tf_gradient(w, dz, axis=0)

    x_term = Cx * (_tf_gradient(dudx, dx, axis=2) ** 2 + _tf_gradient(dvdx, dx, axis=1) ** 2 +
                   _tf_gradient(dwdx, dx, axis=2) ** 2)
    y_term = Cy * (_tf_gradient(dudy, dy, axis=2) ** 2 + _tf_gradient(dvdy, dy, axis=1) ** 2 +
                   _tf_gradient(dwdy, dy, axis=2) ** 2)
    z_term = Cz * (_tf_gradient(dudz, dz, axis=2) ** 2 + _tf_gradient(dvdz, dz, axis=1) ** 2 +
                   _tf_gradient(dwdz, dz, axis=2) ** 2)
    return tf.math.reduce_sum(x_term + y_term + z_term)


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
    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_smoothness_cost(u, v, w,
                                         dx, dy, dz, Cx=Cx, Cy=Cy, Cz=Cz)

    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = grad['w']

    # Impermeability condition
    p_z1 = tf.concat([tf.zeros((1, u.shape[1], u.shape[2])),
                      p_z1[1:, :, :]], axis=0)
    if (upper_bc is True):
        p_z1 = tf.concat(
            [p_z1[:-1, :, :], tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float64)], axis=0)
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


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
        xp = tf.ones_like(x) * the_point["x"]
        yp = tf.ones_like(y) * the_point["y"]
        zp = tf.ones_like(z) * the_point["z"]
        up = tf.ones_like(u) * the_point["u"]
        vp = tf.ones_like(v) * the_point["v"]

        the_box = tf.where(tf.math.logical_and(
            tf.math.logical_and(tf.math.abs(x - xp) < roi, tf.math.abs(y - yp) < roi),
            tf.math.abs(z - zp) < roi), 1., 0.)
        J += tf.math.reduce_sum(((u - up) ** 2 + (v - vp) ** 2) * the_box)

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

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        loss = calculate_point_cost(u, v, x, y, z, point_list)

    vars = {'u': u, 'v': v}
    grad = tape.gradient(loss, vars)
    gradJ_u = grad['u']
    gradJ_v = grad['v']
    gradJ_w = tf.zeros_like(gradJ_u)
    gradJ = tf.stack([gradJ_u, gradJ_v, gradJ_w], axis=0)
    gradJ = tf.reshape(gradJ, (3 * np.prod(u.shape),))
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
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dwdz = _tf_gradient(w, dz, axis=0)

    if (anel == 1):
        rho = tf.math.exp(-z / 10000.0)
        drho_dz = _tf_gradient(rho, dz, axis=0)
        anel_term = w / rho * drho_dz
    else:
        anel_term = tf.zeros(w.shape, dtype=tf.float64)

    div = dudx + dvdy + dwdz + anel_term

    return div


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

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_mass_continuity(u, v, w, z, dx, dy, dz, coeff)

    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = grad['w']

    # Impermeability condition
    p_z1 = tf.concat(
        [tf.zeros((1, u.shape[1], u.shape[2])),
         p_z1[1:, :, :]], axis=0)
    if (upper_bc is True):
        p_z1 = tf.concat([p_z1[:-1, :, :],
                          tf.zeros((1, u.shape[1], u.shape[2]), dtype=tf.float64)], axis=0)
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))



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
    rho = np.exp(-grid_z / 10000.0)
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

    fallspeed = A * np.power(10, refl * B) * np.power(1.2 / rho, 0.4)
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
        cost += (Cb * tf.reduce_sum(np.square(u[i] - u_back[i]) * (weights[i]) +
                             tf.math.square(v[i] - v_back[i]) * (weights[i])))
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
    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        loss = calculate_background_cost(u, v, weights, u_back, v_back, Cb=Cb)

    vars = {'u': u, 'v': v}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = tf.zeros_like(p_x1)

    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


def calculate_vertical_vorticity(u, v, w, dx, dy, dz, Ut, Vt):
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
    dvdz = _tf_gradient(v, dz, axis=0)
    dudz = _tf_gradient(u, dz, axis=0)
    dvdx = _tf_gradient(v, dx, axis=2)
    dwdy = _tf_gradient(w, dy, axis=1)
    dwdx = _tf_gradient(w, dx, axis=2)
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=2)
    dudy = _tf_gradient(u, dy, axis=1)
    zeta = dvdx - dudy
    dzeta_dx = _tf_gradient(zeta, dx, axis=2)
    dzeta_dy = _tf_gradient(zeta, dy, axis=1)
    dzeta_dz = _tf_gradient(zeta, dz, axis=0)
    jv_array = ((u - Ut) * dzeta_dx + (v - Vt) * dzeta_dy +
                w * dzeta_dz + (dvdz * dwdx - dudz * dwdy) +
                zeta * (dudx + dvdy))
    return jv_array


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
    dvdz = _tf_gradient(v, dz, axis=0)
    dudz = _tf_gradient(u, dz, axis=0)
    dwdz = _tf_gradient(w, dz, axis=0)
    dvdx = _tf_gradient(v, dx, axis=2)
    dwdy = _tf_gradient(w, dy, axis=1)
    dwdx = _tf_gradient(w, dx, axis=2)
    dudx = _tf_gradient(u, dx, axis=2)
    dvdy = _tf_gradient(v, dy, axis=1)
    dudy = _tf_gradient(u, dy, axis=1)

    zeta = dvdx - dudy
    dzeta_dx = _tf_gradient(zeta, dx, axis=2)
    dzeta_dy = _tf_gradient(zeta, dy, axis=1)
    dzeta_dz = _tf_gradient(zeta, dz, axis=0)
    jv_array = ((u - Ut) * dzeta_dx + (v - Vt) * dzeta_dy +
                w * dzeta_dz + (dvdz * dwdx - dudz * dwdy) +
                zeta * (dudx + dvdy))
    return tf.math.reduce_sum(coeff * tf.math.square(jv_array))


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

    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_vertical_vorticity_cost(u, v, w, dx, dy, dz, Ut, Vt, coeff)
    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = grad['w']

    # Impermeability condition
    p_z1 = tf.concat([tf.zeros((1, u.shape[1], u.shape[2])),
                      p_z1[1:, :, :]], axis=0)
    if (upper_bc is True):
        p_z1 = tf.concat([p_z1[:-1, :, :],
                          tf.zeros((1, u.shape[1], u.shape[2]))], axis=0)
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))


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
        cost += (coeff * tf.math.reduce_sum(
            tf.math.square(u - u_model[i]) * weights[i] +
            tf.math.square(v - v_model[i]) * weights[i]))
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
    
    with tf.GradientTape() as tape:
        tape.watch(u)
        tape.watch(v)
        tape.watch(w)
        loss = calculate_model_cost(
            u, v, w, weights, u_model, v_model, w_model, coeff=coeff)

    vars = {'u': u, 'v': v, 'w': w}
    grad = tape.gradient(loss, vars)
    p_x1 = grad['u']
    p_y1 = grad['v']
    p_z1 = tf.zeros(p_x1.shape)

    # Impermeability condition
    p_z1 = tf.concat([tf.zeros((1, u.shape[1], u.shape[2])),
                      p_z1[1:, :, :]], axis=0)
    if (upper_bc is True):
        p_z1 = tf.concat([p_z1[:-1, :, :],
                          tf.zeros((1, u.shape[1], u.shape[2]))], axis=0)
    y = tf.stack((p_x1, p_y1, p_z1), axis=0)
    return tf.reshape(y, (3 * np.prod(u.shape),))
