import jax.numpy as jnp
import jax.ops as jops
import numpy as np

from jax import vjp
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)
from scipy.optimize import minimize
from copy import deepcopy
from cftime import num2pydate


def source_term(U, V, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1):
    """
    Calculates the source term for U and V
    Parameters
    ----------
    U: Nz x Ny x Nx float array
        U component of provisional wind field.
    V: Nz x Ny x Nx float array
        V component of provisional wind field.
    R0 float numpy array
        The scalar field at time t0.
    R1: float numpy array
        The scalar field at time t1.
    t0: float
        The time in seconds of time t0
    t1: float
        The time in seconds of time t1
    beta: float
        Smoothing coefficient
    dx0, 1: float
        The x grid spacing in m of grid 0 (1).
    dy0, 1: float
        The y grid spacing in m of grid 0 (1).
    Returns
    -------
    The value of the translational cost function
    """
    if isinstance(R0, np.ma.MaskedArray):
        R0 = np.ma.filled(R0, np.nan)
    if isinstance(R1, np.ma.MaskedArray):
        R1 = np.ma.filled(R1, np.nan)
    alpha0 = jnp.where(jnp.isfinite(R0), 1, 0)
    alpha1 = jnp.where(jnp.isfinite(R1), 1, 0)
    R0j = jnp.where(jnp.isnan(R0), 0, R0)
    R1j = jnp.where(jnp.isnan(R1), 0, R1)
    dR0dx = jnp.gradient(R0j, dx0, axis=2)
    dR0dy = jnp.gradient(R0j, dy0, axis=1)
    dR1dx = jnp.gradient(R1j, dx1, axis=2)
    dR1dy = jnp.gradient(R1j, dy1, axis=1)
    dRdT = (R1j - R0j) / (t1 - t0)
    T = t1 - t0
    dUdx = jnp.gradient(U, dx0, axis=2)
    dUdy = jnp.gradient(U, dy0, axis=1)
    dVdx = jnp.gradient(V, dx0, axis=2)
    dVdy = jnp.gradient(V, dy0, axis=1)

    jops.index_update(dUdx, jops.index[:, 0, :],  0.)
    jops.index_update(dUdx, jops.index[:, -1, :], 0.)
    jops.index_update(dVdx, jops.index[:, 0, :], 0.)
    jops.index_update(dVdx, jops.index[:, -1, :], 0.)
    jops.index_update(dUdy, jops.index[:, :, 0], 0.)
    jops.index_update(dUdy, jops.index[:, :, -1], 0.)
    jops.index_update(dVdy, jops.index[:, :, 0], 0.)
    jops.index_update(dVdy, jops.index[:, :, -1], 0.)
    jops.index_update(alpha0, jops.index[:, :, 0], 0.)
    jops.index_update(alpha0, jops.index[:, :, -1], 0.)
    jops.index_update(alpha0, jops.index[:, 0, :], 0.)
    jops.index_update(alpha0, jops.index[:, -1, :], 0.)
    jops.index_update(alpha1, jops.index[:, :, 0], 0.)
    jops.index_update(alpha1, jops.index[:, :, -1], 0.)
    jops.index_update(alpha1, jops.index[:, 0, :], 0.)
    jops.index_update(alpha1, jops.index[:, -1, :], 0.)

    # Integrate PDE dot solution guess over domain
    source = 1 / (beta * T) * (_trap(t0, t1, alpha0 * dRdT * dR0dx, alpha1 * dRdT * dR1dx) + \
            U * _trap(t0, t1, alpha0 * dR0dx ** 2, alpha1 * dR1dx ** 2) + \
            V * _trap(t0, t1, alpha0 * dR0dx * dR0dy, alpha1 * dR1dx * dR1dy))

    source2 = 1 / (beta * T) * (_trap(t0, t1, alpha0 * dRdT * dR0dy, alpha1 * dRdT * dR1dy) + \
            U * _trap(t0, t1, alpha0 * dR0dx * dR0dy, alpha1 * dR1dx * dR1dy) + \
            V * _trap(t0, t1, alpha0 * dR0dy ** 2, alpha1 * dR1dy ** 2))

    return source, source2

def _trap(a, b, fa, fb):
    return (b - a) * (fa + fb) / 2.


# def _cost_wrapper(winds, field_shape, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1):
#     n_points = int(np.prod(field_shape))
#     U = jnp.reshape(winds[0:n_points], field_shape)
#     V = jnp.reshape(winds[n_points:], field_shape)
#     return translation_cost_function(U, V, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1)
#
#
# def _gradJ(winds, field_shape, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1):
#     n_points = int(np.prod(field_shape))
#     U = jnp.reshape(winds[0:n_points], field_shape)
#     V = jnp.reshape(winds[n_points:], field_shape)
#     primals, fun_vjp = vjp(translation_cost_function, U, V, R0, R1,
#                            t0, t1, beta, dx0, dy0, dx1, dy1)
#     p_x1, p_y1, _, _, _, _, _, _, _, _, _ = fun_vjp(1.0)
#     y = jnp.stack([p_x1, p_y1])
#     return np.array(y.flatten().copy(), dtype='float64')


def get_storm_motion(Grid1, Grid2, beta=None, scalar_field='reflectivity',
                     verbose=False, utol=0.001, vtol=0.001):
    """
    Calculates the translational component (mean storm motion) between two
    scans deduced from the motion of a scalar field.

    Parameters
    ----------
    Grid1: pyart.core.Grid
        A Grid object containing the radar data at time t0
    Grid2: pyart.core.Grid
        A Grid object containing the radar data at time t1
    beta: float
        Smoothing factor (None will automatically calculate)
    scalar_field: str
        Name of scalar field
    verbose: bool
        If true, display cost and gradient at each iteration
    utol: float
        Iterate until change in U is less than utol.
    vtol: float
        Iterate until change in V is less than vtol.

    Returns
    -------
    Grid: pyart.core.Grid
        A Grid object with the U and V components of the storm motion
    """
    R0 = Grid1.fields[scalar_field]["data"]
    R1 = Grid2.fields[scalar_field]["data"]
    if beta is None:
        beta = 0.1 * R0.max() ** 2
    t0 = 0.
    t1 = (num2pydate(Grid2.radar_time["data"], Grid2.radar_time["units"]) -
          num2pydate(Grid1.radar_time["data"], Grid1.radar_time["units"]))
    t1 = t1[0].seconds
    n_points = np.prod(R0.shape)
    U = np.ones_like(R0)
    V = np.ones_like(R0)
    dx0 = np.max(np.diff(Grid1.x["data"]))
    dy0 = np.max(np.diff(Grid1.y["data"]))
    dx1 = np.max(np.diff(Grid2.x["data"]))
    dy1 = np.max(np.diff(Grid2.y["data"]))

    dU = np.inf
    dV = np.inf
    # Zero-normal boundary condition
    U[:, 0, :] = U[:, 1, :]
    U[:, :, 0] = U[:, :, 1]
    V[:, 0, :] = V[:, 1, :]
    V[:, :, 0] = V[:, :, 1]
    U[:, -1, :] = U[:, -2, :]
    U[:, :, -1] = U[:, :, -2]
    V[:, -1, :] = V[:, -2, :]
    V[:, :, -1] = V[:, :, -2]
    i = 0
    while(dU > utol and dV > vtol):
        Umax = np.linalg.norm(U.flatten(), 2)
        Vmax = np.linalg.norm(V.flatten(), 2)
        source, source2 = source_term(U, V, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1)
        Uold = U
        Vold = V
        # Zero-normal boundary condition
        omega = 0.5
        U[:, 0, :] = U[:, 1, :]
        U[:, :, 0] = U[:, :, 1]
        V[:, 0, :] = V[:, 1, :]
        V[:, :, 0] = V[:, :, 1]
        U[:, -1, :] = U[:, -2, :]
        U[:, :, -1] = U[:, :, -2]
        V[:, -1, :] = V[:, -2, :]
        V[:, :, -1] = V[:, :, -2]

        U[:, 1:-1, 1:-1] = (U[:, :-2, 1:-1] + U[:, 2:, 1:-1] +
                              U[:, 1:-1, 2:] + U[:, 1:-1, 2:]
                              - dx0 * dy0 * source[:, 1:-1, 1:-1]) / 4.
        V[:, 1:-1, 1:-1] = (V[:, :-2, 1:-1] + V[:, 2:, 1:-1] +
                            V[:, 1:-1, 2:] + V[:, 1:-1, 2:]
                              - dx0 * dy0 * source2[:, 1:-1, 1:-1]) / 4.

        dU = np.abs(np.linalg.norm(U.flatten(), 2) - Umax)
        dV = np.abs(np.linalg.norm(V.flatten(), 2) - Vmax)
        if verbose and i % 10 == 0:
            print("Iteration %d: U max = %3.2f, Vmax = %3.2f" % (i, U.max(), V.max()))
        i += 1
        #div = jnp.gradient(U, dx0, axis=2) + jnp.gradient(V, dy0, axis=1)

    # options = {}
    # options["maxiter"] = 10000.
    # options["disp"] = verbose
    # options["gtol"] = 1e-7
    # bounds = [(-x, x) for x in 100 * np.ones(winds.shape)]
    # result = minimize(_cost_wrapper, winds, args=(R0.shape, R0, R1,
    #                                               t0, t1, beta, dx0, dy0, dx1, dy1),
    #                   jac=_gradJ, method='L-BFGS-B', bounds=bounds, options=options)
    #U = np.reshape(result.x[0:n_points], R0.shape)
    #V = np.reshape(result.x[n_points:], R0.shape)
    U = np.ma.masked_where(~np.logical_or(np.isfinite(R0), np.isfinite(R1)), U)
    V = np.ma.masked_where(~np.logical_or(np.isfinite(R0), np.isfinite(R1)), V)
    StormU = {'data': U, 'units': 'm/s','long_name': 'U component of storm motion',
              'standard_name': 'storm_motion_u', 'min_bca': -np.inf, 'max_bca': np.inf}
    StormV = {'data': V, 'units': 'm/s', 'long_name': 'V component of storm motion',
              'standard_name': 'storm_motion_v', 'min_bca': -np.inf, 'max_bca': np.inf}
    Grid = deepcopy(Grid1)
    Grid.add_field('storm_motion_u', StormU, replace_existing=True)
    Grid.add_field('storm_motion_v', StormV, replace_existing=True)

    return Grid
