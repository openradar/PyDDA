import jax.numpy as jnp
import jax.ops as jops
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
from copy import deepcopy
from cftime import num2pydate
from jax.experimental.ode import odeint
from scipy.ndimage import map_coordinates
from scipy.integrate import cumtrapz


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
                     verbose=False, utol=0.5, vtol=0.5, num_times=2, levels=None):
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
    num_times: int
        Number of time periods for integration

    Returns
    -------
    Grid: pyart.core.Grid
        A Grid object with the U and V components of the storm motion
    """
    R0 = Grid1.fields[scalar_field]["data"].copy()
    R1 = Grid2.fields[scalar_field]["data"].copy()
    if isinstance(R0, np.ma.MaskedArray):
        R0min = np.ma.min(R0)
        R0 = R0.filled(R0min)
    if isinstance(R1, np.ma.MaskedArray):
        R1min = np.ma.min(R1)
        R1 = R1.filled(R1min)
    alpha0 = np.where(np.isfinite(R0), 1, 0)
    alpha1 = np.where(np.isfinite(R1), 1, 0)

    if beta is None:
        beta = 0.1 * np.nanmax(R0) ** 2
    t0 = 0.
    t1 = (num2pydate(Grid2.radar_time["data"], Grid2.radar_time["units"]) -
          num2pydate(Grid1.radar_time["data"], Grid1.radar_time["units"]))
    t1 = t1[0].seconds
    n_points = np.prod(R0.shape)
    U = np.zeros_like(R0)
    V = np.zeros_like(R0)
    dx0 = np.max(np.diff(Grid1.x["data"]))
    dy0 = np.max(np.diff(Grid1.y["data"]))
    dx1 = np.max(np.diff(Grid2.x["data"]))
    dy1 = np.max(np.diff(Grid2.y["data"]))
    x = np.arange(0, R0.shape[2], 1.) * dx0
    y = np.arange(0, R0.shape[1], 1.) * dy0
    if levels is not None:
        levels = np.array(levels)
    else:
        levels = np.arange(0, U.shape[0])


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
    Uold = U.copy()
    Vold = V.copy()
    while(dU > utol and dV > vtol):
        # Then do correction for R
        div = np.gradient(U, dx0, axis=2) + np.gradient(V, dy0, axis=1)
        dt = (t1 - t0)/num_times
        for j in levels:
            if alpha0[j].sum() == 0 and alpha1[j].sum() == 0:
                continue
            U_2d = np.array(U[j])
            V_2d = np.array(V[j])
            R1_2d = np.array(R1[j])
            R0_2d = np.array(R0[j])

            R0f = np.zeros((U.shape[1], U.shape[2], num_times))
            R0b = np.zeros((U.shape[1], U.shape[2], num_times))
            divt = np.zeros((U.shape[1], U.shape[2], num_times, num_times))
            for k in range(num_times):
                xtempf = np.ones(U_2d.shape) * x[None, :]
                xtempb = np.ones(U_2d.shape) * x[None, :]
                ytempf = np.ones(U_2d.shape) * y[:, None]
                ytempb = np.ones(U_2d.shape) * y[:, None]

                varout = map_coordinates(U_2d, [(ytempf)/dy0, (xtempf)/dx0], order=1, mode='nearest')
                k1x = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempf)/dy0, (xtempf)/dx0], order=1, mode='nearest')
                k1y = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempf + k1y)/dy0, (xtempf + k1x)/dx0], order=1,
                                         mode='nearest')
                k2x = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempf + k1y)/dy0, (xtempf + k1x)/dx0], order=1,
                                         mode='nearest')
                k2y = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempf + k2y)/dy0, (xtempf + k2x)/dx0], order=1,
                                         mode='nearest')

                k3x = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempf + k2y)/dy0, (xtempf + k2x)/dx0], order=1,
                                         mode='nearest')
                k3y = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempf + k3y)/dy0, (xtempf + k3x)/dx0], order=1,
                                         mode='nearest')
                k4x = dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempf + k3y)/dy0, (xtempf + k3x)/dx0], order=1,
                                         mode='nearest')
                k4y = dt * varout.reshape(U_2d.shape)

                xtempf += (k1x + 2 * k2x + 2 * k3x + k4x) / 6.
                ytempf += (k1y + 2 * k2y + 2 * k3y + k4y) / 6.

                ind = np.arange(1, num_times - k)
                varout = map_coordinates(div[j], [ytempf/dy0, xtempf/dx0], order=1,
                                         mode='nearest')
                divt[:, :, ind, ind + k] = np.array([np.copy(varout.reshape(U_2d.shape)).T] * (num_times - 1 - k)).T

                varout = map_coordinates(R1_2d, [ytempf/dy0, xtempf/dx0], order=1,
                                         mode='nearest')
                R0f[:, :, num_times - 1 - k] = np.copy(varout.reshape((R0f.shape[0], R0f.shape[1])))

                varout = map_coordinates(U_2d, [ytempb/dy0, xtempb/dx0], order=1, mode='nearest')
                k1x = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [ytempb/dy0, xtempb/dx0], order=1, mode='nearest')
                k1y = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempb + k1y/2.)/dy0, (xtempb + k1x/2.)/dx0], order=1,
                                         mode='nearest')
                k2x = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempb + k1y/2.)/dy0, (xtempb + k1x/2.)/dx0], order=1,
                                         mode='nearest')
                k2y = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempb + k2y/2.)/dy0, (xtempb + k2x/2.)/dx0], order=1,
                                         mode='nearest')
                k3x = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempb + k2y/2.)/dy0, (xtempb + k2x/2.)/dx0], order=1,
                                         mode='nearest')
                k3y = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(U_2d, [(ytempb + k3y)/dy0, (xtempb + k3x)/dx0], order=1,
                                         mode='nearest')
                k4x = -dt * varout.reshape(U_2d.shape)

                varout = map_coordinates(V_2d, [(ytempb + k3y)/dy0, (xtempb + k3x)/dx0], order=1,
                                         mode='nearest')
                k4y = -dt * varout.reshape(U_2d.shape)

                xtempb += (k1x + 2 * k2x + 2 * k3x + k4x) / 6.
                ytempb += (k1y + 2 * k2y + 2 * k3y + k4y) / 6.

                ind = np.arange(1, num_times - k)
                varout = map_coordinates(div[j], [ytempb.ravel()/dy0, xtempb.ravel()/dx0], order=1, mode='nearest')
                divt[:, :, ind, ind - k] = np.array([np.copy(varout.reshape(U_2d.shape)).T] * (num_times - 1 - k)).T

                varout = map_coordinates(R0_2d, [ytempb.ravel()/dy0, xtempb.ravel()/dx0], order=1, mode='nearest')
                R0b[:, :, k] = varout.reshape((R0f.shape[0], R0f.shape[1]))

            gral1 = np.zeros(divt.shape)
            gral1[:, :, :, 1:] = cumtrapz(np.asarray(divt), dx=dt, axis=3)
            gral2 = np.zeros(divt.shape)
            gral2[:, :, :, 1:] = cumtrapz(np.exp(-gral1), dx=dt, axis=3)

            ratio = gral2[:, :, np.arange(num_times), np.arange(num_times)]/gral2[:, :, np.arange(num_times), -1]
            where_alpha = np.logical_or(alpha1[j] == 0, alpha0[j] == 0)
            R0b = R0b + np.asarray(ratio) * (R0f - R0b)
            R0[j] = np.where(where_alpha, R0[j], R0b[:, :, 0])
            R1[j] = np.where(where_alpha, R1[j], R0b[:, :, -1])

        source, source2 = source_term(U, V, R0, R1, t0, t1, beta, dx0, dy0, dx1, dy1)
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
        if verbose and i % 5 == 0 and i > 0:
            dU = np.nanmax(np.abs(U[levels] - Uold[levels]))
            dV = np.nanmax(np.abs(V[levels] - Vold[levels]))
            Uold = U.copy()
            Vold = V.copy()
            print("Iteration %d: U max = %3.2f, Vmax = %3.2f dU = %3.2f dV = %3.2f R0max = %3.2f" %
                  (i, np.nanmax(U[levels]), np.nanmax(V[levels]), dU, dV, np.nanmax(R0[levels])))
        i += 1

    U = np.ma.masked_where(~np.logical_or(np.isfinite(R0), np.isfinite(R1)), U)
    V = np.ma.masked_where(~np.logical_or(np.isfinite(R0), np.isfinite(R1)), V)
    StormU = {'data': U, 'units': 'm/s', 'long_name': 'U component of storm motion',
              'standard_name': 'storm_motion_u', 'min_bca': -np.inf, 'max_bca': np.inf}
    StormV = {'data': V, 'units': 'm/s', 'long_name': 'V component of storm motion',
              'standard_name': 'storm_motion_v', 'min_bca': -np.inf, 'max_bca': np.inf}
    Grid = deepcopy(Grid1)
    Grid.add_field('storm_motion_u', StormU, replace_existing=True)
    Grid.add_field('storm_motion_v', StormV, replace_existing=True)

    return Grid
