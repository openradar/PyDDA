import jax.numpy as jnp
import numpy as np

from jax import grad, jit, vmap
from jax import random
from scipy.optimize import minimize
from copy import deepcopy
from cftime import num2date

def translation_cost_function(U, V, R0, R1, t0, t1, beta):
    """
    Calculates the cost function of the frozen turbulence hypothesis and the
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

    Returns
    -------
    The value of the translational cost function
    """
    alpha1 = jnp.where(R0.mask == False, 1, 0)
    alpha2 = jnp.where(R1.mask == False, 1, 0)
    R0j = jnp.array(R1)
    R1j = jnp.array(R2)
    dR0dx = jnp.gradient(R0j, axis=2)
    dR0dy = jnp.gradient(R0j, axis=1)
    dR1dx = jnp.gradient(R1j, axis=2)
    dR1dy = jnp.gradient(R1j, axis=1)
    dRdT = (R1j - R0j) / (t1 - t0)
    gradh_U2 = jnp.square(jnp.gradient(U, axis=1)) + jnp.square(jnp.gradient(U, axis=2))
    gradh_V2 = jnp.square(jnp.gradient(V, axis=1)) + jnp.square(jnp.gradient(V, axis=2))
    cost1 = jnp.sum(alpha1 * (dRdT + U * dR0dx + V * dR0dy) ** 2 + beta * gradh_U2 + beta * gradh_V2) + \
            jnp.sum(alpha2 * (dRdT + U * dR1dx + V * dR1dy) ** 2 + beta * gradh_U2 + beta * gradh_V2)
    return cost1 / (t1 - t0)


def _cost_wrapper(winds, field_shape, R0, R1, beta):
    U = np.reshape(winds[0:len(winds) / 2], field_shape)
    V = np.reshape(winds[(len(winds) / 2):], field_shape)
    return translation_cost_function(U, V, R0, R1, beta)


def get_storm_motion(Grid1, Grid2, beta=0.5, scalar_field='reflectivity'):
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
        Smoothing factor
    scalar_field:

    Returns
    -------
    Grid: pyart.core.Grid
        A Grid object with the U and V components of the storm motion
    """
    R0 = Grid1.fields[scalar_field].values
    R1 = Grid2.fields[scalar_field].values
    t0 = 0
    t1 = (num2date(Grid1.radar_time) - num2date(Grid0.radar_time)).seconds
    n_points = np.prod(R0.shape)
    winds = np.zeros(2 * n_points)
    fun = lambda x: cost_wrapper(x, R0.shape, R0, R1, t0, t1, beta)
    bounds = [(-x, x) for x in 100. * np.ones(winds.shape)]
    result = minimize(fun, winds, jac=grad(fun), tol=1e-5, method='L-BFGS-B')
    U = np.reshape(result.x[0:n_points], R0.shape)
    V = np.reshape(result.x[n_points:], R0.shape)
    U = np.ma.masked_where(np.logical_or(R0.mask, R1.mask), U)
    V = np.ma.masked_where(np.logical_or(R0.mask, R1.mask), V)
    StormU = {'data': U, 'units': 'm/s', 'long_name': 'U component of storm motion',
              'standard_name': 'storm_motion_u'}
    StormV = {'data': V, 'units': 'm/s', 'long_name': 'V component of storm motion',
              'standard_name': 'storm_motion_v'}
    Grid = deepcopy(Grid1)
    Grid.add_field('storm_motion_u', StormU, replace_existing=True)
    Grid.add_field('storm_motion_v', StormV, replace_existing=True)

    return Grid
