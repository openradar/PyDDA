import numpy as np
import pyart

def make_test_divergence_field(Grid, wind_vel, z_ground, z_top, radius,
                               back_u, back_v, x_center, y_center):
    """
    This function makes a test field with wind convergence at the surface
    and divergence aloft.
    This function makes a useful test for the mass continuity equation.
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
    back_u: float
        The u component of the wind outside of the area of convergence.
    back_v: float
        The v component of the wind outside of the area of convergence.
    x_center: float
        The X-coordinate of the center of the area of convergence in the
        Grid's coordinates.
    y_center: float
        The Y-coordinate of the center of the area of convergence in the
        Grid's coordinates.
    Returns
    -------
    u_init, v_init, w_init: ndarrays of floats
         Initial U, V, W field
    """

    x = Grid.point_x['data']
    y = Grid.point_y['data']
    z = Grid.point_z['data']

    theta = np.arctan2(x - x_center, y - y_center)
    phi = np.pi*((z - z_ground)/(z_top - z_ground))
    r = np.sqrt(np.square(x - x_center) + np.square(y - y_center))

    u = wind_vel*(r/radius)**2*np.cos(phi)*np.sin(theta)*np.ones(x.shape)
    v = wind_vel*(r/radius)**2*np.cos(phi)*np.cos(theta)*np.ones(x.shape)
    w = np.zeros(x.shape)
    u[r > radius] = back_u
    v[r > radius] = back_v

    u = np.ma.array(u)
    v = np.ma.array(v)
    w = np.ma.array(w)
    return u, v, w

