""" Nosetests for continuous integration """
import pydda
import pyart
import numpy as np


def test_calculate_rad_velocity_cost():
    """ Test with a zero velocity field radar """
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000)))

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field('zero_field', {'data': fdata3, '_FillValue': -9999.0})
    vel_field = 'zero_field'
    pydda.retrieval.angles.add_azimuth_as_field(Grid, dz_name='zero_field')
    pydda.retrieval.angles.add_elevation_as_field(Grid, dz_name='zero_field')
    vrs = [np.ma.array(Grid.fields[vel_field]['data'])]
    azs = [Grid.fields['AZ']['data']]
    els = [Grid.fields['EL']['data']]
    u = np.zeros((20, 20, 20))
    v = np.zeros((20, 20, 20))
    w = np.zeros((20, 20, 20))
    rmsVr = 1.0
    wts = [np.ma.zeros((20, 20, 20))]
    weights = [np.ones((20, 20, 20))]
    cost = pydda.cost_functions.calculate_radial_vel_cost_function(
        vrs, azs, els, u, v, w, wts, rmsVr, weights)
    grad = pydda.cost_functions.calculate_grad_radial_vel(
        vrs, azs, els, u, v, w, wts, weights, rmsVr)

    assert cost == 0
    assert np.all(grad == 0)


def test_calculate_fall_speed():
    """ Check to see if fall speeds are realistic """
    ref_field = 10*np.ones((10, 100, 100))
    grid_shape = (10, 100, 100)
    grid_limits = ((0, 10000), (-100000, 100000), (-100000, 100000))
    grid = pyart.testing.make_empty_grid(grid_shape, grid_limits)
    field_dic = {'data': ref_field,
                 'long_name': 'reflectivity',
                 'units': 'dBZ'}
    grid.fields = {'reflectivity': field_dic}
    fall_speed = pydda.cost_functions.calculate_fall_speed(
         grid, refl_field='reflectivity')
    assert fall_speed.shape == (10, 100, 100)
    assert fall_speed[1, 1, 1] < -3


def test_calculate_mass_continuity():
    """ In a constant wind field, div * V = 0, so we should get zero for mass
    continuity cost function and gradient"""
    u = 10*np.ones((10, 10, 10))
    v = 10*np.ones((10, 10, 10))
    w = 0*np.ones((10, 10, 10))
    dx = 100.0
    dy = 100.0
    dz = 100.0
    z = np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.calculate_mass_continuity(
        u, v, w, z, dx, dy, dz)
    cost_grad = pydda.cost_functions.calculate_mass_continuity_gradient(
        u, v, w, z, dx, dy, dz)

    assert cost == 0
    assert np.all(cost_grad == 0)

    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10-i] = i
    cost = pydda.cost_functions.calculate_mass_continuity(
        u, v, w, z, dx, dy, dz)
    assert cost > 0


def test_calculate_smoothness_cost():
    """ The Laplacian of a constant field is zero """
    u = 10*np.ones((10, 10, 10))
    v = 10*np.ones((10, 10, 10))
    w = 0*np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0
    z = np.arange(0, 1000.0, 100)

    cost = pydda.cost_functions.calculate_smoothness_cost(
        u, v, w)
    cost_grad = pydda.cost_functions.calculate_smoothness_gradient(
        u, v, w)

    assert cost == 0
    assert np.all(cost_grad == 0)

    """ Now, put in a discontinuity """
    u[:, :, 5] = -10
    cost = pydda.cost_functions.calculate_smoothness_cost(
        u, v, w)
    assert cost > 0


def test_background_cost():
    """ Zero cost when background matches wind field, nonzero otherwise """
    u = 10*np.ones((10, 10, 10))
    v = 10*np.ones((10, 10, 10))
    w = 0*np.ones((10, 10, 10))
    weights = np.ones((10, 10, 10))

    u_back = 10*np.ones(10)
    v_back = 10*np.ones(10)
    cost = pydda.cost_functions.calculate_background_cost(
        u, v, w, weights, u_back, v_back)
    grad = pydda.cost_functions.calculate_background_gradient(
        u, v, w, weights, u_back, v_back)

    assert cost == 0
    assert np.all(grad == 0)

    u_back = 4*np.ones(10)
    v_back = 4*np.ones(10)

    cost = pydda.cost_functions.calculate_background_cost(
        u, v, w, weights, u_back, v_back)
    grad = pydda.cost_functions.calculate_background_gradient(
        u, v, w, weights, u_back, v_back)

    assert cost > 0
    assert np.any(grad > 0)


def test_vert_vorticity():
    u = 10*np.ones((10, 10, 10))
    v = 10*np.ones((10, 10, 10))
    w = 0*np.ones((10, 10, 10))

    dx = 100.0
    dy = 100.0
    dz = 100.0

    z = np.arange(0, 1000.0, 100)
    cost = pydda.cost_functions.calculate_vertical_vorticity_cost(
        u, v, w, dx, dy, dz, 10, 10)
    cost_grad = pydda.cost_functions.calculate_vertical_vorticity_gradient(
        u, v, w, dx, dy, dz, 10, 10)

    assert cost == 0
    assert np.all(cost_grad == 0)

    """ Put in a convergent wind field (u decreases from 10 to 0 m/s) """
    v = np.zeros((10, 10, 10))
    for i in np.arange(10, 0, -1):
        u[:, :, 10-i] = i
        v[:, :, 10-i] = i
    cost = pydda.cost_functions.calculate_vertical_vorticity_cost(
        u, v, w, z, dx, dy, dz, 10, 10)
    assert cost > 0


def test_point_cost():
    u = 1 * np.ones((10, 10, 10))
    v = 1 * np.ones((10, 10, 10))
    w = 0 * np.ones((10, 10, 10))

    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    z = np.linspace(-10, 10, 10)
    x, y, z = np.meshgrid(x, y, z)

    my_point1 = {'x': 0, 'y': 0, 'z': 0, 'u': 2., 'v': 2., 'w': 0.}
    cost = pydda.cost_functions.calculate_point_cost(u, v, x, y, z, [my_point1], roi=2.0)
    grad = pydda.cost_functions.calculate_point_gradient(u, v, x, y, z, [my_point1], roi=2.0)

    assert cost > 0
    assert np.all(grad <= 0)

    my_point1 = {'x': 0, 'y': 0, 'z': 0, 'u': -2., 'v': -2., 'w': 0.}
    my_point2 = {'x': 3, 'y': 3, 'z': 0, 'u': 2., 'v': 2., 'w': 0.}

    cost = pydda.cost_functions.calculate_point_cost(u, v, x, y, z, [my_point1], roi=2.0)
    grad = pydda.cost_functions.calculate_point_gradient(u, v, x, y, z, [my_point1], roi=2.0)
    assert cost > 0
    assert np.all(grad >= 0)

    cost = pydda.cost_functions.calculate_point_cost(u, v, x, y, z, [my_point1, my_point2], roi=2.0)
    grad = pydda.cost_functions.calculate_point_gradient(u, v, x, y, z, [my_point1, my_point2], roi=2.0)
    assert cost > 0
    assert ~np.all(grad >= 0)

    my_point1 = {'x': 0, 'y': 0, 'z': 0, 'u': 1., 'v': 1., 'w': 0.}
    my_point2 = {'x': 3, 'y': 3, 'z': 0, 'u': 1., 'v': 1., 'w': 0.}
    cost = pydda.cost_functions.calculate_point_cost(u, v, x, y, z, [my_point1, my_point2], roi=2.0)
    grad = pydda.cost_functions.calculate_point_gradient(u, v, x, y, z, [my_point1, my_point2], roi=2.0)
    assert cost == 0
    assert np.all(grad == 0)


def test_model_cost():
    u = 10*np.ones((10, 10, 10))
    v = 10*np.ones((10, 10, 10))
    w = 0*np.ones((10, 10, 10))
    weights = np.ones((10, 10, 10))

    cost = pydda.cost_functions.calculate_model_cost(
        u, v, w, weights, u, v, w)
    cost_grad = pydda.cost_functions.calculate_model_gradient(
        u, v, w, weights, u, v, w)

    # If model == observations, then cost is zero
    assert cost == 0
    assert np.all(cost_grad == 0)

    # If model is further from obs, cost is greater
    cost1 = pydda.cost_functions.calculate_model_cost(
        u, v, w, weights, u - 1, v, w)
    cost2 = pydda.cost_functions.calculate_model_cost(
        u, v, w, weights, u - 1, v - 1, w)
    assert cost2 > cost1
