import pyart
import numpy as np
import pydda

def make_normal_storm(sigma, mu):
    """
    Make a sample Grid with a gaussian storm target.
    """
    test_grid = pyart.testing.make_empty_grid(
        [1, 101, 101], [(1, 1), (-50000., 50000.), (-50000., 50000.)])
    x = test_grid.x['data']
    y = test_grid.y['data']
    z = test_grid.z['data']
    zg, yg, xg = np.meshgrid(z, y, x, indexing='ij')
    r = np.sqrt((xg - mu[0])**2 + (yg - mu[1])**2)
    r = r/1e3
    term1 = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    term2 = -1.0 * (r**2 / (2.0 * sigma**2))
    data = term1 * np.exp(term2)
    rdic = {
        'data': data,
        'long_name': 'reflectivity',
        'units': 'dBz'}
    test_grid.fields.update({'reflectivity': rdic})
    return test_grid

def test_advection():
    Grid1 = make_normal_storm(1., (0., 0.))
    Grid2 = make_normal_storm(1., (40000., 40000.))
    Grid1.fields["reflectivity"]["data"] = Grid1.fields["reflectivity"]["data"]
    Grid2.fields["reflectivity"]["data"] = Grid2.fields["reflectivity"]["data"]
    Grid1.radar_time = {'long_name': 'Time in seconds of the volume start for each radar',
                        'units': 'seconds since 2005-12-24T12:30:08Z',
                        'calendar': 'gregorian',
                        'data': np.ma.array([3e-6])}
    Grid2.radar_time = {'long_name': 'Time in seconds of the volume start for each radar',
                        'units': 'seconds since 2005-12-24T12:50:08Z',
                        'calendar': 'gregorian',
                        'data': np.ma.array([3e-6])}
    Grid = pydda.retrieval.get_storm_motion(Grid1, Grid2, verbose=True)
    np.testing.assert_almost_equal(Grid.fields["storm_motion_u"]["data"].max(), 2000/60.)
    np.testing.assert_almost_equal(Grid.fields["storm_motion_v"]["data"].max(), 2000/60.)