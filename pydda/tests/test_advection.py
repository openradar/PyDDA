import pyart
import numpy as np
import pydda

def test_advection():
    Grid1 = pyart.testing.make_normal_storm(10, (0, 0))
    Grid2 = pyart.testing.make_normal_storm(10, (20, 0))
    Grid1.radar_time = {'long_name': 'Time in seconds of the volume start for each radar',
                        'units': 'seconds since 2005-12-24T12:30:08Z',
                        'calendar': 'gregorian',
                        'data': np.ma.array([3e-6])}
    Grid2.radar_time = {'long_name': 'Time in seconds of the volume start for each radar',
                        'units': 'seconds since 2005-12-24T12:40:08Z',
                        'calendar': 'gregorian',
                        'data': np.ma.array([3e-6])}
    Grid = pydda.retrieval.get_storm_motion(Grid1, Grid2)
    np.testing.assert_almost_equal(Grid["storm_motion_u"]["data"],
                                   0.033 * np.zeros_like(Grid["reflectivity"]["data"]))
    np.testing.assert_almost_equal(Grid["storm_motion_v"]["data"],
                                   np.zeros_like(Grid["reflectivity"]["data"]))