import urllib
import pyart
import pydda
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import pytest
import os

from distributed import Client, LocalCluster

# This test currently does not work on Travis CI due to
# memory limitations. I have included it as a test that I
# will run manually.
@pytest.mark.skip_if("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", 
                     "Skipping this test on Travis CI.")
def test_hurricane_florence():
    hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
                'hrrr.t06z.wrfprsf00.grib2')
    urllib.request.urlretrieve(hrrr_url, 'test.grib2')

    mhx_url = ('https://drive.google.com/uc?export=download&id=' + 
               '11Q9G99QzMVfIncHtd8c5nvReDn0U4f1-')
    urllib.request.urlretrieve(mhx_url, 'grid_mhx.nc')
    ltx_url = ('https://drive.google.com/uc?export=download&id=' + 
               '1BfZjAvGY16rBfeFBihYI5gZhmcO8EPRM')
    urllib.request.urlretrieve(ltx_url, 'grid_ltx.nc')

    grid_mhx = pyart.io.read_grid('grid_mhx.nc')
    grid_ltx = pyart.io.read_grid('grid_ltx.nc')

    grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
                                                            'test.grib2')

    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
        grid_mhx, (0.0, 0.0, 0.0))
    out_grids = pydda.retrieval.get_dd_wind_field_nested(
        [grid_mhx, grid_ltx], u_init, v_init, w_init, Co=1.0, Cm=1500.0, 
        Cmod=1e-3, mask_outside_opt=True, vel_name='corrected_velocity',
        model_fields=["hrrr"], client=client)

    u = out_grids[1].fields["u"]["data"]
    assert u.max() > 30
    assert u.min() < -30
    client.close()
    cluster.close()
