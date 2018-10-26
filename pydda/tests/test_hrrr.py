import pydda
import pyart
import urllib

def test_add_hrrr_field():
    hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
                'hrrr.t06z.wrfprsf00.grib2')
    urllib.request.urlretrieve(hrrr_url, 'test.grib2')
    grid_mhx = pyart.io.read_grid(pydda.tests.MHX_GRID)
    grid_ltx = pyart.io.read_grid(pydda.tests.LTX_GRID)
    grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
            'test.grib2')
    u = grid_mhx.fields["U_hrrr"]["data"]
    
    assert(u.max() > 40)
    assert(u.min() < -40)
    
