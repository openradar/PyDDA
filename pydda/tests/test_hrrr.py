import requests
import tempfile
import pyart
import pydda
import pytest
import os

try:
    import cfgrib
    CFGRIB_AVAILABLE = 1
except:
    CFGRIB_AVAILABLE = 0


def test_add_hrrr_data():
    hrrr_url = 'https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/hrrr.t06z.wrfprsf00.grib2'
    r = requests.get(hrrr_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with temp_file as f:
        f.write(r.content)
    grid_mhx = pyart.io.read_grid(pydda.tests.MHX_GRID)
    grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
               temp_file.name)
 
    u = grid_mhx.fields["U_hrrr"]["data"]
    v = grid_mhx.fields["V_hrrr"]["data"]
    # Are we actually retrieving a hurricane?
    assert u.max() > 40
    assert u.min() < 40
    assert v.max() > 40
    assert v.min() < 40 

    
