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


@pytest.mark.skipif(CFGRIB_AVAILABLE == 0, reason="Cfgrib not installed!")
def test_hrrr_constraint():
    hrrr_url = 'https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/hrrr.t06z.wrfprsf00.grib2'
    r = requests.get(hrrr_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with temp_file as f:
        f.write(r.content)
    
        
    grid_mhx = pyart.io.read_grid('../tests/data/grid_ltx.nc')
    grid_ltx = pyart.io.read_grid('../tests/data/grid_mhx.nc')

    grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
               temp_file.name)
    u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(grid_mhx, (0.0, 0.0, 0.0)) 
    out_grids = pydda.retrieval.get_dd_wind_field([grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.0, Cm=0.0,
                                              Cmod=1e-3, mask_outside_opt=True, vel_name='corrected_velocity',
                                              model_fields=["hrrr"]
                                              )
    u = out_grids[0].fields["u"]["data"]
    v = out_grids[0].fields["v"]["data"]
    # Are we actually retrieving a hurricane?
    assert u.max() > 40
    assert u.min() < 40
    assert v.max() > 40
    assert v.min() < 40

    
