import pyart
import pydda
import numpy as np

berr_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//berr_200601200050.nc")
cpol_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//cpol_200601200050.nc")

u_init, v_init, w_init = pydda.retrieval.make_constant_wind_field(cpol_grid, (0,0,0), vel_field='VT')
u, v, w = pydda.retrieval.get_dd_wind_field(berr_grid, cpol_grid, u_init, v_init, w_init, vel_name='VT', refl_field='DT')
