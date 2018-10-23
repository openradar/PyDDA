import urllib
import pyart
import pytest
import os

hrrr_url = 'https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/hrrr.t06z.wrfprsf00.grib2'   
    urllib.request.urlretrieve(hrrr_url)

ltx_grid = pyart.io.read_grid('../tests/data/grid_ltx.nc')
mhx_grid = pyart.io.read_grid('../tests/data/grid_mhx.nc')

grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
            'hrrr.t06z.wrfprsf00.grib2')

out_grids = pydda.retrieval.get_dd_wind_field([grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.0, Cm=0.0,
                                              Cmod=1e-3, mask_outside_opt=True, vel_name='corrected_velocity',
                                              model_fields=["hrrr"]
                                              )

fig = plt.figure(figsize=(15,10)) 
ax = plt.axes(projection=ccrs.PlateCarree())
ax = pydda.vis.plot_horiz_xsection_barbs_map(out_grids, ax=ax, bg_grid_no=-1, level=1, barb_spacing_x_km=20.0,
                                             barb_spacing_y_km=20.0)

plt.title(out_grids[0].time['units'][13:] + ' winds at 0.5 km')


