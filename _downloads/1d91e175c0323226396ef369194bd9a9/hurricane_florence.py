"""
Example on integrating radar and HRRR data
------------------------------------------

This is an example of how to retrieve winds in Hurricane Florence.
In this example, we use data from 2 NEXRAD radars as well as from
the HRRR to retrieve the winds.

Author: Robert C. Jackson
"""

import urllib
import pyart
import pydda
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

hrrr_url = ('https://pando-rgw01.chpc.utah.edu/hrrr/prs/20180914/' +
            'hrrr.t06z.wrfprsf00.grib2')
urllib.request.urlretrieve(hrrr_url, 'test.grib2')

grid_mhx = pyart.io.read_grid(pydda.tests.MHX_GRID)
grid_ltx = pyart.io.read_grid(pydda.tests.LTX_GRID)

grid_mhx = pydda.constraints.add_hrrr_constraint_to_grid(grid_mhx,
                                                         'test.grib2')
u_init, v_init, w_init = pydda.initialization.make_constant_wind_field(
    grid_mhx, (0.0, 0.0, 0.0))
out_grids = pydda.retrieval.get_dd_wind_field(
    [grid_mhx, grid_ltx], u_init, v_init, w_init, Co=0.0, Cm=0.0, Cmod=1e-3,
    mask_outside_opt=True, vel_name='corrected_velocity',
    model_fields=["hrrr"])

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax = pydda.vis.plot_horiz_xsection_barbs_map(
    out_grids, ax=ax, bg_grid_no=-1, level=1, barb_spacing_x_km=20.0,
    barb_spacing_y_km=20.0)

plt.title(out_grids[0].time['units'][13:] + ' winds at 0.5 km')
