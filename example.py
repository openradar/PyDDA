import pyart
import pydda
from matplotlib import pyplot as plt
import numpy as np

berr_grid = pyart.io.read_grid("berr_Darwin_hires.nc")
cpol_grid = pyart.io.read_grid("cpol_Darwin_hires.nc")

sounding = pyart.io.read_arm_sonde(
    "/home/rjackson/data/soundings/twpsondewnpnC3.b1.20060119.231600.custom.cdf")
print(berr_grid.projection)
print(cpol_grid.get_projparams())
u_back = sounding[1].u_wind
v_back = sounding[1].v_wind
z_back = sounding[1].height
#u_init, v_init, w_init = pydda.retrieval.make_constant_wind_field(cpol_grid, wind=(0.0,0.0,0.0), vel_field='VT')
u_init, v_init, w_init = pydda.retrieval.make_wind_field_from_profile(cpol_grid, sounding, vel_field='VT')
#u_init, v_init, w_init = pydda.retrieval.make_test_divergence_field(
#        cpol_grid, 30, 9.0, 15e3, 20e3, 5, 0, -20e3, 0)
# Test mass continuity by putting convergence at surface and divergence aloft

berr_grid.fields['DT']['data'] = cpol_grid.fields['DT']['data']
# Step 1 - do iterations with just data
Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init,
                                            v_init, w_init,u_back=u_back,
                                            v_back=v_back, z_back=z_back,
                                            Co=100.0, Cm=1500.0, vel_name='VT', 
                                            refl_field='DT', frz=5000.0, 
                                            filt_iterations=0,
                                            mask_w_outside_opt=False)

plt.figure(figsize=(8,8))
pydda.vis.plot_horiz_xsection_barbs(Grids, 'DT', level=6,
                                    vel_contours=[1, 4, 10])

plt.interactive(False)
cpol_z = cpol_grid.fields['DT']['data']

lat_level=45
plt.figure(figsize=(10,10))
plt.pcolormesh(cpol_x[::,lat_level,::], cpol_h[::,lat_level,::], 
               cpol_z[::,lat_level,::], 
               cmap=pyart.graph.cm_colorblind.HomeyerRainbow)
plt.colorbar(label='Z [dBZ]')
plt.barbs(cpol_x[::barb_density_vert,lat_level,::barb_density], 
          cpol_h[::barb_density_vert,lat_level,::barb_density], 
          u['data'][::barb_density_vert,lat_level,::barb_density], 
          w['data'][::barb_density_vert,lat_level,::barb_density])
cs = plt.contour(cpol_x[::,lat_level,::], cpol_h[::,lat_level,::],
                 w['data'][::,lat_level,::], levels=np.arange(1,20,2), 
                 linewidth=16, alpha=0.5)
plt.clabel(cs)
plt.xlabel('X [km]', fontsize=20)
plt.ylabel('Z [m]', fontsize=20)
plt.show()