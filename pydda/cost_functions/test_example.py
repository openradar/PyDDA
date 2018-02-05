import pyart
import pydda
from matplotlib import pyplot as plt
import numpy as np

berr_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//berr_200601200050.nc")
cpol_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//cpol_200601200050.nc")

sounding = pyart.io.read_arm_sonde(
    "C://Users//rjackson//Documents//data//soundings//twpsondewnpnC3.b1.20060119.231600.custom.cdf")
print(berr_grid.projection)
print(cpol_grid.get_projparams())
u_back = sounding[1].u_wind
v_back = sounding[1].v_wind
z_back = sounding[1].height
#u_init, v_init, w_init = pydda.retrieval.make_constant_wind_field(cpol_grid, wind=(0.0,0.0,0.0), vel_field='VT')
u_init, v_init, w_init = pydda.retrieval.make_wind_field_from_profile(cpol_grid, sounding, vel_field='VT')
berr_grid.fields['DT']['data'] = cpol_grid.fields['DT']['data']
# Step 1 - do iterations with just data
u, v, w = pydda.retrieval.get_dd_wind_field(berr_grid, cpol_grid, u_init, v_init, w_init,
                                            u_back=u_back, v_back=v_back, z_back=z_back,
                                            C1=100.0, C2=1000.0, C4=50.0, C5=50.0, C6=1.0,
                                            C7 = 1.0, C8=0.0, vel_name='VT', refl_field='DT',
                                            frz=5000.0, filt_iterations=1)

plt.interactive(False)
cpol_z = cpol_grid.fields['DT']['data']
cpol_h = cpol_grid.point_altitude['data']
cpol_x = cpol_grid.point_x['data']
cpol_y = cpol_grid.point_y['data']
bca = pydda.retrieval.get_bca(berr_grid.radar_longitude['data'],
                  berr_grid.radar_latitude['data'],
                  cpol_grid.radar_longitude['data'],
                  cpol_grid.radar_latitude['data'],
                  berr_grid.point_x['data'][0],
                  berr_grid.point_y['data'][0],
                  cpol_grid.get_projparams())

plt.contourf(cpol_x[7,::,::], cpol_y[7,::,::],
             cpol_z[7,::,::], cmap=pyart.graph.cm.NWSRef, levels=np.arange(0,60,1))
plt.colorbar(label='Z [dBZ]')
plt.quiver(cpol_x[7,::,::], cpol_y[7,::,::], u[7,::,::], v[7,::,::])
cs = plt.contour(cpol_x[7,::,::], cpol_y[7,::,::],
                 w[7,::,::], levels=np.arange(1,10,1), linewidth=4, alpha=0.5)
plt.contour(cpol_x[7,::,::], cpol_y[7,::,::], bca, levels=[np.pi/6, 5*np.pi/6], color='k')
plt.clabel(cs)
plt.colorbar(label='W [m/s]')
plt.xlabel('X [km]', fontsize=20)
plt.ylabel('Y [km]', fontsize=20)

plt.show()

plt.interactive(False)
cpol_z = cpol_grid.fields['DT']['data']


plt.contourf(cpol_x[::,40,::], cpol_h[::,40,::],
             cpol_z[::,40,::], cmap=pyart.graph.cm.NWSRef, levels=np.arange(0,60,1))
plt.colorbar(label='Z [dBZ]')
plt.quiver(cpol_x[::,40,::], cpol_h[::,40,::], u[::,40,::], w[::,40,::])
cs = plt.contour(cpol_x[::,40,::], cpol_h[::,40,::],
                 w[::,40,::], levels=np.arange(1,10,2), linewidth=8, alpha=0.5)
plt.clabel(cs)
plt.colorbar(label='W [m/s]')
plt.xlabel('X [km]', fontsize=20)
plt.ylabel('Z [m]', fontsize=20)
plt.show()