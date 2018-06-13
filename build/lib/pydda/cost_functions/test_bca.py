import pydda
import pyart
import numpy as np

from matplotlib import pyplot as plt


berr_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//berr_200601200050.nc")
cpol_grid = pyart.io.read_grid("C://Users//rjackson//PyDDA//pydda//cpol_200601200050.nc")

cpol_z = cpol_grid.fields['DT']['data']
cpol_h = cpol_grid.point_altitude['data']
cpol_x = cpol_grid.point_longitude['data']
cpol_y = cpol_grid.point_latitude['data']
bca = pydda.retrieval.get_bca(cpol_grid.radar_longitude['data'],
                  cpol_grid.radar_latitude['data'],
                  berr_grid.radar_longitude['data'],
                  berr_grid.radar_latitude['data'],
                  cpol_grid.point_x['data'][0],
                  cpol_grid.point_y['data'][0],
                  cpol_grid.get_projparams())
plt.interactive(False)
plt.contour(cpol_x[7,::,::], cpol_y[7,::,::], bca, levels=[np.pi/6, 5*np.pi/6], color='k')
plt.text(cpol_grid.radar_longitude['data'], cpol_grid.radar_latitude['data'], 'CPOL')
plt.text(berr_grid.radar_longitude['data'], berr_grid.radar_latitude['data'], 'Berrima')
plt.show()