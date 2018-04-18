#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:20:29 2018

@author: rjackson
"""
import math
import matplotlib.pyplot as plt
import pyart

from .. import retrieval


def plot_horiz_xsection_barbs(Grids, background_field='reflectivity', 
                              level=3, cmap='pyart_LangRainbow12', 
                              vmin=None, vmax=None, vel_contours=None,
                              u_field='u', v_field='v', w_field='w',
                              show_lobes=True, title_flag=True, 
                              axes_labels_flag=True, colorbar_flag=True,
                              bg_grid_no=0):
    
    grid_bg = Grids[bg_grid_no].fields[background_field]['data']
    
    if(vmin == None):
        vmin = grid_bg.min()
        
    if(vmax == None):
        vmax = grid_bg.max()
        
    grid_h = Grids[0].point_altitude['data']
    grid_x = Grids[0].point_x['data']
    grid_y = Grids[0].point_y['data']
    u = Grids[0].fields[u_field]['data']
    v = Grids[0].fields[v_field]['data']
    w = Grids[0].fields[w_field]['data']
    
    plt.pcolormesh(grid_x[level,::,::], grid_y[level,::,::], grid_bg[7,:,:],
                   cmap=cmap)
    barb_density_x = int(grid_x.shape[2]/10)
    barb_density_y = int(grid_y.shape[2]/10)
    plt.barbs(grid_x[level,::barb_density_y,::barb_density_x], 
              grid_y[level,::barb_density_y,::barb_density_x], 
              u[level,::barb_density_y,::barb_density_x], 
              v[level,::barb_density_y,::barb_density_x])
    
    if(colorbar_flag == True):
        cp = Grids[bg_grid_no].fields[background_field]['standard_name']
        cp.replace('_', ' ')
        cp = cp + '[' + Grids[bg_grid_no].fields[background_field]['units'] 
        cp = cp + ']'
        plt.colorbar(label=(cp))
    
    if(vel_contours is not None):
        cs = plt.contour(grid_x[level,::,::], grid_y[level,::,::],
                         w[level,::,::], levels=vel_contours, linewidth=16, 
                         alpha=0.5)
        plt.clabel(cs)
        
    bca_min = math.radians(Grids[0].fields[u_field]['min_bca'])
    bca_max = math.radians(Grids[0].fields[u_field]['max_bca'])
    
    if(show_lobes is True):
        for i in range(len(Grids)):
            for j in range(len(Grids)):
                if (i != j):
                    bca = retrieval.get_bca(Grids[i].radar_longitude['data'],
                                            Grids[i].radar_latitude['data'],
                                            Grids[j].radar_longitude['data'], 
                                            Grids[j].radar_latitude['data'],
                                            Grids[i].point_x['data'][0],
                                            Grids[i].point_y['data'][0],
                                            Grids[i].get_projparams())

                    plt.contour(grid_x[level,::,::], grid_x[level,::,::], bca, 
                                    levels=[math.radians(bca_min),
                                            math.radians(bca_max)], color='k')
               
    
    if(axes_labels_flag == True):
        plt.xlabel(('X [' + Grids[0].point_x['units'] + ']'))
        plt.ylabel(('Y [' + Grids[0].point_x['units'] + ']'))
    
    if(title_flag == True):
        plt.title(('PyDDA retreived winds @' + str(grid_h[level,0,0]) + ' ' +
                   str(Grids[0].point_altitude['units'])))
        
    plt.xlim([grid_x.min(), grid_x.max()])
    plt.ylim([grid_y.min(), grid_y.max()])
        
        