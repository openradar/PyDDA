#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:20:29 2018

@author: rjackson
"""
import math
import matplotlib.pyplot as plt
import pyart
import numpy as np

from .. import retrieval


def plot_horiz_xsection_barbs(Grids, background_field='reflectivity', level=1,
                              cmap='pyart_LangRainbow12', 
                              vmin=None, vmax=None, vel_contours=None,
                              u_field='u', v_field='v', w_field='w',
                              show_lobes=True, title_flag=True, 
                              axes_labels_flag=True, colorbar_flag=True,
                              bg_grid_no=0, barb_spacing_x_km=10.0,
                              barb_spacing_y_km=10.0):
    
    grid_bg = Grids[bg_grid_no].fields[background_field]['data']
    
    if(vmin == None):
        vmin = grid_bg.min()
        
    if(vmax == None):
        vmax = grid_bg.max()
        
    grid_h = Grids[0].point_altitude['data']
    grid_x = Grids[0].point_x['data']
    grid_y = Grids[0].point_y['data']
    dx = np.diff(grid_x, axis=2)[0,0,0]
    dy = np.diff(grid_y, axis=1)[0,0,0]
    u = Grids[0].fields[u_field]['data']
    v = Grids[0].fields[v_field]['data']
    w = Grids[0].fields[w_field]['data']
    
    plt.pcolormesh(grid_x[level,::,::], grid_y[level,::,::], grid_bg[level,:,:],
                   cmap=cmap)
    barb_density_x = int((dx/1000.0)*barb_spacing_x_km)
    barb_density_y = int((dx/1000.0)*barb_spacing_y_km)
    plt.barbs(grid_x[level,::barb_density_y,::barb_density_x], 
              grid_y[level,::barb_density_y,::barb_density_x], 
              u[level,::barb_density_y,::barb_density_x], 
              v[level,::barb_density_y,::barb_density_x])
    
    if(colorbar_flag == True):
        cp = Grids[bg_grid_no].fields[background_field]['long_name']
        cp.replace(' ', '_')
        cp = cp + ' [' + Grids[bg_grid_no].fields[background_field]['units']
        cp = cp + ']'
        plt.colorbar(label=(cp))


    if(vel_contours is not None):
        w_filled = np.ma.filled(w[level,::,::], fill_value=0)
        cs = plt.contour(grid_x[level,::,::], grid_y[level,::,::],
                         w_filled, levels=vel_contours, linewidths=2, 
                         alpha=0.5)
        plt.clabel(cs)

        
    bca_min = math.radians(Grids[0].fields[u_field]['min_bca'])
    bca_max = math.radians(Grids[0].fields[u_field]['max_bca'])
    
    if(show_lobes is True):
        for i in range(len(Grids)):
            for j in range(len(Grids)):
                if (i != j):
                    bca = retrieval.get_bca(Grids[j].radar_longitude['data'],
                                            Grids[j].radar_latitude['data'],
                                            Grids[i].radar_longitude['data'], 
                                            Grids[i].radar_latitude['data'],
                                            Grids[j].point_x['data'][0],
                                            Grids[j].point_y['data'][0],
                                            Grids[j].get_projparams())
                    
                    plt.contour(grid_x[level,::,::], grid_y[level,::,::], bca, 
                                    levels=[bca_min, bca_max], color='k')
               
    
    if(axes_labels_flag == True):
        plt.xlabel(('X [' + Grids[0].point_x['units'] + ']'))
        plt.ylabel(('Y [' + Grids[0].point_y['units'] + ']'))
    
    if(title_flag == True):
        plt.title(('PyDDA retreived winds @' + str(grid_h[level,0,0]) + ' ' +
                   str(Grids[0].point_altitude['units'])))
        
    plt.xlim([grid_x.min(), grid_x.max()])
    plt.ylim([grid_y.min(), grid_y.max()])
        

def plot_xz_xsection_barbs(Grids, background_field='reflectivity', level=1, 
                           cmap='pyart_LangRainbow12', 
                           vmin=None, vmax=None, vel_contours=None,
                           u_field='u', v_field='v', w_field='w',
                           title_flag=True, axes_labels_flag=True, 
                           colorbar_flag=True,
                           bg_grid_no=0, barb_spacing_x_km=10.0,
                           barb_spacing_z_km=1.0):
    
    grid_bg = Grids[bg_grid_no].fields[background_field]['data']
    
    if(vmin == None):
        vmin = grid_bg.min()
        
    if(vmax == None):
        vmax = grid_bg.max()
        
    grid_h = Grids[0].point_altitude['data']
    grid_x = Grids[0].point_x['data']
    grid_y = Grids[0].point_y['data']
    dx = np.diff(grid_x, axis=2)[0,0,0]
    dz = np.diff(grid_y, axis=1)[0,0,0]
    u = Grids[0].fields[u_field]['data']
    v = Grids[0].fields[v_field]['data']
    w = Grids[0].fields[w_field]['data']
    
    plt.pcolormesh(grid_x[:,level,:], grid_h[:,level,:], grid_bg[:,level,:],
                   cmap=cmap)
    barb_density_x = int((dx/1000.0)*barb_spacing_x_km)
    barb_density_z = int((dz/1000.0)*barb_spacing_z_km)
    plt.barbs(grid_x[::barb_density_z,level,::barb_density_x], 
              grid_h[::barb_density_z,level,::barb_density_x], 
              u[::barb_density_z,level,::barb_density_x], 
              w[::barb_density_z,level,::barb_density_x])
    
    if(colorbar_flag == True):
        cp = Grids[bg_grid_no].fields[background_field]['long_name']
        cp.replace(' ', '_')
        cp = cp + ' [' + Grids[bg_grid_no].fields[background_field]['units'] 
        cp = cp + ']'
        plt.colorbar(label=(cp))
    
    if(vel_contours is not None):
        w_filled = np.ma.filled(w[::,level,::], fill_value=0)
        cs = plt.contour(grid_x[::,level,::], grid_h[::,level,::],
                         w_filled, levels=vel_contours, linewidths=2, 
                         alpha=0.5)
        plt.clabel(cs)
  
    
    if(axes_labels_flag == True):
        plt.xlabel(('X [' + Grids[0].point_x['units'] + ']'))
        plt.ylabel(('Z [' + Grids[0].point_z['units'] + ']'))
    
    if(title_flag == True):
        if(grid_y[0,level,0] > 0):
            plt.title(('PyDDA retreived winds @' + str(grid_y[0,level,0]) + ' ' +
                       str(Grids[0].point_altitude['units']) + ' north of ' + 
                       'origin.'))
        else:
            plt.title(('PyDDA retreived winds @' + str(-grid_y[0,level,0]) + ' ' +
                       str(Grids[0].point_altitude['units']) + ' south of ' + 
                       'origin.'))
        
    plt.xlim([grid_x.min(), grid_x.max()])
    plt.ylim([grid_h.min(), grid_h.max()])


def plot_yz_xsection_barbs(Grids, background_field='reflectivity', level=1, 
                           cmap='pyart_LangRainbow12', 
                           vmin=None, vmax=None, vel_contours=None,
                           u_field='u', v_field='v', w_field='w',
                           title_flag=True, axes_labels_flag=True, 
                           colorbar_flag=True,
                           bg_grid_no=0, barb_spacing_x_km=10.0,
                           barb_spacing_z_km=1.0):
    
    grid_bg = Grids[bg_grid_no].fields[background_field]['data']
    
    if(vmin == None):
        vmin = grid_bg.min()
        
    if(vmax == None):
        vmax = grid_bg.max()
        
    grid_h = Grids[0].point_altitude['data']
    grid_x = Grids[0].point_x['data']
    grid_y = Grids[0].point_y['data']
    dx = np.diff(grid_x, axis=2)[0,0,0]
    dz = np.diff(grid_y, axis=1)[0,0,0]
    u = Grids[0].fields[u_field]['data']
    v = Grids[0].fields[v_field]['data']
    w = Grids[0].fields[w_field]['data']
    
    plt.pcolormesh(grid_y[::,::,level], grid_h[::,::,level], grid_bg[::,::,level],
                   cmap=cmap)
    barb_density_x = int((dx/1000.0)*barb_spacing_x_km)
    barb_density_z = int((dz/1000.0)*barb_spacing_z_km)
    plt.barbs(grid_y[::barb_density_z,::barb_density_x, level], 
              grid_h[::barb_density_z,::barb_density_x, level], 
              v[::barb_density_z,::barb_density_x, level], 
              w[::barb_density_z,::barb_density_x, level])
    
    if(colorbar_flag == True):
        cp = Grids[bg_grid_no].fields[background_field]['long_name']
        cp.replace(' ', '_')
        cp = cp + ' [' + Grids[bg_grid_no].fields[background_field]['units'] 
        cp = cp + ']'
        plt.colorbar(label=(cp))
    
    if(vel_contours is not None):
        w_filled = np.ma.filled(w[::,::,level], fill_value=0)
        cs = plt.contour(grid_y[::, ::, level], grid_h[::, ::, level],
                         w_filled, levels=vel_contours, linewidths=2, 
                         alpha=0.5)
        plt.clabel(cs)

    
    if(axes_labels_flag == True):
        plt.xlabel(('Y [' + Grids[0].point_y['units'] + ']'))
        plt.ylabel(('Z [' + Grids[0].point_z['units'] + ']'))
    
    if(title_flag == True):
        if(grid_x[0,0, level] > 0):
            plt.title(('PyDDA retreived winds @' + str(grid_x[0,0,level]) +
                       ' ' + str(Grids[0].point_altitude['units']) +
                       ' east of origin.'))
        else:
            plt.title(('PyDDA retreived winds @' + str(-grid_x[0,0,level]) +
                       ' ' + str(Grids[0].point_altitude['units']) +
                       ' west of origin.'))
        
    plt.xlim([grid_y.min(), grid_y.max()])
    plt.ylim([grid_h.min(), grid_h.max()])
        
                