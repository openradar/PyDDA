import numpy as np
import pyart

from netCDF4 import Dataset
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata


def make_constant_wind_field(Grid, wind=(0.0,0.0,0.0), vel_field=None):
    """
    This function makes a constant wind field given a wind vector.

    This function is useful for specifying the intialization arrays
    for get_dd_wind_field. 

    Parameters
    ==========

    Grid: Py-ART Grid object
        This is the Py-ART Grid containing the coordinates for the analysis 
        grid.
    wind: 3-tuple of floats
        The 3-tuple specifying the (u,v,w) of the wind field.
    vel_field: String
        The name of the velocity field. None will automatically
        try to detect this field.

    Returns
    =======

    u: 3D float array 
        Returns a 3D float array containing the u component of the wind field.
        The shape will be the same shape as the fields in Grid.
    v: 3D float array 
        Returns a 3D float array containing the v component of the wind field.
        The shape will be the same shape as the fields in Grid.
    w: 3D float array 
        Returns a 3D float array containing the u component of the wind field.
        The shape will be the same shape as the fields in Grid.
    """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name('corrected_velocity')
    analysis_grid_shape = Grid.fields[vel_field]['data'].shape

    u = wind[0]*np.ones(analysis_grid_shape)
    v = wind[1]*np.ones(analysis_grid_shape)
    w = wind[2]*np.ones(analysis_grid_shape)
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    return u, v, w


def make_wind_field_from_profile(Grid, profile, vel_field=None):
    """
    This function makes a 3D wind field from a sounding. 

    This function is useful for using sounding data as an initialization
    for get_dd_wind_field.

    Parameters
    ==========
    Grid: Py-ART Grid object
        This is the Py-ART Grid containing the coordinates for the analysis 
        grid.
    profile: PyART HorizontalWindProfile
        This is the HorizontalWindProfile of the sounding
    wind: 3-tuple of floats
        The 3-tuple specifying the (u,v,w) of the wind field.
    vel_field: String
        The name of the velocity field in Grid. None will automatically
        try to detect this field.

    Returns
    =======

    u: 3D float array 
        Returns a 3D float array containing the u component of the wind field.
        The shape will be the same shape as the fields in Grid.
    v: 3D float array 
        Returns a 3D float array containing the v component of the wind field.
        The shape will be the same shape as the fields in Grid.
    w: 3D float array 
        Returns a 3D float array containing the u component of the wind field.
        The shape will be the same shape as the fields in Grid.
        """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name('corrected_velocity')
    analysis_grid_shape = Grid.fields[vel_field]['data'].shape
    u = np.ones(analysis_grid_shape)
    v = np.ones(analysis_grid_shape)
    w = np.zeros(analysis_grid_shape)
    u_back = profile.u_wind
    v_back = profile.v_wind
    z_back = profile.height
    u_interp = interp1d(
        z_back, u_back, bounds_error=False, fill_value='extrapolate')
    v_interp = interp1d(
        z_back, v_back, bounds_error=False, fill_value='extrapolate')
    u_back2 = u_interp(np.asarray(Grid.z['data']))
    v_back2 = v_interp(np.asarray(Grid.z['data']))
    for i in range(analysis_grid_shape[0]):
        u[i] = u_back2[i]
        v[i] = v_back2[i]
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    return u, v, w


""" Makes a test wind field that converges at center near ground and
    Diverges aloft at center """
def make_test_divergence_field(Grid, wind_vel, z_ground, z_top, radius,
                               back_u, back_v, x_center, y_center):
    """
    This function makes a test field with wind convergence at the surface
    and divergence aloft.

    This function makes a useful test for the mass continuity equation.

    Parameters
    ----------
    Grid: Py-ART Grid object
        This is the Py-ART Grid containing the coordinates for the analysis 
        grid.
    wind_vel: float
        The maximum wind velocity.
    z_ground: float 
        The bottom height where the maximum convergence occurs
    z_top: float
        The height where the maximum divergence occurrs
    back_u: float
        The u component of the wind outside of the area of convergence.
    back_v: float
        The v component of the wind outside of the area of convergence.
    x_center: float
        The X-coordinate of the center of the area of convergence in the 
        Grid's coordinates.
    y_center: float
        The Y-coordinate of the center of the area of convergence in the 
        Grid's coordinates.
    
 
    Returns
    -------
    u_init, v_init, w_init: ndarrays of floats
         Initial U, V, W field
    """
    
    x = Grid.point_x['data']
    y = Grid.point_y['data']
    z = Grid.point_z['data']
    
    
    theta = np.arctan2(x-x_center, y-y_center)
    phi = np.pi*((z-z_ground)/(z_top-z_ground))
    r = np.sqrt(np.square(x-x_center) + np.square(y-y_center))
    
    u = wind_vel*(r/radius)**2*np.cos(phi)*np.sin(theta)*np.ones(x.shape)
    v = wind_vel*(r/radius)**2*np.cos(phi)*np.cos(theta)*np.ones(x.shape)
    w = np.zeros(x.shape)
    u[r > radius] = back_u
    v[r > radius] = back_v
    
    u = np.ma.array(u)
    v = np.ma.array(v)
    w = np.ma.array(w)
    return u,v,w


def make_background_from_wrf(Grid, file_path, wrf_time, 
                             radar_loc, vel_field=None):
    """
    This function makes an initalization field based off of the u and w
    from a WRF run. Only u and v are used from the WRF file.
    
    Parameters
    ----------
    Grid: Py-ART Grid object
        This is the Py-ART Grid containing the coordinates for the 
        analysis grid.
    file_path: str
        This is the path to the WRF grid
    wrf_time: datetime
        The timestep to derive the intialization field from.
    radar_loc: tuple
        The (X, Y) location of the radar in the WRF grid. The output
        coordinate system will be centered around this location
        and given the same grid specification that is specified
        in Grid.
    vel_field: str, or None
        This string contains the name of the velocity field in the 
        Grid. None will try to automatically detect this value.
        
    Returns
    -------
    u: 3D ndarray 
        The initialization u field.
    v: 3D ndarray
        The initialization v field.
    w: 3D ndarray
        The initialization w field.
        
    """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name('corrected_velocity')
        
    analysis_grid_shape = Grid.fields[vel_field]['data'].shape
    u = np.ones(analysis_grid_shape)
    v = np.ones(analysis_grid_shape)
    w = np.zeros(analysis_grid_shape)
    
    # Load WRF grid
    wrf_cdf = Dataset(file_path, mode='r')
    W_wrf = wrf_cdf.variables['W'][:]
    V_wrf = wrf_cdf.variables['V'][:]
    U_wrf = wrf_cdf.variables['U'][:]
    PH_wrf = wrf_cdf.variables['PH'][:]
    PHB_wrf = wrf_cdf.variables['PHB'][:]
    alt_wrf = (PH_wrf+PHB_wrf)/9.81
    
    new_grid_x = Grid.point_x['data']
    new_grid_y = Grid.point_y['data']
    new_grid_z = Grid.point_z['data']
    
    # Find timestep from datetime
    time_wrf = wrf_cdf.variables['Times']
    ntimes = time_wrf.shape[0]
    dts_wrf = []
    for i in range(ntimes):
        x = ''.join([x.decode() for x in time_wrf[i]])
        dts_wrf.append(datetime.strptime(x, '%Y-%m-%d_%H:%M:%S'))
    
    dts_wrf = np.array(dts_wrf)
    timestep = np.where(dts_wrf == wrf_time)
    if(len(timestep[0]) == 0):
        raise ValueError(("Time " + str(wrf_time) + " not found in WRF file!"))
        
    x_len = wrf_cdf.__getattribute__('WEST-EAST_GRID_DIMENSION')
    y_len = wrf_cdf.__getattribute__('SOUTH-NORTH_GRID_DIMENSION')
    dx = wrf_cdf.DX
    dy = wrf_cdf.DY
    x = np.arange(0, x_len)*dx-radar_loc[0]*1e3
    y = np.arange(0, y_len)*dy-radar_loc[1]*1e3
    z = np.mean(alt_wrf[timestep[0],:,:,:], axis=(0,2,3))
    x, y, z = np.meshgrid(x,y,z)
    z = np.squeeze(alt_wrf[timestep[0],:,:,:])

    z_stag = (z[1:,:,:]+z[:-1,:,:])/2.0
    x_stag = (x[:,:,1:]+x[:,:,:-1])/2.0
    y_stag = (y[:,1:,:]+y[:,:-1,:])/2.0
    
    
    W_wrf = np.squeeze(W_wrf[timestep[0],:,:,:])
    V_wrf = np.squeeze(V_wrf[timestep[0],:,:,:])
    U_wrf = np.squeeze(U_wrf[timestep[0],:,:,:])
    
    w = griddata((z_stag, y, x), W_wrf, 
                        (new_grid_z,new_grid_y,new_grid_x),
                        fill_value=0.)
    v = griddata((z, y_stag, x), V_wrf, 
                        (new_grid_z,new_grid_y,new_grid_x),
                        fill_value=0.)
    u = griddata((z, y, x_stag), U_wrf, 
                        (new_grid_z,new_grid_y,new_grid_x),
                        fill_value=0.)
    return u, v, w
    
    
