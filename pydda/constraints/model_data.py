import numpy as np
import pyart
import gc
import tempfile
import os

# We want cfgrib to be an optional dependency to ensure Windows compatibility
try:
    import cfgrib
    CFGRIB_AVAILABLE = True
except:
    CFGRIB_AVAILABLE = False

# We really only need the API to download the data, make ECMWF API an
# optional dependency since not everyone will have a login from the start.
try:
    from ecmwfapi import ECMWFDataServer
    ECMWF_AVAILABLE = True
except:
    ECMWF_AVAILABLE = False

from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import griddata, NearestNDInterpolator
from copy import deepcopy


def download_needed_era_data(Grid, start_date, end_date, file_name):
    """
    This function will download the ERA interim data in the region
    specified by the input Py-ART Grid within the interval specified by
    start_date and end_date. This is useful for the batch processing of
    files since the ECMWF API is limited to 20 queued requests at a time.
    This is also useful if you want to store ERA interim data for future
    use without having to download it again.

    You need to have the ECMWF API and an ECMWF account set up in order to
    use this feature. Go to this website for instructions on installing the
    API and setting up your account:

    https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets

    Parameters
    ----------
    Grid: Py-ART Grid
        The input Py-ART Grid to modify.
    start_date: datetime
        The start date of the file to download.
    end_date: datetime
        The end date of the file to download.
    file_name:
        The name of the destination file.
    """

    if ECMWF_AVAILABLE is False and file_name is None:
        raise (ModuleNotFoundError,
               ("The ECMWF API is not installed. Go to" +
                "https://confluence.ecmwf.int/display/WEBAPI" +
                "/Access+ECMWF+Public+Datasets" +
                " in order to use the auto download feature."))

    print("Download ERA Interim data...")
    # ERA interim data is in pressure coordinates
    # Retrieve u, v, w, and geopotential
    # Geopotential is needed to convert into height coordinates

    retrieve_dict = {}
    retrieve_dict['stream'] = "oper"
    retrieve_dict['levtype'] = "pl"
    retrieve_dict['param'] = "131.128/132.128/135.128/129.128"
    retrieve_dict['dataset'] = "interim"
    retrieve_dict['levelist'] = ("1/2/3/5/7/10/20/30/50/70/100/125/150/" +
                                 "175/200/225/250/300/350/400/450/500/" +
                                 "550/600/650/700/750/775/800/825/850/" +
                                 "875/900/925/950/975/1000")
    retrieve_dict['step'] = "0"
    retrieve_dict['time'] = "00/06/12/18"
    retrieve_dict['date'] = (start_date.strftime("%Y-%m-%d") + '/to/' +
                             end_date.strftime("%Y-%m-%d"))
    retrieve_dict['class'] = "ei"
    retrieve_dict['grid'] = "0.75/0.75"
    N = "%4.1f" % Grid.point_latitude["data"].max()
    S = "%4.1f" % Grid.point_latitude["data"].min()
    E = "%4.1f" % Grid.point_longitude["data"].max()
    W = "%4.1f" % Grid.point_longitude["data"].min()

    retrieve_dict['area'] = N + "/" + W + "/" + S + "/" + E
    retrieve_dict['format'] = "netcdf"
    retrieve_dict['target'] = file_name
    server = ECMWFDataServer()
    server.retrieve(retrieve_dict)


def make_constraint_from_era_interim(Grid, file_name=None, vel_field=None):
    """
    This function will read ERA Interim in NetCDF format and add it 
    to the Py-ART grid specified by Grid. PyDDA will automatically download
    the ERA Interim data that you need for the scan. It will chose the domain
    that is enclosed by the analysis grid and the time period that is closest
    to the scan. It will then do a Nearest Neighbor interpolation of the 
    ERA-Interim u and v winds to the analysis grid. 

    You need to have the ECMWF API and an ECMWF account set up in order to
    use this feature. Go to this website for instructions on installing the
    API and setting up your account:

    https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets

    Parameters
    ----------
    Grid: Py-ART Grid
        The input Py-ART Grid to modify.
    file_name: str or None
        The netCDF file containing the ERA Interim data. Setting to None will
        invoke the API in order to attempt to download the data. If the web
        API is experiencing delays, it is better to use it to download the
        file and then refer to it here. If this file does not exist
        PyDDA will use the API to create the file.
    dest_era_file: str or None
        If this is not None, then the ERA file that is saved using the
        automatic download feature will be saved
        to this file for future reading. This is useful in case the 
        web API is experiencing delays. This is not used if file_name
        is specified.
    vel_field: str or None
        The name of the velocity field in the Py-ART grid. Set to None to
        have Py-DDA attempt to automatically detect it.

    Returns
    -------
    new_Grid: Py-ART Grid
        The Py-ART Grid with the ERA Interim data added into the "u_erainterim",
        "v_erainterim", and "w_erainterim" fields.

    """
    if vel_field is None:
        vel_field = pyart.config.get_field_name('corrected_velocity')

    if ECMWF_AVAILABLE is False and file_name is None:
        raise (ModuleNotFoundError,
               ("The ECMWF API is not installed. Go to" +
                "https://confluence.ecmwf.int/display/WEBAPI" +
                "/Access+ECMWF+Public+Datasets" +
                " in order to use the auto download feature."))

    grid_time = datetime.strptime(Grid.time["units"],
                                  "seconds since %Y-%m-%dT%H:%M:%SZ")
    hour_rounded_to_nearest_6 = int(6 * round(float(grid_time.hour)/6))

    if hour_rounded_to_nearest_6 == 24:
        grid_time = grid_time + timedelta(days=1)
        grid_time = datetime(grid_time.year, grid_time.month,
                             grid_time.day, 0, grid_time.minute,
                             grid_time.second)
    else:
        grid_time = datetime(grid_time.year, grid_time.month,
                             grid_time.day,
                             hour_rounded_to_nearest_6,
                             grid_time.minute, grid_time.second)

    if file_name is not None:
        if not os.path.isfile(file_name):
            raise FileNotFoundError(file_name + " not found!")

    if file_name is None:
        print("Download ERA Interim data...")
        # ERA interim data is in pressure coordinates
        # Retrieve u, v, w, and geopotential
        # Geopotential is needed to convert into height coordinates

        retrieve_dict = {}
        retrieve_dict['stream'] = "oper"
        retrieve_dict['levtype'] = "pl"
        retrieve_dict['param'] = "131.128/132.128/135.128/129.128"
        retrieve_dict['dataset'] = "interim"
        retrieve_dict['levelist'] = ("1/2/3/5/7/10/20/30/50/70/100/125/150/" +
                                     "175/200/225/250/300/350/400/450/500/" +
                                     "550/600/650/700/750/775/800/825/850/" +
                                     "875/900/925/950/975/1000")
        retrieve_dict['step'] = "0"
        retrieve_dict['time'] = "%02d" % hour_rounded_to_nearest_6
        retrieve_dict['date'] = grid_time.strftime("%Y-%m-%d")
        retrieve_dict['class'] = "ei"
        retrieve_dict['grid'] = "0.75/0.75"
        N = "%4.1f" % Grid.point_latitude["data"].max()
        S = "%4.1f" % Grid.point_latitude["data"].min()
        E = "%4.1f" % Grid.point_longitude["data"].max()
        W = "%4.1f" % Grid.point_longitude["data"].min()

        retrieve_dict['area'] = N + "/" + W + "/" + S + "/" + E
        retrieve_dict['format'] = "netcdf"
        tfile = tempfile.NamedTemporaryFile()
        retrieve_dict['target'] = tfile.name
        file_name = tfile.name
        server = ECMWFDataServer()
        server.retrieve(retrieve_dict)
        time_step = 0

    ERA_grid = Dataset(file_name, mode='r')
    base_time = datetime.strptime(ERA_grid.variables["time"].units,
                                  "hours since %Y-%m-%d %H:%M:%S.%f")
    time_seconds = ERA_grid.variables["time"][:]
    our_time = np.array([base_time + timedelta(seconds=int(x)) for x in time_seconds])
    time_step = np.argmin(np.abs(base_time - grid_time))

    analysis_grid_shape = Grid.fields[vel_field]['data'].shape

    height_ERA = ERA_grid.variables["z"][:]
    u_ERA = ERA_grid.variables["u"][:]
    v_ERA = ERA_grid.variables["v"][:]
    w_ERA = ERA_grid.variables["w"][:]
    lon_ERA = ERA_grid.variables["longitude"][:]
    lat_ERA = ERA_grid.variables["latitude"][:]
    radar_grid_lat = Grid.point_latitude['data']
    radar_grid_lon = Grid.point_longitude['data']
    radar_grid_alt = Grid.point_z['data']
    u_flattened = u_ERA[time_step].flatten()
    v_flattened = v_ERA[time_step].flatten()
    w_flattened = w_ERA[time_step].flatten()

    the_shape = u_ERA.shape
    lon_mgrid, lat_mgrid = np.meshgrid(lon_ERA, lat_ERA)

    lon_mgrid = np.tile(lon_mgrid, (the_shape[1], 1, 1))
    lat_mgrid = np.tile(lat_mgrid, (the_shape[1], 1, 1))
    lon_flattened = lon_mgrid.flatten()
    lat_flattened = lat_mgrid.flatten()
    height_flattened = height_ERA[time_step].flatten()
    height_flattened -= Grid.radar_altitude["data"]

    u_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        u_flattened, rescale=True)
    v_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        v_flattened, rescale=True)
    w_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        w_flattened, rescale=True)
    u_new = u_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    v_new = v_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    w_new = w_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    new_grid = deepcopy(Grid)

    u_dict = {'data': u_new, 'long_name': "U from ERA-Interim", 'units': "m/s"}
    v_dict = {'data': v_new, 'long_name': "V from ERA-Interim", 'units': "m/s"}
    w_dict = {'data': w_new, 'long_name': "W from ERA-Interim", 'units': "m/s"}

    new_grid.add_field("U_erainterim", u_dict, replace_existing=True)
    new_grid.add_field("V_erainterim", v_dict, replace_existing=True)
    new_grid.add_field("W_erainterim", w_dict, replace_existing=True)

    # Free up memory

    ERA_grid.close()

    if 'tfile' in locals():
        tfile.close()

    return new_grid


def make_constraint_from_wrf(Grid, file_path, wrf_time,
                             radar_loc, vel_field=None):
    """
    This function makes an initalization field based off of the u and w
    from a WRF run in netCDF format.
    Only u and v are used from the WRF netCDF file.

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
    Grid: Py-ART Grid object
        This Py-ART Grid will contain the model u, v, and w.

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
    z = np.mean(alt_wrf[timestep[0], :, :, :], axis=(0, 2, 3))
    x, y, z = np.meshgrid(x, y, z)
    z = np.squeeze(alt_wrf[timestep[0], :, :, :])

    z_stag = (z[1:, :, :]+z[:-1, :, :])/2.0
    x_stag = (x[:, :, 1:]+x[:, :, :-1])/2.0
    y_stag = (y[:, 1:, :]+y[:, :-1, :])/2.0

    W_wrf = np.squeeze(W_wrf[timestep[0], :, :, :])
    V_wrf = np.squeeze(V_wrf[timestep[0], :, :, :])
    U_wrf = np.squeeze(U_wrf[timestep[0], :, :, :])

    w = griddata((z_stag, y, x), W_wrf,
                 (new_grid_z, new_grid_y, new_grid_x),
                 fill_value=0.)
    v = griddata((z, y_stag, x), V_wrf,
                 (new_grid_z, new_grid_y, new_grid_x),
                 fill_value=0.)
    u = griddata((z, y, x_stag), U_wrf,
                 (new_grid_z, new_grid_y, new_grid_x),
                 fill_value=0.)
    u_dict = {'data': u, 'long_name': "U from WRF", 'units': "m/s"}
    v_dict = {'data': v, 'long_name': "V from WRF", 'units': "m/s"}
    w_dict = {'data': w, 'long_name': "W from WRF", 'units': "m/s"}
    Grid.add_field("U_wrf", u_dict, replace_existing=True)
    Grid.add_field("V_wrf", v_dict, replace_existing=True)
    Grid.add_field("W_wrf", w_dict, replace_existing=True)

    return Grid


def add_hrrr_constraint_to_grid(Grid, file_path):
    """
    This function will read an HRRR GRIB2 file and create the constraining
    u, v, and w fields for the model constraint

    Parameters
    ----------
    Grid: Py-ART Grid
        The Py-ART Grid to use as the grid specification. The HRRR values
    will be interpolated to the Grid's specficiation and added as a field.
    file_path: string
        The path to the GRIB2 file to load.

    Returns
    -------
    Grid: Py-ART Grid
        This returns the Py-ART grid with the HRRR u, and v fields added.
    """

    if(CFGRIB_AVAILABLE is False):
        raise RuntimeError(("The cfgrib optional dependency needs to be " +
                            "installed for the HRRR integration feature."))

    the_grib = cfgrib.open_file(
        file_path, filter_by_keys={'typeOfLevel': 'isobaricInhPa'})

    # Load the HRRR data and tranform longitude coordinates
    grb_u = the_grib.variables['u']
    grb_v = the_grib.variables['v']
    grb_w = the_grib.variables['w']
    gh = the_grib.variables['gh']

    lat = the_grib.variables['latitude'].data[:, :]
    lon = the_grib.variables['longitude'].data[:, :]
    lon[lon > 180] = lon[lon > 180] - 360

    # Convert geometric height to geopotential height
    EARTH_MEAN_RADIUS = 6.3781e6
    gh = gh.data[:, :, :]
    height = (EARTH_MEAN_RADIUS*gh)/(EARTH_MEAN_RADIUS-gh)
    height = height - Grid.radar_altitude['data']

    radar_grid_lat = Grid.point_latitude['data']
    radar_grid_lon = Grid.point_longitude['data']
    radar_grid_alt = Grid.point_z['data']
    lat_min = radar_grid_lat.min()
    lat_max = radar_grid_lat.max()
    lon_min = radar_grid_lon.min()
    lon_max = radar_grid_lon.max()
    lon_r = np.tile(lon, (height.shape[0], 1, 1))
    lat_r = np.tile(lat, (height.shape[0], 1, 1))
    lon_flattened = lon_r.flatten()
    lat_flattened = lat_r.flatten()
    height_flattened = gh.flatten()
    the_box = np.where(np.logical_and.reduce(
       (lon_flattened >= lon_min, lat_flattened >= lat_min,
        lon_flattened <= lon_max, lat_flattened <= lat_max)))[0]

    lon_flattened = lon_flattened[the_box]
    lat_flattened = lat_flattened[the_box]
    height_flattened = height_flattened[the_box]

    u_flattened = grb_u.data[:, :, :].flatten()
    u_flattened = u_flattened[the_box]
    u_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        u_flattened, rescale=True)
    u_new = u_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    v_flattened = grb_v.data[:, :, :].flatten()
    v_flattened = v_flattened[the_box]
    v_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        v_flattened, rescale=True)
    v_new = v_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    w_flattened = grb_w.data[:, :, :].flatten()
    w_flattened = w_flattened[the_box]
    w_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened),
        w_flattened, rescale=True)
    w_new = w_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    new_grid = deepcopy(Grid)

    u_dict = {'data': u_new, 'long_name': "U from HRRR ", 'units': "m/s"}
    v_dict = {'data': v_new, 'long_name': "V from HRRR ", 'units': "m/s"}
    w_dict = {'data': w_new, 'long_name': "W from HRRR ", 'units': "m/s"}

    new_grid.add_field("U_hrrr", u_dict, replace_existing=True)
    new_grid.add_field("V_hrrr", v_dict, replace_existing=True)
    new_grid.add_field("W_hrrr", w_dict, replace_existing=True)
    
    # Free up memory
    del grb_u, grb_v, grb_w, lat, lon
    del the_grib 
    gc.collect()
    return new_grid
