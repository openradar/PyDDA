import numpy as np
import pyart
import gc
import os
import tempfile

# We want cfgrib to be an optional dependency to ensure Windows compatibility
try:
    import cfgrib

    CFGRIB_AVAILABLE = True
except ImportError:
    CFGRIB_AVAILABLE = False


from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import interp1d, griddata
from scipy.interpolate import NearestNDInterpolator
from copy import deepcopy

try:
    import cdsapi

    ERA5_AVAILABLE = True
except ImportError:
    ERA5_AVAILABLE = False


def make_initialization_from_era5(
    Grid, file_name=None, vel_field=None, dest_era_file=None
):
    """
    Written by: Hamid Ali Syed and Bobby Jackson
    This function will read ERA 5 in NetCDF format
    and add it to the Py-ART grid specified by Grid.
    PyDDA will automatically download the ERA 5 data
    that you need for the scan. It will chose the
    domain that is enclosed by the analysis grid and
    the time period that is closest to the scan.
    It will then do a Nearest Neighbor interpolation of the
    ERA-5 u and v winds to the analysis grid.

    You need to have the ERA5 CDSAPI and an CDS Copernicus
    account set up in order to use this feature. Go to this
    website for instructions on installing the API and
    setting up your account:

    https://cds.climate.copernicus.eu/api-how-to

    Parameters
    ----------
    Grid: Py-ART Grid
        The input Py-ART Grid to modify.
    file_name: str or None
        The netCDF file containing the ERA 5 data. If the web
        API is experiencing delays, it is better to use it to download the
        file and then refer to it here. If this file does not exist
        PyDDA will use the API to create the file.
    vel_field: str or None
        The name of the velocity field in the Py-ART grid. Set to None to
        have Py-DDA attempt to automatically detect it.
    dest_era_file:
        If this is not None, PyDDA will save the interpolated grid
        into this file.
    Returns
    -------
    new_Grid: Py-ART Grid
        The Py-ART Grid with the ERA5 data added into the "u_era",
        "v_era", and "w_era" fields.

    """
    if vel_field is None:
        vel_field = pyart.config.get_field_name("corrected_velocity")

    if ERA5_AVAILABLE is False and file_name is None:
        raise (
            ModuleNotFoundError,
            (
                "The CDSAPI is not installed. Please go to"
                + "https://cds.climate.copernicus.eu/api-how-to"
                + "in order to use the auto download feature."
            ),
        )

    grid_time = datetime.strptime(
        Grid.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
    )
    hour_rounded_to_nearest_1 = int(round(float(grid_time.hour)))

    if hour_rounded_to_nearest_1 == 24:
        grid_time = grid_time + timedelta(days=1)
        grid_time = datetime(
            grid_time.year,
            grid_time.month,
            grid_time.day,
            0,
            grid_time.minute,
            grid_time.second,
        )
    else:
        grid_time = datetime(
            grid_time.year,
            grid_time.month,
            grid_time.day,
            hour_rounded_to_nearest_1,
            grid_time.minute,
            grid_time.second,
        )

    if file_name is not None:
        if not os.path.isfile(file_name):
            raise FileNotFoundError(file_name + " not found!")

    if file_name is None:
        print("Downloading ERA5 data...")
        # ERA5 data is in pressure coordinates
        # Retrieve u, v, w, and geopotential
        # Geopotential is needed to convert into height coordinates

        N = round(Grid.point_latitude["data"].max(), 2)
        S = round(Grid.point_latitude["data"].min(), 2)
        E = round(Grid.point_longitude["data"].max(), 2)
        W = round(Grid.point_longitude["data"].min(), 2)

        retrieve_dict = {}
        pname = "reanalysis-era5-pressure-levels"
        retrieve_dict["product_type"] = "reanalysis"
        retrieve_dict["format"] = "netcdf"
        retrieve_dict["variable"] = [
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "geopotential",
        ]
        retrieve_dict["pressure_level"] = [
            "1",
            "2",
            "3",
            "5",
            "7",
            "10",
            "20",
            "30",
            "50",
            "70",
            "100",
            "125",
            "150",
            "175",
            "200",
            "225",
            "250",
            "300",
            "350",
            "400",
            "450",
            "500",
            "550",
            "600",
            "650",
            "700",
            "750",
            "775",
            "800",
            "825",
            "850",
            "875",
            "900",
            "925",
            "950",
            "975",
            "1000",
        ]
        retrieve_dict["year"] = grid_time.strftime("%Y")
        retrieve_dict["month"] = grid_time.strftime("%m")
        retrieve_dict["day"] = grid_time.strftime("%d")
        retrieve_dict["time"] = grid_time.strftime("%H:00")
        retrieve_dict["area"] = [N, W, S, E]
        if dest_era_file is not None:
            retrieve_dict["target"] = dest_era_file
            file_name = dest_era_file
        else:
            tfile = tempfile.NamedTemporaryFile()
            retrieve_dict["target"] = tfile.name
            file_name = tfile.name
        server = cdsapi.Client()
        server.retrieve(name=pname, request=retrieve_dict, target=file_name)

    ERA_grid = Dataset(file_name, mode="r")
    base_time = datetime.strptime(
        ERA_grid.variables["time"].units, "hours since %Y-%m-%d %H:%M:%S.%f"
    )

    time_seconds = ERA_grid.variables["time"][:]
    our_time = np.array([base_time + timedelta(seconds=int(x)) for x in time_seconds])
    time_step = np.argmin(np.abs(base_time - grid_time))

    analysis_grid_shape = Grid.fields[vel_field]["data"].shape

    height_ERA = ERA_grid.variables["z"][:]
    u_ERA = ERA_grid.variables["u"][:]
    v_ERA = ERA_grid.variables["v"][:]
    w_ERA = ERA_grid.variables["w"][:]
    lon_ERA = ERA_grid.variables["longitude"][:]
    lat_ERA = ERA_grid.variables["latitude"][:]
    radar_grid_lat = Grid.point_latitude["data"]
    radar_grid_lon = Grid.point_longitude["data"]
    radar_grid_alt = Grid.point_z["data"]
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
        (height_flattened, lat_flattened, lon_flattened), u_flattened, rescale=True
    )
    v_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened), v_flattened, rescale=True
    )
    w_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened), w_flattened, rescale=True
    )
    u_new = u_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    v_new = v_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)
    w_new = w_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    # Free up memory
    ERA_grid.close()

    if "tfile" in locals():
        tfile.close()

    u_field = {}
    u_field["data"] = u_new
    u_field["standard_name"] = "u_wind"
    u_field["long_name"] = "meridional component of wind velocity"
    v_field = {}
    v_field["data"] = v_new
    v_field["standard_name"] = "v_wind"
    v_field["long_name"] = "zonal component of wind velocity"
    w_field = {}
    w_field["data"] = w_new
    w_field["standard_name"] = "w_wind"
    w_field["long_name"] = "vertical component of wind velocity"
    temp_grid = deepcopy(Grid)
    temp_grid.add_field("u", u_field, replace_existing=True)
    temp_grid.add_field("v", v_field, replace_existing=True)
    temp_grid.add_field("w", w_field, replace_existing=True)
    return temp_grid


def make_constant_wind_field(Grid, wind=(0.0, 0.0, 0.0), vel_field=None):
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

    new_Grid: Py-ART Grid
        The Py-ART Grid with the constant wind field added in the u, v, and w
        fields.
    """

    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name("corrected_velocity")
    analysis_grid_shape = Grid.fields[vel_field]["data"].shape

    u = wind[0] * np.ones(analysis_grid_shape)
    v = wind[1] * np.ones(analysis_grid_shape)
    w = wind[2] * np.ones(analysis_grid_shape)
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    u_field = {}
    u_field["data"] = u
    u_field["standard_name"] = "u_wind"
    u_field["long_name"] = "meridional component of wind velocity"
    v_field = {}
    v_field["data"] = v
    v_field["standard_name"] = "v_wind"
    v_field["long_name"] = "zonal component of wind velocity"
    w_field = {}
    w_field["data"] = w
    w_field["standard_name"] = "w_wind"
    w_field["long_name"] = "vertical component of wind velocity"

    temp_grid = deepcopy(Grid)
    temp_grid.add_field("u", u_field, replace_existing=True)
    temp_grid.add_field("v", v_field, replace_existing=True)
    temp_grid.add_field("w", w_field, replace_existing=True)
    return temp_grid


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

    new_Grid: Py-ART Grid
        The Py-ART Grid with the sounding wind field added in the u, v, and w
        fields.
    """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name("corrected_velocity")
    analysis_grid_shape = Grid.fields[vel_field]["data"].shape
    u = np.ones(analysis_grid_shape)
    v = np.ones(analysis_grid_shape)
    w = np.zeros(analysis_grid_shape)
    u_back = profile.u_wind
    v_back = profile.v_wind
    z_back = profile.height
    u_interp = interp1d(z_back, u_back, bounds_error=False, fill_value="extrapolate")
    v_interp = interp1d(z_back, v_back, bounds_error=False, fill_value="extrapolate")
    u_back2 = u_interp(np.asarray(Grid.z["data"]))
    v_back2 = v_interp(np.asarray(Grid.z["data"]))
    for i in range(analysis_grid_shape[0]):
        u[i] = u_back2[i]
        v[i] = v_back2[i]
    u = np.ma.filled(u, 0)
    v = np.ma.filled(v, 0)
    w = np.ma.filled(w, 0)
    u_field = {}
    u_field["data"] = u
    u_field["standard_name"] = "u_wind"
    u_field["long_name"] = "meridional component of wind velocity"
    v_field = {}
    v_field["data"] = v
    v_field["standard_name"] = "v_wind"
    v_field["long_name"] = "zonal component of wind velocity"
    w_field = {}
    w_field["data"] = w
    w_field["standard_name"] = "w_wind"
    w_field["long_name"] = "vertical component of wind velocity"

    temp_grid = deepcopy(Grid)
    temp_grid.add_field("u", u_field, replace_existing=True)
    temp_grid.add_field("v", v_field, replace_existing=True)
    temp_grid.add_field("w", w_field, replace_existing=True)
    return temp_grid


def make_background_from_wrf(Grid, file_path, wrf_time, radar_loc, vel_field=None):
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
    new_Grid: Py-ART Grid
        The Py-ART Grid with the WRF wind field added in the u, v, and w
        fields.
    """
    # Parse names of velocity field
    if vel_field is None:
        vel_field = pyart.config.get_field_name("corrected_velocity")

    analysis_grid_shape = Grid.fields[vel_field]["data"].shape
    u = np.ones(analysis_grid_shape)
    v = np.ones(analysis_grid_shape)
    w = np.zeros(analysis_grid_shape)

    # Load WRF grid
    wrf_cdf = Dataset(file_path, mode="r")
    W_wrf = wrf_cdf.variables["W"][:]
    V_wrf = wrf_cdf.variables["V"][:]
    U_wrf = wrf_cdf.variables["U"][:]
    PH_wrf = wrf_cdf.variables["PH"][:]
    PHB_wrf = wrf_cdf.variables["PHB"][:]
    alt_wrf = (PH_wrf + PHB_wrf) / 9.81

    new_grid_x = Grid.point_x["data"]
    new_grid_y = Grid.point_y["data"]
    new_grid_z = Grid.point_z["data"]

    # Find timestep from datetime
    time_wrf = wrf_cdf.variables["Times"]
    ntimes = time_wrf.shape[0]
    dts_wrf = []
    for i in range(ntimes):
        x = "".join([x.decode() for x in time_wrf[i]])
        dts_wrf.append(datetime.strptime(x, "%Y-%m-%d_%H:%M:%S"))

    dts_wrf = np.array(dts_wrf)
    timestep = np.where(dts_wrf == wrf_time)
    if len(timestep[0]) == 0:
        raise ValueError(("Time " + str(wrf_time) + " not found in WRF file!"))

    x_len = wrf_cdf.__getattribute__("WEST-EAST_GRID_DIMENSION")
    y_len = wrf_cdf.__getattribute__("SOUTH-NORTH_GRID_DIMENSION")
    dx = wrf_cdf.DX
    dy = wrf_cdf.DY
    x = np.arange(0, x_len) * dx - radar_loc[0] * 1e3
    y = np.arange(0, y_len) * dy - radar_loc[1] * 1e3
    z = np.mean(alt_wrf[timestep[0], :, :, :], axis=(0, 2, 3))
    x, y, z = np.meshgrid(x, y, z)
    z = np.squeeze(alt_wrf[timestep[0], :, :, :])

    z_stag = (z[1:, :, :] + z[:-1, :, :]) / 2.0
    x_stag = (x[:, :, 1:] + x[:, :, :-1]) / 2.0
    y_stag = (y[:, 1:, :] + y[:, :-1, :]) / 2.0

    W_wrf = np.squeeze(W_wrf[timestep[0], :, :, :])
    V_wrf = np.squeeze(V_wrf[timestep[0], :, :, :])
    U_wrf = np.squeeze(U_wrf[timestep[0], :, :, :])

    w = griddata(
        (z_stag, y, x), W_wrf, (new_grid_z, new_grid_y, new_grid_x), fill_value=0.0
    )
    v = griddata(
        (z, y_stag, x), V_wrf, (new_grid_z, new_grid_y, new_grid_x), fill_value=0.0
    )
    u = griddata(
        (z, y, x_stag), U_wrf, (new_grid_z, new_grid_y, new_grid_x), fill_value=0.0
    )

    u_field = {}
    u_field["data"] = u
    u_field["standard_name"] = "u_wind"
    u_field["long_name"] = "meridional component of wind velocity"
    v_field = {}
    v_field["data"] = v
    v_field["standard_name"] = "v_wind"
    v_field["long_name"] = "zonal component of wind velocity"
    w_field = {}
    w_field["data"] = w
    w_field["standard_name"] = "w_wind"
    w_field["long_name"] = "vertical component of wind velocity"
    temp_grid = deepcopy(Grid)
    temp_grid.add_field("u", u_field, replace_existing=True)
    temp_grid.add_field("v", v_field, replace_existing=True)
    temp_grid.add_field("w", w_field, replace_existing=True)
    return temp_grid


def make_intialization_from_hrrr(Grid, file_path):
    """
    This function will read an HRRR GRIB2 file and return initial guess
    u, v, and w fields from the model

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
        The shape will be the same shape as the fields in Grid and will
        correspond to the same x, y, and z locations as in Grid.
    """

    if CFGRIB_AVAILABLE is False:
        raise RuntimeError(
            (
                "The cfgrib optional dependency needs to be "
                + "installed for the HRRR integration feature."
            )
        )

    the_grib = cfgrib.open_file(
        file_path, filter_by_keys={"typeOfLevel": "isobaricInhPa"}
    )

    # Load the HRR data and tranform longitude coordinates
    grb_u = the_grib.variables["u"]
    grb_v = the_grib.variables["v"]
    grb_w = the_grib.variables["w"]
    gh = the_grib.variables["gh"]

    lat = the_grib.variables["latitude"].data[:, :]
    lon = the_grib.variables["longitude"].data[:, :]
    lon[lon > 180] = lon[lon > 180] - 360

    # Convert geometric height to geopotential height
    EARTH_MEAN_RADIUS = 6.3781e6
    gh = gh.data[:, :, :]
    height = (EARTH_MEAN_RADIUS * gh) / (EARTH_MEAN_RADIUS - gh)
    height = height - Grid.radar_altitude["data"]

    radar_grid_lat = Grid.point_latitude["data"]
    radar_grid_lon = Grid.point_longitude["data"]
    radar_grid_alt = Grid.point_z["data"]
    lat_min = radar_grid_lat.min()
    lat_max = radar_grid_lat.max()
    lon_min = radar_grid_lon.min()
    lon_max = radar_grid_lon.max()
    lon_r = np.tile(lon, (height.shape[0], 1, 1))
    lat_r = np.tile(lat, (height.shape[0], 1, 1))
    lon_flattened = lon_r.flatten()
    lat_flattened = lat_r.flatten()
    height_flattened = gh.flatten()
    the_box = np.where(
        np.logical_and.reduce(
            (
                lon_flattened >= lon_min,
                lat_flattened >= lat_min,
                lon_flattened <= lon_max,
                lat_flattened <= lat_max,
            )
        )
    )[0]

    lon_flattened = lon_flattened[the_box]
    lat_flattened = lat_flattened[the_box]
    height_flattened = height_flattened[the_box]

    u_flattened = grb_u.data[:, :, :].flatten()
    u_flattened = u_flattened[the_box]
    u_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened), u_flattened, rescale=True
    )
    u_new = u_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    v_flattened = grb_v.data[:, :, :].flatten()
    v_flattened = v_flattened[the_box]
    v_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened), v_flattened, rescale=True
    )
    v_new = v_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    w_flattened = grb_v.data[:, :, :].flatten()
    w_flattened = w_flattened[the_box]
    w_interp = NearestNDInterpolator(
        (height_flattened, lat_flattened, lon_flattened), w_flattened, rescale=True
    )
    w_new = w_interp(radar_grid_alt, radar_grid_lat, radar_grid_lon)

    del grb_u, grb_v, grb_w, lat, lon
    del the_grib
    gc.collect()

    u_field = {}
    u_field["data"] = u_new
    u_field["standard_name"] = "u_wind"
    u_field["long_name"] = "meridional component of wind velocity"
    v_field = {}
    v_field["data"] = v_new
    v_field["standard_name"] = "v_wind"
    v_field["long_name"] = "zonal component of wind velocity"
    w_field = {}
    w_field["data"] = w_new
    w_field["standard_name"] = "w_wind"
    w_field["long_name"] = "vertical component of wind velocity"
    temp_grid = deepcopy(Grid)
    temp_grid.add_field("u", u_field, replace_existing=True)
    temp_grid.add_field("v", v_field, replace_existing=True)
    temp_grid.add_field("w", w_field, replace_existing=True)
    return temp_grid
