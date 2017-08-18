import numpy as np


def rsl_get_slantr_and_elev(gr, h):
    """
Author: Timothy Lang (timothy.j.lang@nasa.gov)
Given ground range and height, return slant range and elevation.

Inputs
  gr  ground range (km)
  h   height (km)

Outputs
  slantr    slant range
  elev      elevation in degrees

This Python function is adapted from the Radar Software Library routine
RSL_get_slantr_and_elev, written by John Merritt and Dennis Flanigan
    """
    Re = 4.0/3.0 * 6371.1  # Effective earth radius in km.
    rh = h + Re
    slantrsq = Re**2 + rh**2 - (2 * Re * rh * np.cos(gr/Re))
    slantr = np.sqrt(slantrsq)
    elev = np.arccos((Re**2 + slantrsq - rh**2)/(2 * Re * slantr))
    elev = elev * 180.0 / np.pi
    elev -= 90.0
    return slantr, elev


def gc_dist(lat1, lon1, lat2, lon2):
    """
    Input lat1/lon1 and lat2/lon2 as decimal degrees.
    Returns great circle distance in km. Can run in vectorized form!
    """
    re = np.float64(6371.1)  # km
    lat1r = lat1 * np.pi / np.float64(180.0)
    lon1r = lon1 * np.pi / np.float64(180.0)
    lat2r = lat2 * np.pi / np.float64(180.0)
    lon2r = lon2 * np.pi / np.float64(180.0)
    dist = np.arccos(np.sin(lat1r) * np.sin(lat2r) + np.cos(lat1r) *
                     np.cos(lat2r) * np.cos(lon2r-lon1r))*re
    return dist


def gc_bear_array(lat1, lon1, lat2, lon2):
    """
    Input lat1/lon1 and lat2/lon2 as decimal degrees.
    Returns initial bearing (deg) from lat1/lon1 to lat2/lon2.
    """
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    thetar = np.arctan2(np.sin(lon2r-lon1r) * np.cos(lat2r),
                        np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) *
                        np.cos(lat2r) * np.cos(lon2r-lon1r))
    theta = np.rad2deg(thetar)
    return (theta + 360.0) % 360.0


def _add_field_to_object(
        radar, field, field_name='AZ', units='degrees from north',
        long_name='Azimuth', standard_name='Azimuth', dz_name='DT'):
    """
    Adds an unmasked field to the Py-ART radar object.
    """
    field_dict = {'data': np.ma.asanyarray(field),
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  'missing_value': 1.0 * radar.fields[dz_name]['_FillValue']}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar


def add_azimuth_as_field(grid, dz_name='DT', az_name='AZ', bad=-32768):
    """
    Add azimuth field to a Py-ART Grid object. The bearing to each gridpoint
    is computed using the Haversine method and Great Circle approximation.

    Parameters
    ----------
    grid : Py-ART Grid object
        Input Grid object for modification.
    dz_name : str
        Name of the reflectivity field in the Grid.
    az_name : str
        Name of the azimuth field to add to the Grid. DDA engine expects 'AZ'.
    bad : int or float
        Bad data value

    Returns
    -------
    grid : Py-ART Grid object
        Output Grid object with azimuth field added.
    """
    az = gc_bear_array(
        grid.radar_latitude['data'][0], grid.radar_longitude['data'][0],
        grid.point_latitude['data'], grid.point_longitude['data'])
    cond = np.isfinite(az)
    az = np.ma.masked_invalid(az)
    grid = _add_field_to_object(grid, az, dz_name=dz_name, field_name=az_name)
    return grid


def add_elevation_as_field(grid, dz_name='DT', el_name='EL', bad=-32768):
    """
    Add elevation field to a Py-ART Grid object. The elevation to each
    gridpoint is computed using the standard radar beam propagation
    equation. Grid is assumed to reference against 0 m MSL,
    *not* AGL. Ground range computed using the Haversine method.

    Parameters
    ----------
    grid : Py-ART Grid object
        Input Grid object for modification.
    dz_name : str
        Name of the reflectivity field in the Grid.
    el_name : str
        Name of the elevation field to add to Grid. DDA engine expects 'EL'.
    bad : int or float
        Bad data value

    Returns
    -------
    grid : Py-ART Grid object
        Output Grid object with elevation field added.
    """
    gr = gc_dist(
        grid.radar_latitude['data'][0], grid.radar_longitude['data'][0],
        grid.point_latitude['data'], grid.point_longitude['data'])
    h3 = 0.0 * gr
    for i in range(len(grid.z['data'])):
        h3[i, :, :] = grid.z['data'][i] - grid.radar_altitude['data'][0]
    sr, el = rsl_get_slantr_and_elev(gr, h3/1000.0)
    cond = np.isfinite(el)
    el = np.ma.masked_invalid(el)
    grid = _add_field_to_object(grid, el, dz_name=dz_name, field_name=el_name)
    return grid
