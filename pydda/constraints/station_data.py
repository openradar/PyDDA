import json
import pandas as pd
import warnings
import numpy as np
import pyart
import time

from datetime import datetime, timedelta
from six import StringIO

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def get_iem_obs(Grid, window=60.):
    """
    Returns all of the station observations from the Iowa Mesonet for a given Grid in the format
    needed for PyDDA.

    Parameters
    ----------
    Grid: pyART Grid
        The Grid to retrieve the station data for.
    window: float
        The window (in minutes) to look for the nearest observation in time.

    Returns
    -------
    station_data: list of dicts
        A list of dictionaries containing the following entries as keys:

        *lat* - Latitude of the site (float)

        *lon* - Longitude of the site (float)

        *u* - u wind at the site (float)

        *v* - v wind at the site (float)

        *w* - w wind at the site (assumed to be 0) (float)

        *site_id* - Station ID (string)

        *x*, *y*, *z* - The (x, y, z) coordinates of the site in the Grid. (floats)
    """

    # First query the database for all of the JSON info for every station
    # Only add stations whose lat/lon are within the Grid's boundaries
    regions = """AF AL_ AI_ AQ_ AG_ AR_ AK AL AM_ AO_ AS_ AR AW_ AU_ AT_ 
         AZ_ BA_ BE_ BB_ BG_ BO_ BR_ BF_ BT_ BS_ BI_ BM_ BB_ BY_ BZ_ BJ_ BW_ AZ CA CA_AB
         CA_BC CD_ CK_ CF_ CG_ CL_ CM_ CO CO_ CN_ CR_ CT CU_ CV_ CY_ CZ_ DE DK_ DJ_ DM_ DO_ 
         DZ EE_ ET_ FK_ FM_ FJ_ FI_ FR_ GF_ PF_ GA_ GM_ GE_ DE_ GH_ GI_ KY_ GB_ GR_ GL_ GD_
         GU_ GT_ GN_ GW_ GY_ HT_ HN_ HK_ HU_ IS_ IN_ ID_ IR_ IQ_ IE_ IL_ IT_ CI_ JM_ JP_ 
         JO_ KZ_ KE_ KI_ KW_ LA_ LV_ LB_ LS_ LR_ LY_ LT_ LU_ MK_ MG_ MW_ MY_ MV_ ML_ CA_MB
         MH_ MR_ MU_ YT_ MX_ MD_ MC_ MA_ MZ_ MM_ NA_ NP_ AN_ NL_ CA_NB NC_ CA_NF NF_ NI_
         NE_ NG_ MP_ KP_ CA_NT NO_ CA_NS CA_NU OM_ CA_ON PK_ PA_ PG_ PY_ PE_ PH_ PN_ PL_
         PT_ CA_PE PR_ QA_ CA_QC RO_ RU_RW_ SH_ KN_ LC_ VC_ WS_ ST_ CA_SK SA_ SN_ RS_ SC_
         SL_ SG_ SK_ SI_ SB_ SO_ ZA_ KR_ ES_ LK_ SD_ SR_ SZ_ SE_ CH_ SY_ TW_ TJ_ TZ_ TH_
         TG_ TO_ TT_ TU TN_ TR_ TM_ UG_ UA_ AE_ UN_ UY_  UZ_ VU_ VE_ VN_ VI_ YE_ CA_YT ZM_ ZW_
         EC_ EG_ FL GA GQ_ HI HR_ IA ID IL IO_ IN KS KH_ KY KM_ LA MA MD ME
         MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SV_ SD TD_ TN TX UT VA VT VG_
         WA WI WV WY"""

    networks = ["AWOS"]
    grid_lon_min = Grid.point_longitude["data"].min()
    grid_lon_max = Grid.point_longitude["data"].max()
    grid_lat_min = Grid.point_latitude["data"].min()
    grid_lat_max = Grid.point_latitude["data"].max()
    for region in regions.split():
        networks.append("%s_ASOS" % (region,))

    site_list = []
    elevations = []
    for network in networks:
        # Get metadata
        uri = ("https://mesonet.agron.iastate.edu/" "geojson/network/%s.geojson"
              ) % (network,)
        data = urlopen(uri)
        jdict = json.load(data)
        for site in jdict["features"]:
            lat = site["geometry"]["coordinates"][1]
            lon = site["geometry"]["coordinates"][0]
            if lat >= grid_lat_min and lat <= grid_lat_max and lon >= grid_lon_min and lon <= grid_lon_max:
                site_list.append((site["properties"]["sid"], site["properties"]["elevation"]))


    # Get the timestamp for each request
    grid_time = datetime.strptime(Grid.time["units"],
                                  "seconds since %Y-%m-%dT%H:%M:%SZ")
    start_time = grid_time - timedelta(minutes=window / 2.)
    end_time = grid_time + timedelta(minutes=window / 2.)

    SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += start_time.strftime("year1=%Y&month1=%m&day1=%d&")
    service += end_time.strftime("year2=%Y&month2=%m&day2=%d&")
    station_obs = []
    for stations, elevations in site_list:
        uri = "%s&station=%s" % (service, stations)
        print("Downloading: %s" % (stations,))
        data = _download_data(uri)
        buf = StringIO()
        buf.write(data)
        buf.seek(0)

        my_df = pd.read_csv(buf, skiprows=5)
        stat_dict = {}
        if len(my_df['lat'].values) == 0:
            warnings.warn(
                "No data available at station %s between time %s and %s" %
                (stations, start_time.strftime('%Y-%m-%d %H:%M:%S'),
                 end_time.strftime('%Y-%m-%d %H:%M:%S')))
        else:
            stat_dict['lat'] = my_df['lat'].values[0]
            stat_dict['lon'] = my_df['lon'].values[0]
            stat_dict['x'], stat_dict['y'] = pyart.core.geographic_to_cartesian(
                stat_dict['lon'], stat_dict['lat'], Grid.get_projparams())
            stat_dict['x'] = stat_dict['x'][0]
            stat_dict['y'] = stat_dict['y'][0]
            stat_dict['z'] = elevations - Grid.origin_altitude["data"][0]
            if my_df['drct'].values[0] == 'M':
                continue
            drct = float(my_df['drct'].values[0])
            s_ms = float(my_df['sknt'].values[0]) * 0.514444
            stat_dict['u'] = -np.sin(np.deg2rad(drct)) * s_ms
            stat_dict['v'] = -np.cos(np.deg2rad(drct)) * s_ms
            stat_dict['site_id'] = stations
            station_obs.append(stat_dict)
        buf.close()

    return station_obs

def _download_data(uri):
    attempt = 0
    while attempt < 6:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""
