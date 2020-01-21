import numpy as np
import pyart
import json

def get_iem_obs(Grid):
    """
    Returns all of the station observations from the Iowa Mesonet for a given Grid in the format
    needed for PyDDA.

    Parameters
    ----------
    Grid

    Returns
    -------

    """

    # First query the database for all of the JSON info for every station
    # Only add stations whose lat/lon are within the Grid's boundaries
    regions = """AF AL_ AI_ AQ_ AG_ AR_ AK AL AM_ AO_ AS_ AR AW_ AU_ AT_ 
         AZ_ BS_ BB_ BY_ BZ_ BJ_ AZ CA CA_AB CO CT DE DZ FL GA HI IA ID IL
         IN KS KY LA MA MD ME
         MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
         WA WI WV WY"""
    # IEM quirk to have Iowa AWOS sites in its own labeled network
    networks = ["AWOS"]
    for region in regions.split():
        networks.append("%s_ASOS" % (region,))

    for network in networks: