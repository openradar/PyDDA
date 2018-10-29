"""
pydda.tests.sample_files
==========================

Sample radar files in a number of formats.  Many of these files
are incomplete, they should only be used for testing, not production.

.. autosummary::
    :toctree: generated/

    EXAMPLE_RADAR0
    EXAMPLE_RADAR1
    SOUNDING_PATH
    LTX_GRID
    MHX_GRID
"""
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

EXAMPLE_RADAR0 = os.path.join(DATA_PATH, 'example_grid_radar0.nc')
EXAMPLE_RADAR1 = os.path.join(DATA_PATH, 'example_grid_radar1.nc')
SOUNDING_PATH = os.path.join(DATA_PATH, 'test_sounding.cdf')
LTX_GRID = os.path.join(DATA_PATH, 'grid_ltx.nc')
MHX_GRID = os.path.join(DATA_PATH, 'grid_mhx.nc')
BERR_GRID = os.path.join(DATA_PATH, 'berrwinds.20060120.004000.nc')
CPOL_GRID = os.path.join(DATA_PATH, 'cpolwinds.20060120.004000.nc')

