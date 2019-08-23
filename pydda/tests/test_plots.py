""" Testing module for PyDDA visualizations """
from matplotlib import use
use('agg')
import pydda
import pyart
import pytest
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(7,7))
    pydda.vis.plot_horiz_xsection_barbs(Grids, None, 'reflectivity', level=6,
                                        w_vel_contours=[3, 6, 9],
                                        barb_spacing_x_km=5.0,
                                        barb_spacing_y_km=15.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_xz_xsection_barbs(Grids, None, 'reflectivity', level=40,
                                     w_vel_contours=[3, 6, 9],
                                     barb_spacing_x_km=10.0,
                                     barb_spacing_z_km=2.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_yz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_yz_xsection_barbs(Grids, None, 'reflectivity', level=40,
                                     w_vel_contours=[1, 3, 5, 7],
                                     barb_spacing_y_km=10.0,
                                     barb_spacing_z_km=2.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_barbs_map():
    berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pydda.vis.plot_horiz_xsection_barbs_map(
        [cpol_grid, berr_grid], ax, 'reflectivity', bg_grid_no=-1, 
        level=7, w_vel_contours=[3, 5, 8])
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(7,7))
    pydda.vis.plot_horiz_xsection_streamlines(Grids, None, 'reflectivity',
                                              level=6,
                                              w_vel_contours=[3, 6, 9])
    return fig



@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_streamlines_map():
    berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pydda.vis.plot_horiz_xsection_streamlines_map(
        [cpol_grid, berr_grid], ax=ax, bg_grid_no=-1, 
        level=7, w_vel_contours=[3, 5, 8])
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_xz_xsection_streamlines(Grids, None, 'reflectivity', 
                                           level=40,
                                           w_vel_contours=[3, 6, 9])
    plt.ylim([0,15])
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_yz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)] 
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_yz_xsection_streamlines(Grids, None, 'reflectivity',
                                           level=40,
                                           w_vel_contours=[1, 3, 5, 7])
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_quiver():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(7,7))
    pydda.vis.plot_horiz_xsection_quiver(Grids, None, 'reflectivity', 
                                         level=6,
                                         w_vel_contours=[3, 6, 9],
                                         quiver_spacing_x_km=5.0,
                                         quiver_spacing_y_km=5.0,
                                         quiver_width=0.005,
                                         quiverkey_len=10.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xsection_quiver_map():
    berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pydda.vis.plot_horiz_xsection_quiver_map(
        [cpol_grid, berr_grid], ax=ax, bg_grid_no=-1, 
        level=7, w_vel_contours=[3, 5, 8], quiver_width=0.005,
        quiverkey_len=10.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_xz_xsection_quiver():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_xz_xsection_quiver(Grids, None, 'reflectivity', level=40,
                                      w_vel_contours=[3, 6, 9],
                                      quiver_spacing_x_km=5.0,
                                      quiver_spacing_z_km=1.0,
                                      quiver_width=0.005,
                                      quiverkey_len=10.0)
    plt.ylim([0,15])
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_horiz_yz_xsection_quiver():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_yz_xsection_quiver(Grids, None, 'reflectivity', level=40,
                                      w_vel_contours=[1, 3, 5, 7],
                                      quiver_spacing_y_km=5.0,
                                      quiver_spacing_z_km=1.0,
                                      quiver_width=0.005,
                                      quiverkey_len=10.0)
    return fig

