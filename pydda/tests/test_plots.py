""" Testing module for PyDDA visualizations """
from matplotlib import use
use('agg')
import pydda
import pyart
import pytest
import matplotlib.pyplot as plt


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(7,7))
    pydda.vis.plot_horiz_xsection_barbs(Grids, None, 'DT', level=6,
                                        w_vel_contours=[3, 6, 9],
                                        barb_spacing_x_km=5.0,
                                        barb_spacing_y_km=15.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_xz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_xz_xsection_barbs(Grids, None, 'DT', level=40,
                                     w_vel_contours=[3, 6, 9],
                                     barb_spacing_x_km=10.0,
                                     barb_spacing_z_km=2.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_yz_xsection_barbs():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_yz_xsection_barbs(Grids, None,'DT', level=40,
                                     w_vel_contours=[1, 3, 5, 7],
                                     barb_spacing_y_km=10.0,
                                     barb_spacing_z_km=2.0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(7,7))
    pydda.vis.plot_horiz_xsection_streamlines(Grids, None, 'DT', level=6,
                                              w_vel_contours=[3, 6, 9])
    return fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_xz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)]
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_xz_xsection_streamlines(Grids, None, 'DT', level=40,
                                           w_vel_contours=[3, 6, 9],
                                           thickness_divisor=5.0)
    plt.ylim([0,15])
    return fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_horiz_yz_xsection_streamlines():
    Grids = [pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0),
             pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)] 
    fig = plt.figure(figsize=(9,5))
    pydda.vis.plot_yz_xsection_streamlines(Grids, None,'DT', level=40,
                                           w_vel_contours=[1, 3, 5, 7])
    return fig
