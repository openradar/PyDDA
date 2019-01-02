"""
===========================
pydda.vis (pydda.vis)
===========================

.. currentmodule:: pydda.vis

A visualization module for plotting generated wind fields. There is basic
support for the visualization of wind fields over a given background field
using barbs, streamlines, and quivers.

.. autosummary::
    :toctree: generated/

    plot_horiz_xsection_barbs
    plot_xz_xsection_barbs
    plot_yz_xsection_barbs
    plot_horiz_xsection_barbs_map
    plot_horiz_xsection_streamlines
    plot_xz_xsection_streamlines
    plot_yz_xsection_streamlines
    plot_horiz_xsection_streamlines_map
    plot_horiz_xsection_quiver
    plot_xz_xsection_quiver
    plot_yz_xsection_quiver
    plot_horiz_xsection_quiver_map

"""


from .barb_plot import plot_horiz_xsection_barbs, plot_xz_xsection_barbs
from .barb_plot import plot_yz_xsection_barbs, plot_horiz_xsection_barbs_map
from .streamline_plot import plot_horiz_xsection_streamlines
from .streamline_plot import plot_horiz_xsection_streamlines_map
from .streamline_plot import plot_xz_xsection_streamlines
from .streamline_plot import plot_yz_xsection_streamlines
from .quiver_plot import plot_horiz_xsection_quiver
from .quiver_plot import plot_horiz_xsection_quiver_map
from .quiver_plot import plot_horiz_xsection_quiver, plot_xz_xsection_quiver
from .quiver_plot import plot_yz_xsection_quiver
