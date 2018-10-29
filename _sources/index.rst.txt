.. PyDDA documentation master file, created by
   sphinx-quickstart on Tue May 15 12:19:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PyDDA documentation!
===================================

.. image:: logo.png
    :width: 400px
    :align: center
    :height: 200px
    :alt: alternate text

This is the main page for the documentation of PyDDA. Below are links that 
provide documentation on the installation and use of PyDDA as well as 
description of each of PyDDA's subroutines.

=========================
System Requirements
=========================

This works on any modern version of Linux, Mac OS X, and Windows. For Windows,
HRRR data integration is not supported. In addition, since PyDDA takes advtange
of parallelism we recommend:
::
    An Intel machine with at least 4 cores
    8 GB RAM
    1 GB hard drive space
::

While PyDDA will work on less than this, you may run into performance issues.

=========================
Installation instructions
=========================

The GitHub repository for PyDDA is available at:

`<https://github.com/openradar/PyDDA>`_

Before you install PyDDA, ensure that the following dependencies are installed:
::

    Python 3.5+
    Py-ART 1.9.0+
    scipy 1.0.1+
    numpy 1.13.1+
    matplotlib 1.5.3+
    cartopy 0.16.0+
::

In order to use the HRRR data constraint, cfgrib needs to be installed. `cfgrib 
<http://github.com/ecmwf/cfgrib>`_ currently only works on Mac OS and Linux, so 
this is an optional dependency of PyDDA so that Windows users can still use PyDDA. 
In order to install cfgrib, simply do: 

.. _cfgrib: https://github.com/ecmwf/cfgrib

::

 pip install cfgrib
::

Right now there is only one method to install PyDDA, which is from source. To
do this, just type in the following commands assuming you have the above 
dependencies installed.

::

 git clone https://github.com/rcjackson/PyDDA
 cd PyDDA
 python setup.py install
::
 
Contents:

.. toctree::
   :maxdepth: 3
    
   contributors_guide/index
   source/auto_examples/index
   dev_reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

