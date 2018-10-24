# PyDDA (Pythonic Direct Data Assimilation)
![alt text](https://github.com/openradar/PyDDA/blob/pydda_devel/pydda%20logo.png "Logo Title Text 1")

### A Pythonic Multiple Doppler Radar Wind Retrieval Package

This software is designed to retrieve wind kinematics (u,v,w) in precipitation storm systems from
 one or more Doppler weather radars using three dimensional data assimilation. Other constraints, including 
 background fields (eg reanalysis) can be added. 

This package is a rewrite of the Potvin et al. (2012) and Shapiro et al (2009) wind retrieval techniques into a purely
 Pythonic package for easier integration with Py-ART and Python. This new package uses a faster minimization technique,
  L-BFGS-B, which provides a factor of 2 to 5 speedup versus using the predecessor code, 
   [NASA-Multidop](https://github.com/nasa/MultiDop), as well as a more elegant syntax as well as support for an 
   arbitrary number of radars. The code is also threadsafe and has been tested using HPC tools such as Dask on large 
   (100+ core) clusers. 


The user has an option to adjust strength of data, mass continuity constraints as well as implement a low pass filter. 
This new version now also has an option to plot a horizontal cross section of a wind barb plot overlaid on a background 
field from a grid. 

Angles.py is from Multidop and was written by Timothy Lang of NASA.

Right now this has been tested on and depends on:

    Python 3.5+

    Py-ART 1.9.0
    
    scipy 1.0.1
    
    numpy 1.13.1
    
    matplotlib 1.5.3
    
    cartopy 0.15.1
    
In addition, in order to use the capability to load HRRR data as a constraint, the [cfgrib](https://github.com/ecmwf/cfgrib) package is needed. Since this does not work on Windows, this is an optional depdenency for those who wish to use HRRR data. To install cfgrib, simply do:

    pip install cfgrib
    
=======
## Links to important documentation

1. [Examples](http://openradarscience.org/PyDDA/source/auto_examples/plot_examples.html)
2. [Developer reference guide](http://openradarscience.org/PyDDA/dev_reference/index.html)


## Installation instructions
Right now there is only one method to install PyDDA, which is from source. To
do this, just type in the following commands assuming you have the above 
dependencies installed.

```
git clone https://github.com/rcjackson/PyDDA
cd PyDDA
python setup.py install
```

=======
## Acknowledgments
Core components of the software are adopted from the [Multidop](https://github.com/nasa/MultiDop) package by converting the C code to Python. 

The development of this software is supported by the Climate Model Development and Validation (CMDV) activity which is funded by the Office of Biological and Environmental Research in the US Department of Energy Office of Science.


## References
You must cite these papers if you use PyDDA:

Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49, https://doi.org/10.1175/JTECH-D-11-00019.1

Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
