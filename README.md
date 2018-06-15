# PyDDA
![alt text](https://github.com/rcjackson/PyDDA/blob/pydda_devel/pydda%20logo.png "Logo Title Text 1")

Pythonic multiple doppler package

This package is a rewrite of the Potvin et al. (2012) and Shapiro et al (2009) wind retrieval techniques into a purely Pythonic package for easier integration with Py-ART and Python. This new package uses a faster minimization technique, L-BFGS-B, which provides a factor of 2 to 5 speedup versus using Multidop, as well as a more elegant syntax as well as support for an arbitrary number of radars (at least 2).

The user has an option to adjust strength of data, mass continuity constraints as well as implement a low pass filter. This new version now also has an option to plot a horizontal cross section of a wind barb plot overlaid on a background field from a grid. 

The code here is based off of Potvin et al. (2012) and Shapiro et al. (2009).

Angles.py is from Multidop and was written by Timothy Lang of NASA.

Right now this has been tested on and depends on:

    Python 3.5+

    Py-ART 1.9.0
    
    scipy 1.0.1
    
    numpy 1.13.1
    
    matplotlib 1.5.3

## Links to important documentation

1. [Examples](https://rcjackson.github.io/PyDDA/source/auto_examples/plot_examples.html)
2. [Developer reference guide](https://rcjackson.github.io/PyDDA/dev_reference/index.html)

## Installation instructions
Right now there is only one method to install PyDDA, which is from source. To
do this, just type in the following commands assuming you have the above 
dependencies installed.

```
git clone https://github.com/rcjackson/PyDDA
cd PyDDA
python setup.py install
```

## References
You must cite these papers if you use PyDDA:

Potvin, C.K., A. Shapiro, and M. Xue, 2012: Impact of a Vertical Vorticity Constraint in Variational Dual-Doppler Wind Analysis: Tests with Real and Simulated Supercell Data. J. Atmos. Oceanic Technol., 29, 32–49, https://doi.org/10.1175/JTECH-D-11-00019.1

Shapiro, A., C.K. Potvin, and J. Gao, 2009: Use of a Vertical Vorticity Equation in Variational Dual-Doppler Wind Analysis. J. Atmos. Oceanic Technol., 26, 2089–2106, https://doi.org/10.1175/2009JTECHA1256.1
