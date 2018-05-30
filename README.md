# PyDDA
Pythonic multiple doppler package

This package is a rewrite of DDA into a purely Pythonic package for easier integration with Py-ART (and works on Windows!)

The user has an option to adjust strength of data, mass continuity constraints as well as implement a low pass filter. This new version now also has an option to plot a horizontal cross section of a wind barb plot overlaid on a background field from a grid. More documentation to be coming in the next few weeks!

The code here is based off of Potvin et al. (2012) and Shapiro et al. (2009).

Angles.py is from Multidop and was written by Timothy Lang of NASA.

Right now this has been tested on and depends on:

    Python 3.5+

    Py-ART 1.9.0
    
    scipy 1.0.1
    
    numpy 1.13.1
    
    matplotlib 1.5.3
    
## Installation instructions
Right now there is only one method to install PyDDA, which is from source. To
do this, just type in the following commands assuming you have the above 
dependencies installed.

```
git clone https://github.com/rcjackson/PyDDA
cd PyDDA
python setup.py install
```
    
