#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:45:56 2018

@author: rjackson
"""

""" Setup for PyDDA Subpackages. """

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    """ Configuration of pydda subpackages. """
    config = Configuration('pydda', parent_package, top_path)
    config.add_subpackage('cost_functions')
    config.add_subpackage('retrieval')
    config.add_subpackage('vis')
    config.add_subpackage('initialization')
    config.add_subpackage('tests')
    config.add_subpackage('constraints')
    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
