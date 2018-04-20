#!/usr/bin/env python3


"""Py-DDA

A package for the multi-Doppler analysis of radar radial velocity data. 

"""


DOCLINES = __doc__.split("\n")

import glob

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


NAME = 'pydda'
MAINTAINER = 'Robert Jackson'
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
LICENSE = 'BSD'
PLATFORMS = "Linux, Windows, OSX"
MAJOR = 0
MINOR = 1
MICRO = 0
#SCRIPTS = glob.glob('scripts/*')
#TEST_SUITE = 'nose.collector'
#TESTS_REQUIRE = ['nose']
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def configuration(parent_package='', top_path=None):
    """ Configuration of PyDDA package. """
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('pydda')
    return config


def setup_package():
    """ Setup of PyDDA  package. """
    setup(
        name=NAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        license=LICENSE,
        platforms=PLATFORMS,
        configuration=configuration,
        include_package_data=True,
        #test_suite=TEST_SUITE,
        #tests_require=TESTS_REQUIRE,
        #scripts=SCRIPTS,
    )

if __name__ == '__main__':
    setup_package()
