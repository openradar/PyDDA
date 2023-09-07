"""
========================================
PyDDA: Pythonic Direct Data Assimilation
========================================
"""

from . import cost_functions
from . import retrieval
from . import vis
from . import initialization
from . import tests
from . import constraints

__version__ = "1.5.0"

print("Welcome to PyDDA %s" % __version__)
print("If you are using PyDDA in your publications, please cite:")
print("Jackson et al. (2020) Journal of Open Research Science")
print("Detecting Jax...")
try:
    import jax

    print("Jax engine enabled!")
except ImportError:
    print("Jax is not installed on your system, unable to use Jax engine.")

print("Detecting TensorFlow...")
try:
    import tensorflow

    print("TensorFlow detected. Checking for tensorflow-probability...")
    import tensorflow_probability

    print("TensorFlow-probability detected. TensorFlow engine enabled!")
except (ImportError, AttributeError) as e:
    print(
        "Unable to load both TensorFlow and tensorflow-probability. "
        + "TensorFlow engine disabled."
    )
    print(e)
