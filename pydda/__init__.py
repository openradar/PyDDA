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

__version__ = '1.0.1'

print("Welcome to PyDDA 1.0")
print("Detecting Jax...")
try:
    import jax
    print("Jax engine enabled!")
except ImportError:
    print("Jax is not installed on your system, unable to use Jax engine.")

print("Detecting TensorFlow...")
try:
    import tensorflow
    print("TensorFlow detected.")
    import tensorflow_probability
    print("TensorFlow-probability detected. TensorFlow engine enabled!")
except (ImportError, AttributeError):
    print("Unable to load TensorFlow or tensorflow-probability. " +
            "TensorFlow engine disabled.")

