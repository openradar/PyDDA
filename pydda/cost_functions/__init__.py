r"""
===========================================
pydda.cost_functions (pydda.cost_functions)
===========================================

.. currentmodule:: pydda.cost_functions

The procedures in this module calculate the individual
cost functions and their gradients. All cost functions 
output a single floating point value. All gradients to cost
functions will output an 1D numpy array whose shape is 3 x 
the total number of grid points *N*. The first *N* points will 
correspond to the gradient of the cost function for each value
of *u*, the second *N* points will correspond to the gradient of the 
cost function for each value of *v*, and the third *N* points will 
correspond to the gradient of the cost function for each value
of *w*. 

In order to calculate the gradients of cost functions, assuming 
that your cost function can be written as a functional in the form
of:

.. math::
    J(\vec{u(x,y,z)}) = \int_{domain} f(\vec{u(x,y,z)}) dxdydz

Then, the gradient :math:`\nabla J` is for each :math:`u_{i} \in {\vec{u}}`:

.. math::
     \frac{\delta J}{\delta u_{i}} = \frac{\delta f}{\delta u_{i}(x,y,z)} - 
     \frac{d}{dx}\frac{\delta f}{\delta u'_{i}(x,y,z)}

So, for a cost function such as:

.. math::
    J(\vec{u}) = \int_{domain} (\vec{u}-\vec{u_{back}})^2 dxdydz

We get for each :math:`u_{i} \in {\vec{u}}`:

.. math::
     \frac{\delta J}{\delta u_{i}} = \frac{\delta f}{\delta u_{i}(x,y,z)} - 
     \frac{d}{dx}\frac{\delta L}{\delta u'_{i}(x,y,z)}

Since :math:`f` does not depend on :math:`u'_{i}(x,y,z)`, we have: 

.. math::
     \frac{\delta J}{\delta u_{i}} = 2(u_{i}-\vec{u_{back}}) - 0

    
     \frac{\delta J}{\delta u_{i}} = 2(u_{i}-\vec{u_{back}})
          
Therefore, in order to add your own custom cost functions for your point
observation, you need to explicitly be able to write both the cost function
and its gradient using the methodology above. One you have implemented both
procedures in Python, they then need to be added to 
:py:mod:`pydda.cost_functions`. 

.. autosummary::
    :toctree: generated/

    J_function
    grad_J
    calculate_radial_vel_cost_function
    calculate_grad_radial_vel
    calculate_mass_continuity
    calculate_mass_continuity_gradient
    calculate_smoothness_cost
    calculate_smoothness_gradient
    calculate_background_cost
    calculate_background_gradient 
    calculate_vertical_vorticity_cost
    calculate_vertical_vorticity_gradient
    calculate_model_cost
    calculate_model_gradient
    calculate_fall_speed
    calculate_point_cost
    calculate_point_gradient
"""


from .cost_functions import calculate_radial_vel_cost_function
from .cost_functions import calculate_fall_speed
from .cost_functions import calculate_grad_radial_vel
from .cost_functions import calculate_mass_continuity
from .cost_functions import calculate_mass_continuity_gradient
from .cost_functions import calculate_smoothness_cost
from .cost_functions import calculate_smoothness_gradient
from .cost_functions import calculate_background_gradient
from .cost_functions import calculate_background_cost
from .cost_functions import calculate_vertical_vorticity_cost
from .cost_functions import calculate_vertical_vorticity_gradient
from .cost_functions import calculate_model_cost
from .cost_functions import calculate_model_gradient
from .cost_functions import calculate_point_cost, calculate_point_gradient
from .cost_functions import J_function, grad_J
