import numpy as np

# Adding jax inport statements
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import pyart

# Added to incorpeate JAX within the cost functions
from . import _cost_functions_jax
from . import _cost_functions_numpy
from . import _cost_functions_tensorflow


def J_function(winds, parameters, ref_series=None, time_series=None, print_cost=False):
    """
    Calculates the total cost function. This typically does not need to be
    called directly as get_dd_wind_field is a wrapper around this function and
    :py:func:`pydda.cost_functions.grad_J`.
    In order to add more terms to the cost function, modify this
    function and :py:func:`pydda.cost_functions.grad_J`.

    Parameters
    ----------
    winds: 1-D float array
        The wind field, flattened to 1-D for f_min. The total size of the
        array will be a 1D array of 3*nx*ny*nz elements.
    parameters: DDParameters
        The parameters for the cost function evaluation as specified by the
        :py:func:`pydda.retrieval.DDParameters` class.

    Returns
    -------
    J: float
        The value of the cost function
    """
    if parameters.Cd > 0:
        num_vars = 6
    else:
        num_vars = 3

    if parameters.engine == "tensorflow":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Tensorflow 2.5 or greater is needed in order to use TensorFlow-based PyDDA!"
            )

        winds = tf.reshape(
            winds,
            (
                num_vars,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        winds = tf.math.maximum(winds, tf.constant([-100.0]))
        winds = tf.math.minimum(winds, tf.constant([100.0]))
        # Had to change to float because Jax returns device array (use np.float_())
        Jvel = _cost_functions_tensorflow.calculate_radial_vel_cost_function(
            parameters.vrs,
            parameters.azs,
            parameters.els,
            winds[0],
            winds[1],
            winds[2],
            parameters.wts,
            rmsVr=parameters.rmsVr,
            weights=parameters.weights,
            coeff=parameters.Co,
        )
        # print("apples Jvel", Jvel)

        if parameters.Cm > 0:
            # Had to change to float because Jax returns device array (use np.float_())
            Jmass = _cost_functions_tensorflow.calculate_mass_continuity(
                winds[0],
                winds[1],
                winds[2],
                parameters.z,
                parameters.dx,
                parameters.dy,
                parameters.dz,
                coeff=parameters.Cm,
            )
        else:
            Jmass = 0

        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            Jsmooth = _cost_functions_tensorflow.calculate_smoothness_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                Cx=parameters.Cx,
                Cy=parameters.Cy,
                Cz=parameters.Cz,
            )
        else:
            Jsmooth = 0

        if parameters.Cb > 0:
            Jbackground = _cost_functions_tensorflow.calculate_background_cost(
                winds[0],
                winds[1],
                parameters.bg_weights,
                parameters.u_back,
                parameters.v_back,
                parameters.Cb,
            )
        else:
            Jbackground = 0

        if parameters.Cv > 0:
            # Had to change to float because Jax returns device array (use np.float_())
            Jvorticity = _cost_functions_tensorflow.calculate_vertical_vorticity_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.Ut,
                parameters.Vt,
                coeff=parameters.Cv,
            )
        else:
            Jvorticity = 0

        if parameters.Cmod > 0:
            Jmod = _cost_functions_tensorflow.calculate_model_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.model_weights,
                parameters.u_model,
                parameters.v_model,
                parameters.w_model,
                coeff=parameters.Cmod,
            )
        else:
            Jmod = 0

        if parameters.Cpoint > 0:
            Jpoint = _cost_functions_tensorflow.calculate_point_cost(
                winds[0],
                winds[1],
                parameters.x,
                parameters.y,
                parameters.z,
                parameters.point_list,
                Cp=parameters.Cpoint,
                roi=parameters.roi,
            )
        else:
            Jpoint = 0
    elif parameters.engine == "scipy":
        winds = np.reshape(
            winds,
            (
                num_vars,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        # Had to change to float because Jax returns device array (use np.float_())
        Jvel = _cost_functions_numpy.calculate_radial_vel_cost_function(
            parameters.vrs,
            parameters.azs,
            parameters.els,
            winds[0],
            winds[1],
            winds[2],
            parameters.wts,
            rmsVr=parameters.rmsVr,
            weights=parameters.weights,
            coeff=parameters.Co,
        )
        # print("apples Jvel", Jvel)

        if parameters.Cm > 0:
            # Had to change to float because Jax returns device array (use np.float_())
            Jmass = _cost_functions_numpy.calculate_mass_continuity(
                winds[0],
                winds[1],
                winds[2],
                parameters.z,
                parameters.dx,
                parameters.dy,
                parameters.dz,
                coeff=parameters.Cm,
            )
        else:
            Jmass = 0

        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            Jsmooth = _cost_functions_numpy.calculate_smoothness_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                Cx=parameters.Cx,
                Cy=parameters.Cy,
                Cz=parameters.Cz,
            )
        else:
            Jsmooth = 0

        if parameters.Cb > 0:
            Jbackground = _cost_functions_numpy.calculate_background_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.bg_weights,
                parameters.u_back,
                parameters.v_back,
                parameters.Cb,
            )
        else:
            Jbackground = 0

        if parameters.Cv > 0:
            # Had to change to float because Jax returns device array (use np.float_())
            Jvorticity = _cost_functions_numpy.calculate_vertical_vorticity_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.Ut,
                parameters.Vt,
                coeff=parameters.Cv,
            )
        else:
            Jvorticity = 0

        if parameters.Cmod > 0:
            Jmod = _cost_functions_numpy.calculate_model_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.model_weights,
                parameters.u_model,
                parameters.v_model,
                parameters.w_model,
                coeff=parameters.Cmod,
            )
        else:
            Jmod = 0

        if parameters.Cpoint > 0:
            Jpoint = _cost_functions_numpy.calculate_point_cost(
                winds[0],
                winds[1],
                parameters.x,
                parameters.y,
                parameters.z,
                parameters.point_list,
                Cp=parameters.Cpoint,
                roi=parameters.roi,
            )
        else:
            Jpoint = 0

        if parameters.Cd > 0:
            Jd = _cost_functions_numpy.calc_advection_diffusion_cost(
                winds[0],
                winds[1],
                winds[2],
                winds[3],
                winds[4],
                winds[5],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.ref_series,
                parameters.time_series,
                coeff=parameters.Cd,
            )
        else:
            Jd = 0
    elif parameters.engine == "jax":
        return J_function_jax(winds, parameters)

    if print_cost is True:
        header_string = "Niter  "
        format_string = "{:7d}".format(int(parameters.Nfeval)) + "|"
        if parameters.Co > 0:
            header_string += "| Jvel     "
            format_string += "{:7.4e}".format(float(Jvel)) + "|"

        if parameters.Cm > 0:
            header_string += "| Jmass    "
            format_string += "{:7.4e}".format(float(Jmass)) + "|"

        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            header_string += "| Js       "
            format_string += "{:7.4e}".format(float(Jsmooth)) + "|"

        if parameters.Cb > 0:
            header_string += "| Jbg      "
            format_string += "{:7.4e}".format(float(Jbackground)) + "|"

        if parameters.Cv > 0:
            header_string += "| Jvort    "
            format_string += "{:7.4e}".format(float(Jvorticity)) + "|"

        if parameters.Cmod > 0:
            header_string += "| Jmodel   "
            format_string += "{:7.4e}".format(float(Jmod)) + "|"

        if parameters.Cpoint > 0:
            header_string += "| Jpoint   "
            format_string += "{:7.4e}".format(float(Jpoint)) + "|"
        if parameters.Cd > 0:
            header_string += "| Jd       "
            format_string += "{:7.4e}".format(float(Jd)) + "|"
        print(header_string)
        print(format_string)

    return Jvel + Jmass + Jsmooth + Jbackground + Jvorticity + Jmod + Jpoint + Jd


def grad_J(winds, parameters, ref_series=None, time_series=None, print_grad=False):
    """
    Calculates the gradient of the cost function. This typically does not need
    to be called directly as get_dd_wind_field is a wrapper around this
    function and :py:func:`pydda.cost_functions.J_function`.
    In order to add more terms to the cost function,
    modify this function and :py:func:`pydda.cost_functions.grad_J`.

    Parameters
    ----------
    winds: 1-D float array
        The wind field, flattened to 1-D for f_min
    parameters: DDParameters
        The parameters for the cost function evaluation as specified by the
        :py:func:`pydda.retrieve.DDParameters` class.
    ref_series: 4-D float array
        The reflectivity (or radial velocity) time series
        for the advection-diffusion cost function.
        The array shape is num_times * nz * ny * nx.
    time_series: 1-D float array
        Time in seconds since the epoch for each timestep in `ref_series`.

    Returns
    -------
    grad: 1D float array
        Gradient vector of cost function
    """
    if parameters.Cd > 0:
        num_vars = 6
    else:
        num_vars = 3
    if parameters.engine == "tensorflow":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Tensorflow 2.5 or greater is needed in order to use TensorFlow-based PyDDA!"
            )
        winds = tf.reshape(
            winds,
            (
                3,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )

        winds = tf.math.maximum(winds, tf.constant([-100.0]))
        winds = tf.math.minimum(winds, tf.constant([100.0]))
        grad = _cost_functions_tensorflow.calculate_grad_radial_vel(
            parameters.vrs,
            parameters.els,
            parameters.azs,
            winds[0],
            winds[1],
            winds[2],
            parameters.wts,
            parameters.weights,
            parameters.rmsVr,
            coeff=parameters.Co,
            upper_bc=parameters.upper_bc,
            lower_bc=parameters.lower_bc,
        )

        if parameters.Cm > 0:
            print(grad.shape)
            add = _cost_functions_tensorflow.calculate_mass_continuity_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.z,
                parameters.dx,
                parameters.dy,
                parameters.dz,
                coeff=parameters.Cm,
                upper_bc=parameters.upper_bc,
                lower_bc=parameters.lower_bc,
            )
            print(add.shape)
            grad = grad + add

        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            grad += _cost_functions_tensorflow.calculate_smoothness_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                Cx=parameters.Cx,
                Cy=parameters.Cy,
                Cz=parameters.Cz,
                upper_bc=parameters.upper_bc,
            )

        if parameters.Cb > 0:
            grad += _cost_functions_tensorflow.calculate_background_gradient(
                winds[0],
                winds[1],
                parameters.bg_weights,
                parameters.u_back,
                parameters.v_back,
                parameters.Cb,
            )

        if parameters.Cv > 0:
            grad += _cost_functions_tensorflow.calculate_vertical_vorticity_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.Ut,
                parameters.Vt,
                coeff=parameters.Cv,
                upper_bc=parameters.upper_bc,
                lower_bc=parameters.lower_bc,
            ).numpy()

        if parameters.Cmod > 0:
            grad += _cost_functions_tensorflow.calculate_model_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.model_weights,
                parameters.u_model,
                parameters.v_model,
                parameters.w_model,
                coeff=parameters.Cmod,
            )

        if parameters.Cpoint > 0:
            grad += _cost_functions_tensorflow.calculate_point_gradient(
                winds[0],
                winds[1],
                parameters.x,
                parameters.y,
                parameters.z,
                parameters.point_list,
                Cp=parameters.Cpoint,
                roi=parameters.roi,
            )
        if parameters.const_boundary_cond is True:
            grad = tf.reshape(
                grad,
                (
                    3,
                    parameters.grid_shape[0],
                    parameters.grid_shape[1],
                    parameters.grid_shape[2],
                ),
            )

            grad = tf.concat(
                [
                    tf.zeros(
                        (
                            1,
                            parameters.grid_shape[0],
                            parameters.grid_shape[1],
                            parameters.grid_shape[2],
                        ),
                        dtype=tf.float32,
                    ),
                    grad[:, :, 1:-1, :],
                    tf.zeros(
                        (
                            1,
                            parameters.grid_shape[0],
                            parameters.grid_shape[1],
                            parameters.grid_shape[2],
                        ),
                        dtype=tf.float32,
                    ),
                ],
                axis=0,
            )
            grad = tf.concat(
                [
                    tf.zeros(
                        (
                            1,
                            parameters.grid_shape[0],
                            parameters.grid_shape[1],
                            parameters.grid_shape[2],
                        ),
                        dtype=tf.float32,
                    ),
                    grad[:, :, :, -1:1],
                    tf.zeros(
                        (
                            1,
                            parameters.grid_shape[0],
                            parameters.grid_shape[1],
                            parameters.grid_shape[2],
                        ),
                        dtype=tf.float32,
                    ),
                ],
                axis=0,
            )
            grad = tf.reshape(grad, [-1])
    elif parameters.engine == "scipy":
        grad = np.zeros((num_vars * np.prod(parameters.grid_shape),), dtype="float32")
        winds = np.reshape(
            winds,
            (
                num_vars,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        final_ind = 3 * np.prod(parameters.grid_shape)
        grad[0:final_ind] = grad[
            0:final_ind
        ] + _cost_functions_numpy.calculate_grad_radial_vel(
            parameters.vrs,
            parameters.els,
            parameters.azs,
            winds[0],
            winds[1],
            winds[2],
            parameters.wts,
            parameters.weights,
            parameters.rmsVr,
            coeff=parameters.Co,
            upper_bc=parameters.upper_bc,
        )
        normV = np.linalg.norm(grad[0:final_ind], 2)
        if parameters.Cm > 0:
            add = _cost_functions_numpy.calculate_mass_continuity_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.z,
                parameters.dx,
                parameters.dy,
                parameters.dz,
                coeff=parameters.Cm,
                upper_bc=parameters.upper_bc,
            )
            normM = np.linalg.norm(add, 2)
            grad[0:final_ind] += add
        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            add = _cost_functions_numpy.calculate_smoothness_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                Cx=parameters.Cx,
                Cy=parameters.Cy,
                Cz=parameters.Cz,
                upper_bc=parameters.upper_bc,
            )
            normS = np.linalg.norm(add, 2)
            grad[0:final_ind] += add

        if parameters.Cb > 0:
            add = _cost_functions_numpy.calculate_background_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.bg_weights,
                parameters.u_back,
                parameters.v_back,
                parameters.Cb,
            )
            normB = np.linalg.norm(add, 2)
            grad[0:final_ind] += add

        if parameters.Cv > 0:
            add = _cost_functions_numpy.calculate_vertical_vorticity_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.Ut,
                parameters.Vt,
                coeff=parameters.Cv,
                upper_bc=parameters.upper_bc,
            )
            normV = np.linalg.norm(add, 2)
            grad[0:final_ind] += add
        if parameters.Cmod > 0:
            add = _cost_functions_numpy.calculate_model_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.model_weights,
                parameters.u_model,
                parameters.v_model,
                parameters.w_model,
                coeff=parameters.Cmod,
            )
            normMod = np.linalg.norm(add, 2)
            grad[0:final_ind] += add

        if parameters.Cpoint > 0:
            add = _cost_functions_numpy.calculate_point_gradient(
                winds[0],
                winds[1],
                parameters.x,
                parameters.y,
                parameters.z,
                parameters.point_list,
                Cp=parameters.Cpoint,
                roi=parameters.roi,
            )
            normP = np.linalg.norm(add, 2)
            grad[0:final_ind] += add
        if parameters.Cd > 0:
            add = _cost_functions_numpy.calc_advection_diffusion_gradient(
                winds[0],
                winds[1],
                winds[2],
                winds[3],
                winds[4],
                winds[5],
                parameters.dx,
                parameters.dy,
                parameters.dz,
                parameters.ref_series,
                parameters.time_series,
                coeff=parameters.Cd,
            )
            normD = np.linalg.norm(add, 2)
            grad += add

        # Let's see if we need to enforce strong boundary conditions
        if parameters.const_boundary_cond is True:
            grad = np.reshape(
                grad,
                (
                    num_vars,
                    parameters.grid_shape[0],
                    parameters.grid_shape[1],
                    parameters.grid_shape[2],
                ),
            )
            grad[:, :, 0, :] = 0
            grad[:, :, -1, :] = 0
            grad[:, :, :, 0] = 0
            grad[:, :, :, -1] = 0
            grad = grad.flatten()
    elif parameters.engine == "jax":
        grad = grad_jax(winds, parameters)
        if parameters.const_boundary_cond is True:
            grad = jnp.reshape(
                grad,
                (
                    num_vars,
                    parameters.grid_shape[0],
                    parameters.grid_shape[1],
                    parameters.grid_shape[2],
                ),
            )
            grad.at[:, :, 0, :].set(0)
            grad.at[:, :, -1, :].set(0)
            grad.at[:, :, :, 0].set(0)
            grad.at[:, :, :, -1].set(0)
            grad = grad.flatten()
        return grad

    if print_grad is True:
        print("|gradJ| = %5.4e" % np.linalg.norm(grad, 2))
        header_string = "Niter  "
        format_string = "{:7d}".format(int(parameters.Nfeval)) + "|"
        if parameters.Co > 0:
            header_string += "| gradJvel "
            format_string += "{:7.4e}".format(float(normV)) + "|"

        if parameters.Cm > 0:
            header_string += "| gJmass   "
            format_string += "{:7.4e}".format(float(normM)) + "|"

        if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
            header_string += "| gradJs   "
            format_string += "{:7.4e}".format(float(normS)) + "|"

        if parameters.Cb > 0:
            header_string += "| gradJbg  "
            format_string += "{:7.4e}".format(float(normB)) + "|"

        if parameters.Cv > 0:
            header_string += "| gJvort   "
            format_string += "{:7.4e}".format(float(normV)) + "|"

        if parameters.Cmod > 0:
            header_string += "| gJmodel  "
            format_string += "{:7.4e}".format(float(normMod)) + "|"

        if parameters.Cpoint > 0:
            header_string += "| gJpoint  "
            format_string += "{:7.4e}".format(float(normP)) + "|"
        if parameters.Cd > 0:
            header_string += "| gradJd   "
            format_string += "{:7.4e}".format(float(normD)) + "|"
        print(header_string)
        print(format_string)
    return grad


def J_function_jax(winds, parameters, print_cost=False):
    if not JAX_AVAILABLE:
        raise ImportError("Jax is needed in order to use the Jax-based PyDDA!")
    if parameters.Cd > 0:
        num_vars = 6
    else:
        num_vars = 3
    winds = jnp.reshape(
        winds,
        (
            num_vars,
            parameters.grid_shape[0],
            parameters.grid_shape[1],
            parameters.grid_shape[2],
        ),
    )
    # Had to change to float because Jax returns device array (use np.float_())
    Jvel = _cost_functions_jax.calculate_radial_vel_cost_function(
        parameters.vrs,
        parameters.azs,
        parameters.els,
        winds[0],
        winds[1],
        winds[2],
        parameters.wts,
        rmsVr=parameters.rmsVr,
        weights=parameters.weights,
        coeff=parameters.Co,
    )

    if parameters.Cm > 0:
        # Had to change to float because Jax returns device array (use np.float_())
        Jmass = _cost_functions_jax.calculate_mass_continuity(
            winds[0],
            winds[1],
            winds[2],
            parameters.z,
            parameters.dx,
            parameters.dy,
            parameters.dz,
            coeff=parameters.Cm,
        )
    else:
        Jmass = 0

    if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
        Jsmooth = _cost_functions_jax.calculate_smoothness_cost(
            winds[0],
            winds[1],
            winds[2],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            Cx=parameters.Cx,
            Cy=parameters.Cy,
            Cz=parameters.Cz,
        )
    else:
        Jsmooth = 0

    if parameters.Cb > 0:
        Jbackground = _cost_functions_jax.calculate_background_cost(
            winds[0],
            winds[1],
            winds[2],
            parameters.bg_weights,
            parameters.u_back,
            parameters.v_back,
            parameters.Cb,
        )
    else:
        Jbackground = 0

    if parameters.Cv > 0:
        # Had to change to float because Jax returns device array (use np.float_())
        Jvorticity = _cost_functions_jax.calculate_vertical_vorticity_cost(
            winds[0],
            winds[1],
            winds[2],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            parameters.Ut,
            parameters.Vt,
            coeff=parameters.Cv,
        )
    else:
        Jvorticity = 0

    if parameters.Cmod > 0:
        Jmod = _cost_functions_jax.calculate_model_cost(
            winds[0],
            winds[1],
            winds[2],
            parameters.model_weights,
            parameters.u_model,
            parameters.v_model,
            parameters.w_model,
            coeff=parameters.Cmod,
        )
    else:
        Jmod = 0

    if parameters.Cpoint > 0:
        Jpoint = _cost_functions_jax.calculate_point_cost(
            winds[0],
            winds[1],
            parameters.x,
            parameters.y,
            parameters.z,
            parameters.point_list,
            Cp=parameters.Cpoint,
            roi=parameters.roi,
        )
    else:
        Jpoint = 0

    if parameters.Cd > 0:
        Jd = _cost_functions_jax.calc_advection_diffusion_cost(
            winds[0],
            winds[1],
            winds[2],
            winds[3],
            winds[4],
            winds[5],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            parameters.ref_series,
            parameters.time_series,
            coeff=parameters.Cd,
        )
    else:
        Jd = 0

    return Jvel + Jsmooth + Jmass + Jmod + Jpoint + Jvorticity + Jbackground + Jd


def grad_jax(winds, parameters, print_grad=False):
    if parameters.Cd > 0:
        num_vars = 6
    else:
        num_vars = 3
    winds = jnp.reshape(
        winds,
        (
            num_vars,
            parameters.grid_shape[0],
            parameters.grid_shape[1],
            parameters.grid_shape[2],
        ),
    )
    grad = jnp.zeros((num_vars * np.prod(parameters.grid_shape),), dtype="float32")

    final_ind = 3 * np.prod(parameters.grid_shape)
    grad = grad.at[0:final_ind].add(
        _cost_functions_jax.calculate_grad_radial_vel(
            parameters.vrs,
            parameters.els,
            parameters.azs,
            winds[0],
            winds[1],
            winds[2],
            parameters.wts,
            parameters.weights,
            parameters.rmsVr,
            coeff=parameters.Co,
            upper_bc=parameters.upper_bc,
        )
    )

    if parameters.Cm > 0:
        add = _cost_functions_jax.calculate_mass_continuity_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.z,
            parameters.dx,
            parameters.dy,
            parameters.dz,
            coeff=parameters.Cm,
            upper_bc=parameters.upper_bc,
        )
        grad = grad.at[0:final_ind].add(add)
    if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
        add = _cost_functions_jax.calculate_smoothness_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            Cx=parameters.Cx,
            Cy=parameters.Cy,
            Cz=parameters.Cz,
            upper_bc=parameters.upper_bc,
        )

        grad = grad.at[0:final_ind].add(add)

    if parameters.Cb > 0:
        add = _cost_functions_jax.calculate_background_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.bg_weights,
            parameters.u_back,
            parameters.v_back,
            parameters.Cb,
        )

        grad = grad.at[0:final_ind].add(add)

    if parameters.Cv > 0:
        add = _cost_functions_jax.calculate_vertical_vorticity_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            parameters.Ut,
            parameters.Vt,
            coeff=parameters.Cv,
            upper_bc=parameters.upper_bc,
        )

        grad = grad.at[0:final_ind].add(add)
    if parameters.Cmod > 0:
        add = _cost_functions_jax.calculate_model_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.model_weights,
            parameters.u_model,
            parameters.v_model,
            parameters.w_model,
            coeff=parameters.Cmod,
        )

        grad = grad.at[0:final_ind].add(add)

    if parameters.Cpoint > 0:
        add = _cost_functions_jax.calculate_point_gradient(
            winds[0],
            winds[1],
            parameters.x,
            parameters.y,
            parameters.z,
            parameters.point_list,
            Cp=parameters.Cpoint,
            roi=parameters.roi,
        )

        grad = grad.at[0:final_ind].add(add)
    if parameters.Cd > 0:
        add = _cost_functions_jax.calc_advection_diffusion_gradient(
            winds[0],
            winds[1],
            winds[2],
            winds[3],
            winds[4],
            winds[5],
            parameters.dx,
            parameters.dy,
            parameters.dz,
            parameters.ref_series,
            parameters.time_series,
            coeff=parameters.Cd,
        )

        grad = grad + add

    return grad


def calculate_fall_speed(grid, refl_field=None, frz=4500.0):
    """
    Estimates fall speed based on reflectivity.

    Uses methodology of Mike Biggerstaff and Dan Betten

    Parameters
    ----------
    Grid: Py-ART Grid
        Py-ART Grid containing reflectivity to calculate fall speed from
    refl_field: str
        String containing name of reflectivity field. None will automatically
        determine the name.
    frz: float
        Height of freezing level in m

    Returns
    -------
    3D float array:
        Float array of terminal velocities

    """
    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name("reflectivity")

    refl = grid[refl_field].values
    grid_z = grid["point_z"].values
    np.zeros(refl.shape)
    A = np.zeros(refl.shape)
    B = np.zeros(refl.shape)
    rho = np.exp(-grid_z / 10000.0)
    A[np.logical_and(grid_z < frz, refl < 55)] = -2.6
    B[np.logical_and(grid_z < frz, refl < 55)] = 0.0107
    A[np.logical_and(grid_z < frz, np.logical_and(refl >= 55, refl < 60))] = -2.5
    B[np.logical_and(grid_z < frz, np.logical_and(refl >= 55, refl < 60))] = 0.013
    A[np.logical_and(grid_z < frz, refl > 60)] = -3.95
    B[np.logical_and(grid_z < frz, refl > 60)] = 0.0148
    A[np.logical_and(grid_z >= frz, refl < 33)] = -0.817
    B[np.logical_and(grid_z >= frz, refl < 33)] = 0.0063
    A[np.logical_and(grid_z >= frz, np.logical_and(refl >= 33, refl < 49))] = -2.5
    B[np.logical_and(grid_z >= frz, np.logical_and(refl >= 33, refl < 49))] = 0.013
    A[np.logical_and(grid_z >= frz, refl > 49)] = -3.95
    B[np.logical_and(grid_z >= frz, refl > 49)] = 0.0148

    fallspeed = A * np.power(10, refl * B) * np.power(1.2 / rho, 0.4)
    print(fallspeed.max())
    del A, B, rho
    return np.ma.masked_invalid(fallspeed)
