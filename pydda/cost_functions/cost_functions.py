import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Adding jax import statements
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import pyart

# Added to incorpeate JAX within the cost functions
from . import _cost_functions_jax
from . import _cost_functions_numpy
from . import _cost_functions_tensorflow


def _fluid_mask(parameters):
    """
    The above-terrain mask used to restrict the mass continuity constraint,
    or None when no terrain boundary condition is in use.
    """
    terrain = getattr(parameters, "terrain", None)
    if terrain is None or getattr(parameters, "Cterrain", 0.0) <= 0:
        return None
    return terrain["fluid"]


def J_function(winds, parameters):
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
    # Only the scipy and jax engines support the terrain boundary condition
    Jterrain = 0
    # Only the scipy and jax engines support the VVAD constraint
    Jvad = 0
    if parameters.engine == "tensorflow":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Tensorflow 2.5 or greater is needed in order to use TensorFlow-based PyDDA!"
            )
        winds_input = winds
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
        # Had to change to float because Jax returns device array (use np.float_())
        radial_cache = getattr(parameters, "_radial_eval_cache", None)
        if radial_cache is not None and radial_cache["source_winds"] is winds_input:
            Jvel = radial_cache["cost"]
        else:
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
        winds_input = winds
        winds = np.reshape(
            winds,
            (
                3,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        # Had to change to float because Jax returns device array (use np.float_())
        radial_cache = getattr(parameters, "_radial_eval_cache", None)
        if radial_cache is not None and radial_cache["source_winds"] is winds_input:
            Jvel = radial_cache["cost"]
        else:
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
                parallel=parameters.parallel,
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
                fluid=_fluid_mask(parameters),
            )
        else:
            Jmass = 0

        if parameters.Cterrain > 0:
            Jterrain = _cost_functions_numpy.calculate_terrain_cost(
                winds[0],
                winds[1],
                winds[2],
                parameters.terrain,
                coeff=parameters.Cterrain,
            )
        else:
            Jterrain = 0

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

        if parameters.Cvad > 0:
            Jvad = _cost_functions_numpy.calculate_vad_cost(
                winds[0],
                winds[1],
                parameters.vad_weights,
                parameters.u_vad,
                parameters.v_vad,
                coeff=parameters.Cvad,
            )
        else:
            Jvad = 0
    elif parameters.engine == "jax":
        return J_function_jax(winds, parameters)

    if parameters.Nfeval % 10 == 0:
        header = "Nfeval | Jvel    | Jmass   | Jsmooth |   Jbg   | Jvort   | Jmodel  | Jpoint  |"
        row = (
            "{:7d}".format(int(parameters.Nfeval))
            + "|"
            + "{:9.4f}".format(float(Jvel))
            + "|"
            + "{:9.4f}".format(float(Jmass))
            + "|"
            + "{:9.4f}".format(float(Jsmooth))
            + "|"
            + "{:9.4f}".format(float(Jbackground))
            + "|"
            + "{:9.4f}".format(float(Jvorticity))
            + "|"
            + "{:9.4f}".format(float(Jmod))
            + "|"
            + "{:9.4f}".format(float(Jpoint))
            + "|"
        )
        if parameters.Cterrain > 0:
            header += " Jterr   |"
            row += "{:9.4f}".format(float(Jterrain)) + "|"
        if parameters.Cvad > 0:
            header += " Jvad    |"
            row += "{:9.4f}".format(float(Jvad)) + "|"
        print(header + " Max w  ")
        print(row + "{:9.4f}".format(np.ma.max(np.ma.abs(winds[2]))))

    parameters.Nfeval += 1
    # print("The cost functions print", Jvel + Jmass)

    return (
        Jvel
        + Jmass
        + Jsmooth
        + Jbackground
        + Jvorticity
        + Jmod
        + Jpoint
        + Jterrain
        + Jvad
    )


def grad_J(winds, parameters):
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

    Returns
    -------
    grad: 1D float array
        Gradient vector of cost function
    """
    if parameters.engine == "tensorflow":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Tensorflow 2.5 or greater is needed in order to use TensorFlow-based PyDDA!"
            )
        winds_input = winds
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
        radial_cache = getattr(parameters, "_radial_eval_cache", None)
        if radial_cache is not None and radial_cache["source_winds"] is winds_input:
            grad = tf.identity(radial_cache["gradient"])
        else:
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
                upper_bc_mask=parameters.upper_bc_mask,
                lower_bc=parameters.lower_bc,
            )

        if parameters.Cm > 0:
            grad += _cost_functions_tensorflow.calculate_mass_continuity_gradient(
                winds[0],
                winds[1],
                winds[2],
                parameters.z,
                parameters.dx,
                parameters.dy,
                parameters.dz,
                coeff=parameters.Cm,
                upper_bc=parameters.upper_bc,
                upper_bc_mask=parameters.upper_bc_mask,
                lower_bc=parameters.lower_bc,
            )

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
                upper_bc_mask=parameters.upper_bc_mask,
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
                upper_bc_mask=parameters.upper_bc_mask,
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
                upper_bc=parameters.upper_bc,
                upper_bc_mask=parameters.upper_bc_mask,
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
        winds_input = winds
        winds = np.reshape(
            winds,
            (
                3,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        radial_cache = getattr(parameters, "_radial_eval_cache", None)
        use_radial_cache = (
            radial_cache is not None and radial_cache["source_winds"] is winds_input
        )
        if parameters.parallel and not use_radial_cache:
            futures = []
            with ThreadPoolExecutor() as pool:
                futures.append(
                    pool.submit(
                        _cost_functions_numpy.calculate_grad_radial_vel,
                        parameters.vrs,
                        parameters.els,
                        parameters.azs,
                        winds[0],
                        winds[1],
                        winds[2],
                        parameters.wts,
                        parameters.weights,
                        parameters.rmsVr,
                        parameters.Co,
                        parameters.upper_bc,
                        parameters.upper_bc_mask,
                        True,
                    )
                )
                if parameters.Cm > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_mass_continuity_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.z,
                            parameters.dx,
                            parameters.dy,
                            parameters.dz,
                            parameters.Cm,
                            1,
                            parameters.upper_bc,
                            parameters.upper_bc_mask,
                            parameters.lower_bc,
                            _fluid_mask(parameters),
                        )
                    )
                if parameters.Cterrain > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_terrain_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.terrain,
                            parameters.Cterrain,
                        )
                    )
                if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_smoothness_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.dx,
                            parameters.dy,
                            parameters.dz,
                            parameters.Cx,
                            parameters.Cy,
                            parameters.Cz,
                            parameters.upper_bc,
                            parameters.upper_bc_mask,
                        )
                    )
                if parameters.Cb > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_background_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.bg_weights,
                            parameters.u_back,
                            parameters.v_back,
                            parameters.Cb,
                        )
                    )
                if parameters.Cv > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_vertical_vorticity_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.dx,
                            parameters.dy,
                            parameters.dz,
                            parameters.Ut,
                            parameters.Vt,
                            parameters.Cv,
                            parameters.upper_bc,
                            parameters.upper_bc_mask,
                        )
                    )
                if parameters.Cmod > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_model_gradient,
                            winds[0],
                            winds[1],
                            winds[2],
                            parameters.model_weights,
                            parameters.u_model,
                            parameters.v_model,
                            parameters.w_model,
                            parameters.Cmod,
                        )
                    )
                if parameters.Cpoint > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_point_gradient,
                            winds[0],
                            winds[1],
                            parameters.x,
                            parameters.y,
                            parameters.z,
                            parameters.point_list,
                            parameters.Cpoint,
                            parameters.roi,
                        )
                    )
                if parameters.Cvad > 0:
                    futures.append(
                        pool.submit(
                            _cost_functions_numpy.calculate_vad_gradient,
                            winds[0],
                            winds[1],
                            parameters.vad_weights,
                            parameters.u_vad,
                            parameters.v_vad,
                            parameters.Cvad,
                            parameters.upper_bc,
                            parameters.upper_bc_mask,
                        )
                    )
            grad = sum(f.result() for f in futures)
        else:
            if use_radial_cache:
                grad = radial_cache["gradient"].copy()
            else:
                grad = _cost_functions_numpy.calculate_grad_radial_vel(
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
                    upper_bc_mask=parameters.upper_bc_mask,
                )

            if parameters.Cm > 0:
                grad += _cost_functions_numpy.calculate_mass_continuity_gradient(
                    winds[0],
                    winds[1],
                    winds[2],
                    parameters.z,
                    parameters.dx,
                    parameters.dy,
                    parameters.dz,
                    coeff=parameters.Cm,
                    upper_bc=parameters.upper_bc,
                    upper_bc_mask=parameters.upper_bc_mask,
                    lower_bc=parameters.lower_bc,
                    fluid=_fluid_mask(parameters),
                )

            if parameters.Cterrain > 0:
                grad += _cost_functions_numpy.calculate_terrain_gradient(
                    winds[0],
                    winds[1],
                    winds[2],
                    parameters.terrain,
                    coeff=parameters.Cterrain,
                )

            if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
                grad += _cost_functions_numpy.calculate_smoothness_gradient(
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
                    upper_bc_mask=parameters.upper_bc_mask,
                )

            if parameters.Cb > 0:
                grad += _cost_functions_numpy.calculate_background_gradient(
                    winds[0],
                    winds[1],
                    winds[2],
                    parameters.bg_weights,
                    parameters.u_back,
                    parameters.v_back,
                    parameters.Cb,
                )

            if parameters.Cv > 0:
                grad += _cost_functions_numpy.calculate_vertical_vorticity_gradient(
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
                    upper_bc_mask=parameters.upper_bc_mask,
                )

            if parameters.Cmod > 0:
                grad += _cost_functions_numpy.calculate_model_gradient(
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
                grad += _cost_functions_numpy.calculate_point_gradient(
                    winds[0],
                    winds[1],
                    parameters.x,
                    parameters.y,
                    parameters.z,
                    parameters.point_list,
                    Cp=parameters.Cpoint,
                    roi=parameters.roi,
                )

            if parameters.Cvad > 0:
                grad += _cost_functions_numpy.calculate_vad_gradient(
                    winds[0],
                    winds[1],
                    parameters.vad_weights,
                    parameters.u_vad,
                    parameters.v_vad,
                    coeff=parameters.Cvad,
                    upper_bc=parameters.upper_bc,
                    upper_bc_mask=parameters.upper_bc_mask,
                )

        # Let's see if we need to enforce strong boundary conditions
        if parameters.const_boundary_cond is True:
            grad = np.reshape(
                grad,
                (
                    3,
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
                    3,
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

    if parameters.Nfeval % 10 == 0:
        print("The gradient of the cost functions is", str(np.linalg.norm(grad, 2)))
    return grad


def J_and_grad(winds, parameters):
    """Return the objective and gradient for a single optimizer callback.

    SciPy's L-BFGS-B accepts a callable that returns ``(f, g)`` when
    ``fprime`` is omitted.  Keeping this adapter here gives the optimizer one
    callback per evaluation and provides a single place to add shared
    objective/gradient intermediates in the future.
    """
    if parameters.engine == "tensorflow":
        shaped_winds = tf.reshape(
            winds,
            (
                3,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        radial_cost, radial_gradient = (
            _cost_functions_tensorflow.calculate_radial_vel_cost_and_gradient(
                parameters.vrs,
                parameters.els,
                parameters.azs,
                shaped_winds[0],
                shaped_winds[1],
                shaped_winds[2],
                parameters.wts,
                parameters.weights,
                parameters.rmsVr,
                coeff=parameters.Co,
                upper_bc=parameters.upper_bc,
                upper_bc_mask=parameters.upper_bc_mask,
                lower_bc=parameters.lower_bc,
            )
        )
        previous_cache = getattr(parameters, "_radial_eval_cache", None)
        parameters._radial_eval_cache = {
            "source_winds": winds,
            "cost": radial_cost,
            "gradient": radial_gradient,
        }
        try:
            return J_function(winds, parameters), grad_J(winds, parameters)
        finally:
            if previous_cache is None:
                del parameters._radial_eval_cache
            else:
                parameters._radial_eval_cache = previous_cache

    if parameters.engine == "jax":
        objective = lambda objective_winds: J_function_jax(objective_winds, parameters)
        value, gradient = jax.value_and_grad(objective)(winds)
        gradient = jnp.reshape(
            gradient,
            (
                3,
                parameters.grid_shape[0],
                parameters.grid_shape[1],
                parameters.grid_shape[2],
            ),
        )
        if parameters.lower_bc == 1:
            gradient = gradient.at[2, 0].set(0)
        if parameters.upper_bc == 1:
            gradient = gradient.at[2, -1].set(0)
        elif parameters.upper_bc == 2 and parameters.upper_bc_mask is not None:
            gradient = gradient.at[2].set(
                jnp.where(parameters.upper_bc_mask, 0, gradient[2])
            )
        if parameters.const_boundary_cond is True:
            gradient = gradient.at[:, :, 0, :].set(0)
            gradient = gradient.at[:, :, -1, :].set(0)
            gradient = gradient.at[:, :, :, 0].set(0)
            gradient = gradient.at[:, :, :, -1].set(0)
        return value, gradient.flatten()

    if parameters.engine != "scipy":
        return J_function(winds, parameters), grad_J(winds, parameters)

    shaped_winds = np.reshape(
        winds,
        (
            3,
            parameters.grid_shape[0],
            parameters.grid_shape[1],
            parameters.grid_shape[2],
        ),
    )
    radial_cost, radial_gradient = (
        _cost_functions_numpy.calculate_radial_vel_cost_and_gradient(
            parameters.vrs,
            parameters.els,
            parameters.azs,
            shaped_winds[0],
            shaped_winds[1],
            shaped_winds[2],
            parameters.wts,
            parameters.weights,
            parameters.rmsVr,
            coeff=parameters.Co,
            upper_bc=parameters.upper_bc,
            upper_bc_mask=parameters.upper_bc_mask,
            parallel=parameters.parallel,
        )
    )
    previous_cache = getattr(parameters, "_radial_eval_cache", None)
    parameters._radial_eval_cache = {
        "source_winds": winds,
        "cost": radial_cost,
        "gradient": radial_gradient,
    }
    try:
        return J_function(winds, parameters), grad_J(winds, parameters)
    finally:
        if previous_cache is None:
            del parameters._radial_eval_cache
        else:
            parameters._radial_eval_cache = previous_cache


def J_function_jax(winds, parameters):
    if not JAX_AVAILABLE:
        raise ImportError("Jax is needed in order to use the Jax-based PyDDA!")

    winds = jnp.reshape(
        winds,
        (
            3,
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
            fluid=_fluid_mask(parameters),
        )
    else:
        Jmass = 0

    if parameters.Cterrain > 0:
        Jterrain = _cost_functions_jax.calculate_terrain_cost(
            winds[0],
            winds[1],
            winds[2],
            parameters.terrain,
            coeff=parameters.Cterrain,
        )
    else:
        Jterrain = 0

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

    if parameters.Cvad > 0:
        Jvad = _cost_functions_jax.calculate_vad_cost(
            winds[0],
            winds[1],
            parameters.vad_weights,
            parameters.u_vad,
            parameters.v_vad,
            coeff=parameters.Cvad,
        )
    else:
        Jvad = 0

    return (
        Jvel
        + Jsmooth
        + Jmass
        + Jmod
        + Jpoint
        + Jvorticity
        + Jbackground
        + Jterrain
        + Jvad
    )


def grad_jax(winds, parameters):
    winds = jnp.reshape(
        winds,
        (
            3,
            parameters.grid_shape[0],
            parameters.grid_shape[1],
            parameters.grid_shape[2],
        ),
    )
    grad = _cost_functions_jax.calculate_grad_radial_vel(
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
        upper_bc_mask=parameters.upper_bc_mask,
    )

    if parameters.Cm > 0:
        grad += _cost_functions_jax.calculate_mass_continuity_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.z,
            parameters.dx,
            parameters.dy,
            parameters.dz,
            coeff=parameters.Cm,
            upper_bc=parameters.upper_bc,
            upper_bc_mask=parameters.upper_bc_mask,
            lower_bc=parameters.lower_bc,
            fluid=_fluid_mask(parameters),
        )

    if parameters.Cterrain > 0:
        grad += _cost_functions_jax.calculate_terrain_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.terrain,
            coeff=parameters.Cterrain,
        )

    if parameters.Cx > 0 or parameters.Cy > 0 or parameters.Cz > 0:
        grad += _cost_functions_jax.calculate_smoothness_gradient(
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
            upper_bc_mask=parameters.upper_bc_mask,
        )

    if parameters.Cb > 0:
        grad += _cost_functions_jax.calculate_background_gradient(
            winds[0],
            winds[1],
            winds[2],
            parameters.bg_weights,
            parameters.u_back,
            parameters.v_back,
            parameters.Cb,
        )

    if parameters.Cv > 0:
        grad += _cost_functions_jax.calculate_vertical_vorticity_gradient(
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
            upper_bc_mask=parameters.upper_bc_mask,
        ).numpy()

    if parameters.Cmod > 0:
        grad += _cost_functions_jax.calculate_model_gradient(
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
        grad += _cost_functions_jax.calculate_point_gradient(
            winds[0],
            winds[1],
            parameters.x,
            parameters.y,
            parameters.z,
            parameters.point_list,
            Cp=parameters.Cpoint,
            roi=parameters.roi,
        )

    if parameters.Cvad > 0:
        grad += _cost_functions_jax.calculate_vad_gradient(
            winds[0],
            winds[1],
            parameters.vad_weights,
            parameters.u_vad,
            parameters.v_vad,
            coeff=parameters.Cvad,
            upper_bc=parameters.upper_bc,
            upper_bc_mask=parameters.upper_bc_mask,
        )
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
