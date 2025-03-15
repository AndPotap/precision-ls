# Samplers
from .samplers import GaussianSampler, LeastSquaresSampler

# Tasks
from .primitives import Identity, Square, ElementwiseMultiply, Read, Linear

# Algos
from .least_squares import LeastSquares
from .least_squares_icl import LeastSquaresICL
from .explicit_gradient import ExplicitGradient
from .multistep_gd import MultistepGradientDescent
from .odes import ODEOperatorSampler, ODEOperatorICL, ODEOperatorICLFinal


def get_data_sampler(data_name, n_dims, n_points=None, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "least_squares": LeastSquaresSampler,
        "ode_operator": ODEOperatorSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims=n_dims, n_points=n_points, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, n_points, **kwargs):
    task_names_to_classes = {
        # Primitives
        "identity": Identity,
        "square": Square,
        "elementwise_multiply": ElementwiseMultiply,
        "read": Read,
        "linear": Linear,
        # Least squares
        "least_squares": LeastSquares,
        "least_squares_icl": LeastSquaresICL,
        # Explicit gradient
        "explicit_gradient": ExplicitGradient,
        # Multistep gradient descent
        "multistep_gd": MultistepGradientDescent,
        # ODE operator
        "ode_operator_icl": ODEOperatorICL,
        "ode_operator_icl_final": ODEOperatorICLFinal,
    }

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(
            n_dims=n_dims, n_points=n_points, **args, **kwargs
        )
    else:
        print("Unknown task")
        raise NotImplementedError
