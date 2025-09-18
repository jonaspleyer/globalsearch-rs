# Six-Hump Camel Back Function using COBYLA method
# The Six-Hump Camel Back function is defined as follows:
#
# $f(x) = (4 - 2.1 x_1^2 + x_1^4 / 3) x_1^2 + x_1 x_2 + (-4 + 4 x_2^2) x_2^2$
#
# The function is defined on the domain $x_1 \in [-3, 3]$ and $x_2 \in [-2, 2]$.
# The function has two global minima at $f(0.0898, -0.7126) = -1.0316$ and $f(-0.0898, 0.7126) = -1.0316$.
# The function is continuous, differentiable and non-convex.
#
# This example demonstrates the use of COBYLA (Constrained Optimization BY Linear Approximations)
# which is a derivative-free method that can handle constraints.

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray

# Create the optimization parameters
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=300,
    wait_cycle=25,
    threshold_factor=0.1,
    distance_factor=0.5,
)


# Function, variable bounds definitions
# Objective function
def obj(x: NDArray[np.float64]) -> float:
    return (
        4 * x[0] ** 2
        - 2.1 * x[0] ** 4
        + x[0] ** 6 / 3
        + x[0] * x[1]
        - 4 * x[1] ** 2
        + 4 * x[1] ** 4
    )


# Variable bounds
def variable_bounds() -> NDArray[np.float64]:
    return np.array([[-3, 3], [-2, 2]])


# Create the problem
# COBYLA is a derivative-free method, so no gradient is needed
problem = gs.PyProblem(obj, variable_bounds)

# Configuration with custom tolerances
custom_cobyla = gs.builders.cobyla(
    max_iter=450,
    step_size=0.5,
    ftol_rel=1e-6,  # Function relative tolerance
    ftol_abs=1e-8,  # Function absolute tolerance
    xtol_rel=1e-10,  # Parameter relative tolerance
    xtol_abs=1e-12,  # Parameter absolute tolerance
)


# Run optimizations with custom a custom COBYLA configuration
custom_sol = gs.optimize(
    problem, params, local_solver="COBYLA", local_solver_config=custom_cobyla, seed=0
)

# Display results
print(custom_sol)
