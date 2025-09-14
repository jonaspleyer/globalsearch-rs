# 5D Levy Function with JAX for Automatic Differentiation
#
# This example demonstrates how to use JAX for automatic differentiation
# with the globalsearch-rs Python bindings
#
# Levy Function
# The Levy function is defined as follows:
#
# $ f(x) = \sin^2(\pi w_1) + \sum_{i=1}^{d-1} (w_i - 1)^2 [1 + 10\sin^2(\pi w_i + 1)] + (w_d - 1)^2 [1 + \sin^2(2\pi w_d)] $
#
# where $w_i = 1 + \frac{x_i - 1}{4}$ for all $i = 1, ..., d$.
#
# The function is defined on the domain $x_i \in [-10, 10]$ for all $i = 1, ..., d$.
# The global minimum is at $f(1, 1, ..., 1) = 0$.
# The function is continuous, differentiable and multimodal.
#
# References:
#
# Surjanovic, S., & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test Functions and Datasets.
# Retrieved from https://www.sfu.ca/~ssurjano/levy.html
#
# Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018).
# JAX: Composable transformations of Python+NumPy programs (Version 0.3.13) [Software]. Retrieved July 2025, from http://github.com/jax-ml/jax

import pyglobalsearch as gs
import jax.numpy as jnp
from jax import grad, jit

print("5D Levy Function Optimization with JAX")
print("=" * 52)


@jit  # JIT compile for maximum performance
def levy_jax(x):
    """5D Levy function implemented in pure JAX using explicit computation"""
    # Transform variables: w_i = 1 + (x_i - 1) / 4
    w1 = 1.0 + (x[0] - 1.0) / 4.0
    w2 = 1.0 + (x[1] - 1.0) / 4.0
    w3 = 1.0 + (x[2] - 1.0) / 4.0
    w4 = 1.0 + (x[3] - 1.0) / 4.0
    w5 = 1.0 + (x[4] - 1.0) / 4.0

    # First term: sin**2(π * w_1)
    term1 = jnp.sin(jnp.pi * w1) ** 2

    # Middle terms: (w_i - 1)² * [1 + 10 * sin**2(π * w_i + 1)] for i = 1 to d-1
    term_mid1 = (w1 - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w1 + 1.0) ** 2)
    term_mid2 = (w2 - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w2 + 1.0) ** 2)
    term_mid3 = (w3 - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w3 + 1.0) ** 2)
    term_mid4 = (w4 - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w4 + 1.0) ** 2)

    # Last term: (w_5 - 1)**2 * [1 + sin**2(2π * w_5)]
    term_last = (w5 - 1.0) ** 2 * (1.0 + jnp.sin(2.0 * jnp.pi * w5) ** 2)

    return term1 + term_mid1 + term_mid2 + term_mid3 + term_mid4 + term_last


# Automatically compute the gradient using JAX and JIT compilation
gradient_jax = jit(grad(levy_jax))


def obj(x) -> float:
    """Objective function wrapper optimized for minimal conversions"""
    result = levy_jax(x)
    return float(result)


def grad_func(x):
    """Gradient function using JAX automatic differentiation"""
    grad_result = gradient_jax(x)
    return grad_result


# x_i ∈ [-10, 10] for all i = 1, ..., 5
bounds_jax = jnp.array(
    [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]
)


def variable_bounds():
    """Variable bounds for the 5D Levy function"""
    return bounds_jax


# Create optimization parameters
params = gs.PyOQNLPParams(
    iterations=1000,
    population_size=5000,
    wait_cycle=10,
    threshold_factor=0.1,
    distance_factor=0.5,
)

print("Optimization Parameters:")
print(f"  Iterations: {params.iterations}")
print(f"  Population size: {params.population_size}")
print(f"  Wait cycle: {params.wait_cycle}")
print(f"  Threshold factor: {params.threshold_factor}")
print(f"  Distance factor: {params.distance_factor}")
print()

# Create the problem with JAX-computed gradient
problem = gs.PyProblem(obj, variable_bounds, grad_func)  # type: ignore

print("Starting optimization...")
print()

# Run optimization with L-BFGS
sol_set = gs.optimize(problem, params, local_solver="LBFGS", seed=0)

# Display results
if sol_set is not None and len(sol_set) > 0:
    print(f"Optimization completed! Found {len(sol_set)} solution(s):")
    print("=" * 50)

    for i, sol in enumerate(sol_set, 1):
        x_opt = sol.x()
        f_opt = sol.fun()

        print(f"Solution #{i}:")
        print(f"  Parameters: {x_opt}")
        print(f"  Objective:  {f_opt:12.8f}")

        # Verify gradient is near zero at optimum
        grad_at_opt = grad_func(jnp.array(x_opt))
        grad_norm = jnp.linalg.norm(grad_at_opt)
        print(f"  Gradient norm: {grad_norm:12.8f}")

        # Check if this is close to the known global minimum [1, 1, 1, 1, 1]
        known_minimum = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        distance_to_optimum = float(jnp.linalg.norm(jnp.array(x_opt) - known_minimum))
        error_sq = float(jnp.square(distance_to_optimum))
        print(f"  Error (squared): {error_sq:.2e}")

        if distance_to_optimum < 0.1:
            print(
                f"  Close to known global minimum (distance: {distance_to_optimum:.6f})"
            )
        else:
            print(f"  Distance to known global minimum: {distance_to_optimum:.6f}")

        print()

else:
    print("No solution found!")
