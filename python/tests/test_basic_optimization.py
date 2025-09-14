"""
Basic optimization tests for pyglobalsearch.
"""

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray


# Function, variable bounds and gradient definitions
def obj(x: NDArray[np.float64]) -> float:
    return x[0] ** 2 + x[1] ** 2


def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * x[0], 2 * x[1]])


def hess(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[2, 0], [0, 2]])


def variable_bounds() -> NDArray[np.float64]:
    return np.array([[-3, 3], [-2, 2]])


# Create optimization parameters
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=500,
    wait_cycle=5,
    threshold_factor=0.75,
    distance_factor=0.1,
)

# Create test problems
problem = gs.PyProblem(obj, variable_bounds)  # No gradient or hessian
problem_grad = gs.PyProblem(obj, variable_bounds, grad)  # With gradient only
problem_full = gs.PyProblem(
    obj, variable_bounds, grad, hess
)  # With gradient and hessian


def test_optimize_lbfgs():
    """Test successful optimization with LBFGS."""
    result = gs.optimize(problem_grad, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_optimize_trustregion():
    """Test successful optimization with TrustRegion."""
    result = gs.optimize(problem_full, params, local_solver="TrustRegion")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_optimize_lbfgs_full():
    """Test LBFGS with problem_full (should work, ignoring Hessian)."""
    result = gs.optimize(problem_full, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_default_local_solver():
    """Test optimization with default local solver (should be LBFGS)."""
    result = gs.optimize(problem_grad, params)  # No local_solver specified
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_single_variable_problem():
    """Test optimization with a single variable."""

    def single_var_obj(x: NDArray[np.float64]) -> float:
        return x[0] ** 2 - 4 * x[0] + 3  # Minimum at x=2, value=-1

    def single_var_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2 * x[0] - 4])

    def single_var_bounds() -> NDArray[np.float64]:
        return np.array([[-5, 5]])

    single_var_problem = gs.PyProblem(
        single_var_obj, single_var_bounds, single_var_grad
    )
    result = gs.optimize(single_var_problem, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert abs(result[0].x()[0] - 2.0) < 1e-6, (
        f"Expected x near 2.0, got {result[0].x()[0]}"
    )
    assert abs(result[0].fun() - (-1.0)) < 1e-6, (
        f"Expected fun near -1.0, got {result[0].fun()}"
    )


def test_solution_format():
    """Test that returned solutions have correct format."""
    result = gs.optimize(problem_grad, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    for solution in result:
        assert hasattr(solution, 'x'), "Solution should have 'x' method"
        assert hasattr(solution, 'fun'), "Solution should have 'fun' method"
        assert isinstance(solution.x(), list), "x() should return a list"
        assert isinstance(solution.fun(), float), "fun() should return a float"
        assert len(solution.x()) == 2, "x() should have 2 elements for this problem"
