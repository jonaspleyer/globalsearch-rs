"""
Tests for advanced features in pyglobalsearch (target_objective, max_time, verbose).
"""

import pyglobalsearch as gs
import numpy as np
import time
from numpy.typing import NDArray


# Function, variable bounds and gradient definitions
def obj(x: NDArray[np.float64]) -> float:
    return x[0] ** 2 + x[1] ** 2


def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * x[0], 2 * x[1]])


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

# Create test problem
problem_grad = gs.PyProblem(obj, variable_bounds, grad)


def test_target_objective_reached():
    """Test that optimization stops when target objective is reached."""
    # Target should be easily reached (global minimum is 0.0)
    target = 0.1
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", target_objective=target
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    # The best solution should be at or below the target
    best_solution = result.best_solution()
    assert best_solution is not None, "Should have a best solution"
    assert best_solution.fun() <= target, (
        f"Best objective {best_solution.fun()} should be <= target {target}"
    )


def test_max_time_parameter():
    """Test that max_time parameter limits optimization time."""
    short_time = 0.1

    start_time = time.time()
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", max_time=short_time
    )
    elapsed_time = time.time() - start_time

    assert result is not None, "Optimization returned None"
    # Note: The actual elapsed time might be slightly longer due to setup overhead
    # but it should be reasonably close to the limit
    assert elapsed_time < short_time + 1.0, (
        f"Elapsed time {elapsed_time} should be close to limit {short_time}"
    )


def test_verbose_parameter():
    """Test that verbose parameter works without errors."""
    # This test mainly ensures verbose doesn't crash the optimization
    result = gs.optimize(problem_grad, params, local_solver="LBFGS", verbose=True)
    assert result is not None, "Optimization with verbose=True returned None"
    assert len(result) > 0, "Should return at least one solution"


def test_target_objective_and_max_time_combined():
    """Test combination of target_objective and max_time."""
    result = gs.optimize(
        problem_grad,
        params,
        local_solver="LBFGS",
        target_objective=0.1,
        max_time=5.0,
        verbose=False,
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    best_solution = result.best_solution()
    assert best_solution is not None, "Should have a best solution"
    # Should reach target quickly (global minimum is 0.0)
    assert best_solution.fun() <= 0.1, (
        f"Best objective {best_solution.fun()} should be <= target 0.1"
    )


def test_target_objective_none():
    """Test that target_objective=None works (should be equivalent to no target)."""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, target_objective=None
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed and no target
    assert len(result1) == len(result2)
    assert abs(result1[0].fun() - result2[0].fun()) < 1e-10


def test_max_time_none():
    """Test that max_time=None works (should be equivalent to no time limit)."""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, max_time=None
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed and no time limit
    assert len(result1) == len(result2)
    assert abs(result1[0].fun() - result2[0].fun()) < 1e-10


def test_verbose_false():
    """Test that verbose=False works (should be equivalent to default)."""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, verbose=False
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed
    assert len(result1) == len(result2)
    assert abs(result1[0].fun() - result2[0].fun()) < 1e-10
