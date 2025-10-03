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


def test_parallel_false():
    """Test that parallel=False works (sequential processing, default behavior)."""
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, parallel=False
    )
    assert result is not None, "Optimization with parallel=False returned None"
    assert len(result) > 0, "Should return at least one solution"

    # Should find the global minimum
    best_solution = result.best_solution()
    assert best_solution is not None, "Should have a best solution"
    assert abs(best_solution.fun()) < 1e-8, (
        f"Should find global minimum (0.0), got {best_solution.fun()}"
    )


def test_parallel_true():
    """Test that parallel=True works (parallel processing with rayon)."""
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, parallel=True
    )
    assert result is not None, "Optimization with parallel=True returned None"
    assert len(result) > 0, "Should return at least one solution"

    # Should find the global minimum
    best_solution = result.best_solution()
    assert best_solution is not None, "Should have a best solution"
    assert abs(best_solution.fun()) < 1e-8, (
        f"Should find global minimum (0.0), got {best_solution.fun()}"
    )


def test_parallel_default():
    """Test that default behavior (no parallel parameter) is sequential."""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, parallel=False
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed (default is sequential)
    assert len(result1) == len(result2)
    assert abs(result1[0].fun() - result2[0].fun()) < 1e-10


def test_parallel_none():
    """Test that parallel=None works (should default to False)."""
    result1 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, parallel=None
    )
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, parallel=False
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed
    assert len(result1) == len(result2)
    assert abs(result1[0].fun() - result2[0].fun()) < 1e-10


def test_parallel_reproducibility():
    """Test that results are reproducible with same seed regardless of parallel setting."""
    # Note: With the same seed, both sequential and parallel should give reproducible results
    # but they might differ from each other due to different execution order
    result_seq1 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=123, parallel=False
    )
    result_seq2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=123, parallel=False
    )

    result_par1 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=123, parallel=True
    )
    result_par2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=123, parallel=True
    )

    # Same configuration should give identical results
    assert abs(result_seq1[0].fun() - result_seq2[0].fun()) < 1e-10, (
        "Sequential runs with same seed should be identical"
    )
    assert abs(result_par1[0].fun() - result_par2[0].fun()) < 1e-10, (
        "Parallel runs with same seed should be identical"
    )

    # Both should find the global minimum (though potentially through different paths)
    assert abs(result_seq1[0].fun()) < 1e-8, "Sequential should find global minimum"
    assert abs(result_par1[0].fun()) < 1e-8, "Parallel should find global minimum"
