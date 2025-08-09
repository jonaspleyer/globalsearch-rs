"""
Tests for bounds filtering functionality (exclude_out_of_bounds).
"""

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray


def simple_exclude_bounds_objective(x: NDArray[np.float64]) -> float:
    """Simple quadratic function with minimum at [0, 0]."""
    return x[0] ** 2 + x[1] ** 2


def simple_exclude_bounds_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient of the quadratic function."""
    return np.array([2 * x[0], 2 * x[1]])


def simple_exclude_bounds_variable_bounds() -> NDArray[np.float64]:
    """Bounds: x[0] and x[1] both in [-1, 1]."""
    return np.array([[-1.0, 1.0], [-1.0, 1.0]])


def test_exclude_out_of_bounds_false():
    """Test the exclude_out_of_bounds=False functionality."""
    # Create problem
    problem = gs.PyProblem(
        simple_exclude_bounds_objective,
        simple_exclude_bounds_variable_bounds,
        simple_exclude_bounds_gradient,
    )

    # Create parameters with smaller population for faster testing
    test_params = gs.PyOQNLPParams(iterations=50, population_size=100)

    # Test with exclude_out_of_bounds=False (default behavior)
    solutions_default = gs.optimize(
        problem=problem, params=test_params, exclude_out_of_bounds=False
    )

    assert solutions_default is not None, "Optimization returned None"
    assert len(solutions_default) > 0, "Should return at least one solution"


def test_exclude_out_of_bounds_true():
    """Test the exclude_out_of_bounds=True functionality."""
    # Create problem
    problem = gs.PyProblem(
        simple_exclude_bounds_objective,
        simple_exclude_bounds_variable_bounds,
        simple_exclude_bounds_gradient,
    )

    # Create parameters with smaller population for faster testing
    test_params = gs.PyOQNLPParams(iterations=50, population_size=100)

    # Test with exclude_out_of_bounds=True
    solutions_filtered = gs.optimize(
        problem=problem, params=test_params, exclude_out_of_bounds=True
    )

    assert solutions_filtered is not None, "Optimization returned None"
    assert len(solutions_filtered) > 0, "Should return at least one solution"

    # Check that all solutions are within bounds
    for sol in solutions_filtered:
        x = sol["x"]
        assert -1.0 <= x[0] <= 1.0, f"x[0]={x[0]} is out of bounds [-1, 1]"
        assert -1.0 <= x[1] <= 1.0, f"x[1]={x[1]} is out of bounds [-1, 1]"
