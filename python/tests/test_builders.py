"""
Tests for the builders module in pyglobalsearch.
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


def test_builders_lbfgs():
    """Test LBFGS created using builders module."""
    # Create line search params first
    morethuente = gs.builders.morethuente(
        c1=1e-5, c2=0.85, width_tolerance=1e-11, bounds=[1e-10, 1e10]
    )

    lbfgs_config = gs.builders.lbfgs(
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_cost=1e-14,
        history_size=6,
        line_search_params=morethuente,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_builders_nelder_mead():
    """Test NelderMead created using builders module."""
    nelder_mead_config = gs.builders.neldermead(
        simplex_delta=0.08,
        sd_tolerance=1e-10,
        max_iter=250,
        alpha=1.2,
        gamma=2.2,
        rho=0.4,
        sigma=0.4,
    )
    result = gs.optimize(
        problem,
        params,
        local_solver="NelderMead",
        local_solver_config=nelder_mead_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-4


def test_builders_trustregion():
    """Test TrustRegion created using builders module."""
    trustregion_config = gs.builders.trustregion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.steihaug(),
        max_iter=120,
        radius=0.8,
        max_radius=80.0,
        eta=0.15,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=trustregion_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_builders_hagerzhang():
    """Test HagerZhang line search created using builders module."""
    hagerzhang_config = gs.builders.hagerzhang(
        delta=0.08,
        sigma=0.92,
        epsilon=1e-7,
        theta=0.45,
        gamma=0.68,
        eta=0.015,
        bounds=[1e-12, 1e12],
    )

    lbfgs_config = gs.builders.lbfgs(
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_cost=1e-12,
        history_size=10,
        line_search_params=hagerzhang_config,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6
